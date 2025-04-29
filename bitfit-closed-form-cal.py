from lib.config import cfg
from lib.utils import *
from data.dataset import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm.utils import accuracy
from utils.vision import *
from timm.models import create_model
from torch.func import jacrev, functional_call
from pathlib import Path

import numpy as np
import torch
import random
import random
import models.model_register
import copy
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vit = model

    def forward(self, X, c):
        out = self.vit(X)
        logits = out["logits"]
        logits = logits.softmax(dim=1)
        # c_logits = logits[:,c] #in case of computing a specific output logit Jacobian
        c_logits = logits
        return c_logits
    
def wrapped_g(params, X, c=0):
    """Wraps the neural network g while keeping its structure intact using functional_call."""
    return functional_call(net_g, params, (X,c)) # Use the model with external parameters

@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader, 
            device, task_id=-1, class_mask=None, text_tokens=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    # switch to evaluation mode
    model.eval()

    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, 10, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            logits = model(input)["logits"]
            loss = criterion(logits, target)
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def set_seed(seed):
    cfg.seed = seed
    torch.cuda.manual_seed_all(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

set_seed(cfg.seed)
cfg.device = torch.device('cuda:0')

# Download & prepare CIFAR10 dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Convert images to tensor
    transforms.Normalize((0.5, 0.5, 0.5),  # Mean for R, G, B
                         (0.5, 0.5, 0.5))  # Std for R, G, B
])

train_dataset = datasets.CIFAR10(
    root='./local_datasets',
    train=True,
    transform=transform,
    download=True
)

test_dataset = datasets.CIFAR10(
    root='./local_datasets',
    train=False,
    transform=transform,
    download=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=2
)

model = create_model(
    cfg.vision.model,
    pretrained=cfg.vision.pretrained,
    num_classes=10,
    drop_rate=cfg.vision.drop,
    drop_path_rate=cfg.vision.drop_path,
    drop_block_rate=None,
    head_type=cfg.vision.head_type,
)

model.to(cfg.device)  
model_head = torch.load("fc_layer.pth") # Load a pre-tuned classification head
model.head.load_state_dict(model_head)
for n,p in model.named_parameters():
    p.requires_grad = False

n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('number of params:', n_parameters) 

# Now I group data in 10 classes. In the current implementation, it doesn't need this step, but if we would like to 
# compute Jacobian for each logit (for saving GPU memory), it would help.
train_split_indices = [[] for _ in range(10)]
split_datasets = []
batch_size = 16
dataloader_cls = []

for k in range(len(train_dataset.targets)):
    train_split_indices[int(train_dataset.targets[k])].append(k)

for cls in range(10):
    subset_train = Subset(train_dataset, train_split_indices[cls])
    sampler_train = torch.utils.data.RandomSampler(subset_train)
    data_loader_train = torch.utils.data.DataLoader(
      subset_train, sampler=sampler_train,
      batch_size=batch_size, #args.batch_size,
      num_workers=4 #args.num_workers,
      # pin_memory=args.pin_mem,
    )
    dataloader_cls.append(data_loader_train)

net_g = Net(model)
net_g.to(cfg.device)
net_g.eval()

tuning_params = dict()
for block_id in range(12):
    name = f'vit.blocks.{block_id}.attn.qkv.bias'
    tuning_params[name] = net_g.vit.blocks[block_id].attn.qkv.bias

jacobian_fn = jacrev(wrapped_g, argnums = 0) 
global_At_A = dict()
global_At_b = dict()

for layer in tuning_params.keys():
    At_A = torch.zeros((2304, 2304)) #2304 is the size of each bias layer weight
    At_b = torch.zeros((2304))
    global_At_A[layer] = At_A
    global_At_b[layer] = At_b

for cls in range(10):
    start_time = time.time()
    for images, labels in dataloader_cls[cls]:
        images = images.to(cfg.device)
        labels = labels.to(cfg.device)
        
        batch_size = labels.size(0)
        with torch.no_grad():
            logits = net_g(images, cls)
            logits = logits.detach().cpu()
    
        J = jacobian_fn(tuning_params, images, cls) 

        J_cpu = {k: v.detach().cpu() for k, v in J.items()}
        labels = F.one_hot(labels, num_classes=10).float().cpu()
        b = (logits - labels).flatten() 
        
        for name in tuning_params.keys():
            A = J_cpu[name].reshape(batch_size*10, -1)
            global_At_A[name].add_(A.T @ A)
            global_At_b[name].add_(A.T @ b)
            
        del J, J_cpu, A, b, images, logits
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.4f} seconds")
        
w_updates = dict()
for layer in global_At_A.keys():
    w_updates[layer] = torch.linalg.solve(global_At_A[layer], global_At_b[layer])  # ((A.T @ A)^-1) @ (A.T @ b)

# Copying updated bias weights to pretrained model
updated_model = copy.deepcopy(model)
# Update the layer's weight
for block_id in range(12):
    name = f'vit.blocks.{block_id}.attn.qkv.bias'
    with torch.no_grad():
        updated_model.blocks[block_id].attn.qkv.bias.copy_(w_updates[name]) 

# Finally, evaluate the new model
print(evaluate(updated_model, test_loader, 
            cfg.device, task_id=0, class_mask=None))

# Save some computed things for later debug
Path("notebooks/logs/ViT_12_bias/").mkdir(parents=True, exist_ok=True)
torch.save(w_updates, 'notebooks/logs/ViT_12_bias/w_updates.pth')
torch.save(global_At_A, 'notebooks/logs/ViT_12_bias/global_AtA.pth')
torch.save(global_At_b, 'notebooks/logs/ViT_12_bias/global_Atb.pth')



