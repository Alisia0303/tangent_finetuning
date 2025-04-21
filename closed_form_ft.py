from lib.config import cfg
from lib.utils import *
from data.dataset import *
from models import load_clip
from models.clip import build_model
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm.utils import accuracy
from utils.vision import *

import numpy as np
import torch
import random
import random

from tangent_alignment.algos import closed_form_linear_clip

class cfgc(object):
    backbonename = 'ViT-B/16'
    NCTX = 16
    CTXINIT = ''
    CSC = False
    CLASS_TOKEN_POSITION = 'end'

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
            
            output = model((input, text_tokens))
            logits,_ = output
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
cfg.device = torch.device('cuda:1')

# 1. Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Convert images to tensor
    transforms.Normalize((0.5, 0.5, 0.5),  # Mean for R, G, B
                         (0.5, 0.5, 0.5))  # Std for R, G, B
])

# 2. Download and load datasets
train_dataset = datasets.CIFAR10(
    root='../data',
    train=True,
    transform=transform,
    download=True
)

test_dataset = datasets.CIFAR10(
    root='../data',
    train=False,
    transform=transform,
    download=True
)

# 3. Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2
)


# 4. Define CLIP model
clip_cfg = cfgc()
backbone_name = clip_cfg.backbonename
url = load_clip._MODELS[backbone_name]
model_path = load_clip._download(url)
model = torch.jit.load(model_path, map_location="cpu").eval()
clip_model = build_model(model.state_dict())
clip_model.to(cfg.device)
n_parameters = sum(p.numel() for p in clip_model.parameters())
print('number of params:', n_parameters) #150M 

for n, p in clip_model.named_parameters():
    p.requires_grad = False

# 5. Handle text labels
def labels_to_text(labels, class_names):
    """
    Convert a batch of target labels into text prompts.
    
    Args:
        labels (torch.Tensor or list): A batch of class indices.
        class_names (list): A list of class names corresponding to indices.
    
    Returns:
        list: A list of text prompts.
    """
    text_prompts = [f"A photo of class {class_names[int(label.cpu())]}" for label in labels]
    return text_prompts

class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
mapping_class_idx_name = dict()
for idx in range(len(class_names)):
    mapping_class_idx_name[idx] = class_names[idx]

all_classes_text = []
for i in range(len(class_names)):
    text_prompts = f"A photo of class {mapping_class_idx_name[i]}" 
    all_classes_text.append(text_prompts)

# 6. Evaluate the whole CIFAR10 test sets.
text_tokens = load_clip.tokenize(all_classes_text).to(cfg.device)

# 7. Closed form solution
updated_clip_model = closed_form_linear_clip(clip_model, train_loader, text_tokens, cfg)

evaluate(updated_clip_model, test_loader, cfg.device, task_id=0, class_mask=None, text_tokens=text_tokens) #Acc@1 81.730 Acc@5 97.930 loss 2.261