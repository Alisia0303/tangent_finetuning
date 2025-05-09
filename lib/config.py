import numpy as np
from easydict import EasyDict as edict



root = edict()
cfg = root

root.prompt_tuning_path = 'config/prompt_tuning.json'

root.run_label = 'merlin'
root.gpu_ids = '0'
root.verbose = False
root.device = None

root.seed = 180

root.model = 'ResNet32'

# Training Dynamics
root.epochs = 2
root.within_batch_update_count = 5
root.n_models = 2
root.learning_rate = 0.001
root.momentum = 0.9
root.weight_decay = 0.0001
root.batch_size_train = 512
root.batch_size_test = 100
root.input_size = 0
root.is_cifar_10 = False
root.is_cifar_100 = False
root.is_mini_imagenet = False

# Placeholders
root.timestamp = 'placeholder'
root.output_dir = 'placeholder'

# Kernels
root.kernels = edict()
root.kernels.dataset_path = ''
root.kernels.batch_size = 32
root.kernels.epochs = 5
root.kernels.learning_rate = 0.001
root.kernels.lr_decay_step = 1100000000
root.kernels.gamma = 0.5
root.kernels.threshold = 0.3
root.kernels.rebuild_dataset_pickle = False
root.kernels.point_estimate = False
root.kernels.encoder = 'ae'
root.kernels.intermediate_test = False
root.kernels.latent_dimension = 2
root.kernels.n_pseudo_weights = 1
root.kernels.n_finetune_src_models = 1
root.kernels.ensembling = edict()
root.kernels.ensembling.min_clf_accuracy = -1
root.kernels.ensembling.max_num_of_models = 50
root.kernels.use_classification_loss = False
root.kernels.use_reconstruction_loss = True
root.kernels.chunking = edict()
root.kernels.chunking.enable = False
root.kernels.chunking.chunk_size = 300
root.kernels.chunking.hidden_size = 100
root.kernels.chunking.last_chunk_size = 0
root.kernels.chunking.num_chunks = 0
root.kernels.attn_module = edict()
root.kernels.attn_module.enable = False
root.kernels.attn_module.temperature = 2000
root.kernels.lora_module = False

# Recall
root.recall = edict()
root.recall.base_model = ''
root.recall.model_path = ''
root.recall.batch_size = 32
root.recall.use_generated_weights = True
root.recall.cumulative_prior = True

root.continual = edict()
root.continual.task = 'permuted_mnist'
root.continual.shuffle_task = False
root.continual.shuffle_datapoints = False
root.continual.rebuild_dataset = False
root.continual.n_tasks = 5
root.continual.n_class_per_task = 0
root.continual.samples_per_task = 1000
root.continual.validation_samples_per_task = 1000
root.continual.epochs = 10
root.continual.n_finetune_epochs = 10
root.continual.finetune_learning_rate = 0.001
root.continual.learning_rate = 0.001
root.continual.batch_size_train = 128
root.continual.batch_size_test = 128

root.continual.method = edict()
root.continual.method.run_merlin = True
root.continual.method.run_ewc = False
root.continual.method.run_gem = False
root.continual.method.run_icarl = False
root.continual.method.run_single_model = False
root.continual.method.run_gss = False
root.continual.method.run_gss_greedy = False

#Cifar-100
root.vision = edict()
root.vision.type = 'class_increment'
root.vision.nb_classes = 100
root.vision.batch_size = 16
root.vision.epochs = 1
root.vision.model = 'vit_base_patch16_224'
root.vision.input_size = 224
root.vision.pretrained = True
root.vision.drop = 0.0
root.vision.drop_path = 0.0
root.vision.opt = 'adam'
root.vision.opt_eps = 1e-8
root.vision.opt_betas = (0.9, 0.999)
root.vision.clip_grad = 1.0
root.vision.momentum = 0.9
root.vision.weight_decay = 0.0
root.vision.reinit_optimizer = True
root.vision.sched = 'constant'
root.vision.lr = 0.03
root.vision.lr_noise = None
root.vision.lr_noise_pct = 0.67
root.vision.lr_noise_std = 1.0
root.vision.warmup_lr = 1e-6
root.vision.min_lr = 1e-5
root.vision.decay_epochs = 30
root.vision.warmup_epochs = 5
root.vision.cooldown_epochs = 10
root.vision.patience_epochs = 10
root.vision.decay_rate = 0.1
root.vision.unscale_lr = True
root.vision.color_jitter = None
root.vision.aa = None
root.vision.smoothing = 0.1
root.vision.train_interpolation = 'bicubic'
root.vision.reprob = 0.0
root.vision.remode = 'pixel'
root.vision.recount = 1
root.vision.data_path = './local_datasets/'
root.vision.dataset = 'Split-CIFAR100'
root.vision.shuffle = False
root.vision.output_dir = './output_cifar/'
root.vision.seed = 42
root.vision.eval = 'store_true'
root.vision.num_workers = 4
root.vision.pin_mem = 'store_true'
root.vision.no_pin_mem = 'store_false'
root.vision.train_mask = True
root.vision.task_inc = False
root.vision.length = 5
root.vision.n_g_prompts = 5
root.vision.sinkhorn_linear = 0
root.vision.top_k = 4
root.vision.M = 10
root.vision.skill_dim = 100
root.vision.initializer = 'uniform'
root.vision.use_prompt_mask = False
root.vision.batchwise_prompt = True
root.vision.predefined_key = ''
root.vision.pull_constraint = True
root.vision.pull_constraint_coeff = 0.1
root.vision.global_pool = 'token'
root.vision.head_type = 'prompt'
root.vision.freeze = ['blocks', 'patch_embed', 'cls_token', 'norm', 'pos_embed']
root.vision.print_freq = 10
root.vision.num_experts = 10

def _merge_a_into_b(a, b):
    """
    Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if not b.__contains__(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """
    Load a config file and merge it into the default options.
    """
    import yaml
    with open(filename, 'r') as f:
        # yaml_cfg = edict(yaml.load(f))
        yaml_cfg = edict(yaml.full_load(f))

    _merge_a_into_b(yaml_cfg, root)
