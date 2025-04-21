import itertools
import logging
import os
from pathlib import Path
from typing import List

import hydra
import lightning as L
import lightning.pytorch as pl
import requests
import torch
import torch.nn.functional as F
import torchmetrics
from datasets import load_dataset
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# from peta.metrics.accuracy import Accuracy
# from peta.models.clip import (
#     CLIP_MODELS,
#     freeze_unless_image_model,
#     get_lora_vision_model,
#     load_clip_model,
# )
# from peta.models.LinearizedModel import LinearizedModelWraper
# from peta.optim import CosineAnnealingWithWarmup
# from peta.utils.logging import TitledLog, setup_colorlogging

import copy
import time
from torch.func import jacrev, vmap, vjp

def closed_form_linear_clip(clip_model, train_loader, text_tokens, config):
    """
    Args:
        clip_model: Huggingface CLIP model (or PeftModel wrapping a CLIP model).
        clip_processor: Processor of CLIP model.
        text: List of text captions (length K for K classes).
        text_embeds: Precomputed text embeddings.
        config: Configuration object containing lora_config, slice_size, etc.
    """


    # text_tokens = text_tokens.float().to(config.device)
    # Initialize new CLIP model
    updated_clip_model = copy.deepcopy(clip_model)


    def single_model_forward(param_slice, image, text_tokens):
        # Update param
        state_dict = clip_model.state_dict()
        full_param = state_dict[name].clone()
        full_param[slice_start:slice_end] = param_slice.reshape(slice_size, full_param.shape[1])
        state_dict[name] = full_param

        # Run with updated params
        from torch.func import functional_call
        output = functional_call(clip_model, state_dict, ((image, text_tokens),))

        logits, _ = output  # Shape: [1, num_classes]
        return logits.squeeze(0).softmax(dim=0)  # Shape: [num_classes]


    
    # def single_vjp(param_slice, input, text_tokens, z_vector):
    #     _, pullback = vjp(single_model_forward, param_slice, input, text_tokens)
    #     vjp_result = pullback(z_vector)[0]
    #     return vjp_result # return A^T @ z

    # Iterate over parameters
    for (name, param), (name_clone, param_clone) in zip(
        clip_model.named_parameters(), 
        updated_clip_model.named_parameters()
    ):
        for target in config.lora_config.target_modules:
            if target in name:
                print(name)
                slice_size = config.slice_size
                num_slices = param.shape[0] // slice_size
                slice_param_size = slice_size * param.shape[1]
                
                # Stop early
                for slice_idx in range(num_slices):
                    if slice_idx == 10: # after q_proj
                        break

                    start_time = time.time()
                    print(f"Slice: {slice_idx+1}/{num_slices}")
                    slice_start = slice_idx * slice_size
                    slice_end = slice_start + slice_size

                    global_At_A = torch.zeros(slice_param_size, slice_param_size).to(config.device)
                    global_At_b = torch.zeros(slice_param_size).to(config.device)
                    for batch_idx, batch in enumerate(tqdm(train_loader)):
                        images, labels = batch
                        images = images.float().to(config.device)
                        labels = F.one_hot(labels.squeeze(), num_classes=10).float().to(config.device)
                        batch_size, output_dim = labels.shape  # output_dim = num_classes

                        with torch.no_grad():
                            outputs = clip_model((images, text_tokens))
                            logits, _ = outputs
                            logits = logits.softmax(dim=1).detach()

                        ##################### Compute Jacobian using vmap
                        param_slice = param[slice_start:slice_end].reshape(-1)  # Shape: (slice_param_size,)
                        jacobian_fn = jacrev(single_model_forward, argnums=0)
                        # jacobian_vmap = vmap(jacobian_fn, in_dims=(None, 0, None))  # Vectorize over batch dim (0) of pixel_values (second parameter of jacobian_fn)
                        jacobian = jacobian_fn(param_slice, images, text_tokens)
                        A_matrix = jacobian.reshape(batch_size * output_dim, slice_param_size)

                        # Compute At_b
                        b_vector = (logits - labels).flatten().half()
                        global_At_A.add_(A_matrix.T @ A_matrix) # A.T@ A should not be full rank since batch_size * output_dim < slice_param_size
                        global_At_b.add_(A_matrix.T @ b_vector)

                        # Delete all intermediate tensors
                        del A_matrix, b_vector, jacobian, param_slice, jacobian_fn
                        del images, labels, logits, outputs
                        # Set variables to None to ensure no references persist
                        A_matrix, b_vector, jacobian, param_slice, jacobian_fn = None, None, None, None, None
                        images, labels, logits, outputs = None, None, None, None
                        # Clear the computational graph by detaching tensors (if any still require gradients)
                        if global_At_A.requires_grad:
                            global_At_A = global_At_A.detach()
                        if global_At_b.requires_grad:
                            global_At_b = global_At_b.detach()
                        # Clear GPU cache
                        torch.cuda.empty_cache()

                    # After processing all batches for this slice, solve the system
                    global_At_A = global_At_A + 1e-3 * torch.eye(global_At_A.shape[0]).to(config.device)
                    
                    # Compute eigendecomposition
                    eigenvalues, eigenvectors = torch.linalg.eigh(global_At_A)

                    # Sort eigenvalues and eigenvectors in descending order
                    idx = eigenvalues.argsort(descending=True)
                    eigenvalues = eigenvalues[idx]
                    eigenvectors = eigenvectors[:, idx]
                    
                    if config.target_rank > 0:
                        # Project At_b onto eigenvectors
                        a_coeff = eigenvectors.T @ global_At_b
                        
                        # Selection criterion: a_coeff^2 / eigenvalues
                        selection_criterion = (a_coeff ** 2) / eigenvalues
                        
                        # Sort by selection criterion in descending order
                        sorted_indices = torch.argsort(selection_criterion, descending=True)

                        # Greedily select eigenvectors based on sorted criterion
                        target_rank = config.target_rank if hasattr(config, 'target_rank') else min(8, num_slices)
                        cumulative_rank = 0
                        selected_indices = []

                        for idx in sorted_indices:
                            # Add the eigenvector corresponding to this index
                            selected_indices.append(idx.item())

                            # Compute temporary solution with the selected eigenvectors
                            E_t_temp = eigenvectors[:, selected_indices]
                            S_t_inv_temp = torch.diag(1.0 / eigenvalues[selected_indices])
                            temp_solution = E_t_temp @ S_t_inv_temp @ (E_t_temp.T @ global_At_b)

                            # Reshape to check rank
                            temp_matrix = temp_solution.reshape(slice_size, param.shape[1])
                            rank = torch.linalg.matrix_rank(temp_matrix)

                            cumulative_rank += rank
                            if cumulative_rank >= target_rank:
                                break
                        
                        # Compute final closed-form solution with selected components
                        E_t = eigenvectors[:, selected_indices]  # Selected eigenvectors
                        S_t_inv = torch.diag(1.0/eigenvalues[selected_indices])  # Inverse of selected eigenvalues
                        
                        # w_update = E_t @ S_t^-1 @ E_t^T @ global_At_b
                        w_update = E_t @ S_t_inv @ (E_t.T @ global_At_b)
                    
                    else:
                        # import pdb; pdb.set_trace()
                        w_update = eigenvectors @ torch.diag(1.0/eigenvalues) @ (eigenvectors.T @ global_At_b)
                    
                    # Reshape to match parameter dimensions
                    w_update = w_update.reshape(slice_size, param.shape[1])
                    
                    # Update the parameter in the cloned model
                    with torch.no_grad():
                        if param_clone.data[slice_start:slice_end].shape == w_update.shape:
                            param_clone.data[slice_start:slice_end] += w_update
                        else:
                            print(f"Shape mismatch: {param_clone.data[slice_start:slice_end].shape} vs {w_update.shape}")

                        # param_clone.data[slice_start:slice_end] +=0
                    
                    # Clean up
                    del global_At_A, global_At_b, eigenvalues, eigenvectors, w_update
                    torch.cuda.empty_cache()

                    if config.target_rank > 0:
                        print(f"Slice {slice_idx+1} completed. Final rank: {cumulative_rank}")
                    else:
                        print("Full rank solution")
                    print(f"Time taken: {time.time() - start_time:.2f} seconds")

    return updated_clip_model