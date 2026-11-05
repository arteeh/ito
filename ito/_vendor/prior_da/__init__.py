import os
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from typing import Union, Optional
import time

from .depth_anything_v2 import build_backbone
from .depth_completion import DepthCompletion
from .utils import (
    depth2disparity, 
    disparity2depth,
    Arguments
)

class PriorDepthAnything(nn.Module):
    VERSIONS = {
        '1.0': ('', 'error'),
        '1.1': ('_1_1', 'spmask')
    }

    def __init__(self, 
        device: str = 'cuda:0', 
        version: str = '1.1',
        mde_dir: Optional[str] = None,
        ckpt_dir: Optional[str] = None,
        frozen_model_size: Optional[str] = None, 
        conditioned_model_size: Optional[str] = None,
        coarse_only: bool = False
    ):
        super(PriorDepthAnything, self).__init__()
        self.args = Arguments()
        postfix, extra_condition = self.VERSIONS[version]
        self.args.extra_condition = extra_condition
        
        self.device = device

        """ 
        For inference stability, we set the output coarse/fine globally. 
        TODO : You can easily modify the code to specify the model to output coarse/fine depth sample-wisely.
        """
        self.coarse_only = coarse_only
        if frozen_model_size:
            self.args.frozen_model_size = frozen_model_size
        if conditioned_model_size:
            self.args.conditioned_model_size = conditioned_model_size
        
        ## Frozon MDE loading.
        if self.args.frozen_model_size in ['vitg']:
            raise ValueError(f'{self.args.frozen_model_size} coming soon...')
        fmde_name = f'depth_anything_v2_{self.args.frozen_model_size}.pth' # Download model checkpoints
        fmde_path = os.path.join(mde_dir, fmde_name)
        print(f"Loading pretrained fmde from {fmde_path}...")
        
        # Initialize Frozon-MDE.
        self.completion = DepthCompletion.build(args=self.args, fmde_path=fmde_path, device=device)
        
        ## Conditioned MDE loading.
        if not coarse_only:
            if self.args.conditioned_model_size in ['vitl', 'vitg']:
                raise ValueError(f'{self.args.conditioned_model_size} coming soon...')
        
            # Initialize and load preptrained `prior-depth-anything` models.
            model = build_backbone(
                depth_size=self.args.conditioned_model_size, 
                encoder_cond_dim=3
            )
            model.construct_aux_layers()

            self.model = self.load_checkpoints(model, ckpt_dir, postfix, self.device).eval()

    
    def load_checkpoints(self, model, ckpt_dir, postfix='', device='cuda:0'):
        ckpt_name = f'prior_depth_anything_{self.args.conditioned_model_size}{postfix}.pth'
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        print(f"Loading checkpoint from {ckpt_path}...")
        
        state_dict = torch.load(ckpt_path, map_location='cpu')
        
        new_state_dict = OrderedDict()
        for key, value in state_dict['model'].items():
            new_key = key.replace("module.", "")
            new_state_dict[new_key] = value
        
        model.load_state_dict(new_state_dict)
        model = model.to(device)
        return model
        
    def forward(self, 
            images: torch.Tensor, 
            sparse_depths: torch.Tensor, 
            sparse_masks: torch.Tensor, 
            cover_masks: torch.Tensor = None, 
            prior_depths: torch.Tensor = None, 
            geometric_depths: torch.Tensor = None, 
            pattern: Optional[str] = None
        ):
        """ To facilitate further research, we batchify the forward process. """
        ##### Coarse stage. #####
        completed_maps = self.completion(
            images=images, 
            sparse_depths=sparse_depths, 
            sparse_masks=sparse_masks, 
            cover_masks=cover_masks, 
            prior_depths=prior_depths, 
            pattern=pattern,
            geometric_depths=geometric_depths
        )
        
        # knn-aligned depths
        comp_cond = completed_maps['scaled_preds'].unsqueeze(1)
        if self.coarse_only:
            coarse_depths = disparity2depth(comp_cond)
            return coarse_depths
        # Global Scale-Shift aligned depths.
        global_cond = completed_maps['global_preds'].unsqueeze(1)
        
        ##### Fine stage. #####
        if self.args.normalize_depth:
            # Obtain the value of norm params.
            masked_min, denom = self.zero_one_normalize(sparse_depths, sparse_masks, affine_only=True)
            
            global_depths = (disparity2depth(global_cond) - masked_min) / denom
            global_cond = depth2disparity(global_depths)
            
            comp_depths = (disparity2depth(comp_cond) - masked_min) / denom
            comp_cond = depth2disparity(comp_depths)
        condition = torch.cat([global_cond, comp_cond], dim=1)
        
        if self.args.extra_condition == 'error':
            uctns = completed_maps['uncertainties'].unsqueeze(1)
            condition = torch.cat([uctns, condition], dim=1)
        elif self.args.extra_condition == 'spmask':
            condition = torch.cat([sparse_masks, condition], dim=1)
        else:
            raise NotImplementedError(
                "The extra condition can only be `error_map` or `sparse_map`.")
            
        # heit = sparse_depths.shape[-2] // 14 * 14
        heit = 518
        if hasattr(self, "timer"):
            torch.cuda.synchronize()
            t0 = time.time()
        metric_disparities = self.model(images, heit, condition=condition, device=self.device)
        if hasattr(self, "timer"):
            torch.cuda.synchronize()
            t1 = time.time()
            self.timer.append(t1 - t0)
            
        metric_depths = disparity2depth(metric_disparities)
        if self.args.normalize_depth:
            metric_depths = metric_depths * denom + masked_min
        return metric_depths
    
    def zero_one_normalize(self, depth_maps, valid_masks=None, affine_only=False):
        
        if valid_masks is not None:
            masked_min = depth_maps.masked_fill(~valid_masks, float('inf')).min(dim=-1).values.min(dim=-1).values  # (B, 1)
            masked_max = depth_maps.masked_fill(~valid_masks, float('-inf')).max(dim=-1).values.max(dim=-1).values  # (B, 1)
        else:
            masked_min = depth_maps.min(dim=-1).values.min(dim=-1).values  # (B, 1)
            masked_max = depth_maps.max(dim=-1).values.max(dim=-1).values  # (B, 1)
        
        denom = masked_max - masked_min
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        masked_min = masked_min.view(-1, 1, 1, 1)  # (B, 1, 1, 1)
        denom = denom.view(-1, 1, 1, 1)
        
        if not affine_only:
            normalized = (depth_maps - masked_min) / denom
            return normalized, (masked_min, denom)
        else:
            return masked_min, denom
