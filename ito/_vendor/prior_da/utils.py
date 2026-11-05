import torch
import numpy as np
from dataclasses import dataclass, field

@dataclass
class Arguments:
    K: int = field(
        default=5, 
        metadata={"help": "K value of KNN"}
    )
    conditioned_model_size: str = field(
        default="vitb", 
        metadata={"help": "Size of conditioned model."}
    )
    frozen_model_size: str = field(
        default="vitb", 
        metadata={"help": "Size of frozen model."}
    )
    normalize_depth: bool = field(
        default=True, 
        metadata={"help": "Whether to normalize depth."}
    )
    normalize_confidence: bool = field(
        default=True, 
        metadata={"help": "Whether to normalize confidence."}
    )
    double_global: bool = field(
        default=False, 
        metadata={"help": "Whether to use double globally-aligned conditions."}
    )

    repo_name: str = field(
        default='Rain729/Prior-Depth-Anything', metadata={"help": "Name of hf-repo."})
    log_dir: str = field(
        default='output', metadata={"help": "The root path to save visualization results."})
    # down_fill_mode: str = field(
    #     default='linear', 
    #     metadata={
    #         "help": (
    #             "The mode to fill in the vacancy in the prior. Only works for `pattern='^downscale_\d*$'`. "
    #             "Choices=('knn', 'global', 'linear')"
    #         )
    #     }
    # )
    
    
# ******************** disparity space ********************
# Adapted from Marigold, available at https://github.com/prs-eth/Marigold
def depth2disparity(depth, return_mask=False):
    if isinstance(depth, torch.Tensor):
        disparity = torch.zeros_like(depth)
    elif isinstance(depth, np.ndarray):
        disparity = np.zeros_like(depth)
    non_negtive_mask = depth > 0
    disparity[non_negtive_mask] = 1.0 / depth[non_negtive_mask]
    if return_mask:
        return disparity, non_negtive_mask
    else:
        return disparity

def disparity2depth(disparity, **kwargs):
    return depth2disparity(disparity, **kwargs)
# ************************* end ****************************
