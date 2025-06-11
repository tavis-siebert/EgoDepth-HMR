from typing import Optional
import torch
from torch.nn import functional as F
from typing import Tuple
import numpy as np

def naive_chamfer_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute a naive symmetric Chamfer Distance (mean of minimum distances).
    Args:
        x: (N, 3) torch.Tensor of predicted points
        y: (M, 3) torch.Tensor of target points
    Returns:
        torch.Tensor: scalar Chamfer Distance
    """
    # print("x shape: ", x.shape, "y shape: ", y.shape)
    x = x.unsqueeze(1)  # (N, 1, 3)
    y = y.unsqueeze(0)  # (1, M, 3)
    dist = torch.norm(x - y, dim=2)  # (N, M)
    cd_xy = dist.min(dim=1)[0].mean()
    cd_yx = dist.min(dim=0)[0].mean()
    return cd_xy + cd_yx


def chunked_chamfer_distance(x: torch.Tensor, y: torch.Tensor, chunk_size: int=100) -> torch.Tensor:
    """
    Compute Chamfer Distance using chunked processing for both dimensions.
    Memory usage is O(chunk_size^2) instead of O(N*M).
    
    Args:
        x: (N, 3) torch.Tensor of predicted points
        y: (M, 3) torch.Tensor of target points
        chunk_size: Size of chunks to process
    
    Returns:
        torch.Tensor: scalar Chamfer Distance
    """
    # print("x shape: ", x.shape, "y shape: ", y.shape)
    device = x.device

    # For each point in x, find the minimum distance to any point in y
    # min_dists_xy = torch.full((x.shape[0],), float('inf'), device=device)
    min_dists_xy = []
    # print("min_dists_xy shape: ", min_dists_xy.shape)
    for i in range(0, x.shape[0], chunk_size):
        x_chunk = x[i:i+chunk_size]  # (chunk_size, 3)
        x_exp = x_chunk.unsqueeze(1)  # (chunk_size, 1, 3)
        
        end_i = min(i + chunk_size, x.shape[0])
        # min_dist_i = torch.full((end_i - i,), float('inf'), device=device)  # (end_i - i,)
        min_dist_i = []
        for j in range(0, y.shape[0], chunk_size):
            # print("Processing chunk: i=", i, "j=", j)
            end_j = min(j + chunk_size, y.shape[0])
            y_chunk = y[j:end_j]  # (chunk_size, 3)
            y_exp = y_chunk.unsqueeze(0)  # (1, chunk_size, 3)
            # print("x_exp shape: ", x_exp.shape, "y_exp shape: ", y_exp.shape)
            
            dist = torch.norm(x_exp - y_exp, dim=2)  # (chunk_size, chunk_size)
            min_dist_in_chunk = dist.min(dim=0)[0]  # (chunk_size,)
            min_dist_i.append(min_dist_in_chunk)
            # print("min_dist_in_chunk shape: ", min_dist_in_chunk.shape)

            # current_mins = min_dists_xy[i:end_i] # (end_i - i,)
            # print("current_mins shape: ", current_mins.shape)
            # min_dists_xy[i:end_i] = torch.min(current_mins, min_dist_in_chunk[:end_i-i])
            # min_dist_i = torch.min(min_dist_i, min_dist_in_chunk[:end_i - i])
        min_dist_i = torch.cat(min_dist_i, dim=0)  # (end_i - i,)
        min_dists_xy.append(min_dist_i.min(dim=0)[0])  # (1,)
    min_dists_xy = torch.Tensor(min_dists_xy)  # (N,)
    
    cd_xy = min_dists_xy.mean()

    # For each point in y, find the minimum distance to any point in x
    # min_dists_yx = torch.full((y.shape[0],), float('inf'), device=device)
    min_dists_yx = []
    for j in range(0, y.shape[0], chunk_size):
        y_chunk = y[j:j+chunk_size]  # (chunk_size, 3)
        y_exp = y_chunk.unsqueeze(1)  # (chunk_size, 1, 3)
        
        end_j = min(j + chunk_size, y.shape[0])
        # min_dists_j = torch.full((end_j - j,), float('inf'), device=device)  # (end_j - j,)
        min_dists_j = []
        for i in range(0, x.shape[0], chunk_size):
            end_i = min(i + chunk_size, x.shape[0])
            x_chunk = x[i:end_i]  # (chunk_size, 3)
            x_exp = x_chunk.unsqueeze(0)  # (1, chunk_size, 3)
            
            dist = torch.norm(y_exp - x_exp, dim=2)  # (chunk_size, chunk_size)
            min_dist_in_chunk = dist.min(dim=0)[0]  # (chunk_size,)
            min_dists_j.append(min_dist_in_chunk)

            # current_mins = min_dists_yx[j:end_j]
            # min_dists_yx[j:end_j] = torch.min(current_mins, min_dist_in_chunk[:end_j-j])
            # min_dists_j = torch.min(min_dists_j, min_dist_in_chunk[:end_j - j])
        min_dists_j = torch.cat(min_dists_j, dim=0)
        min_dists_yx.append(min_dists_j.min(dim=0)[0])  # (1,)
    min_dists_yx = torch.Tensor(min_dists_yx)  # (M,)
    
    cd_yx = min_dists_yx.mean()
    
    return cd_xy + cd_yx
