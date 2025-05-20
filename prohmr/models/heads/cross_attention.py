import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding2D(nn.Module):
    """
    Generates and adds 2D sinusoidal positional encodings.
    """
    def __init__(self, dim, height, width):
        super().__init__()
        if dim % 4 != 0:
            raise ValueError("Dimension must be divisible by 4 for 2D sincos encoding")
        self.height = height
        self.width = width

        pe = torch.zeros(dim, height, width)
        d_h = dim // 2
        d_w = dim - d_h
        div_h = torch.exp(torch.arange(0, d_h, 2) * -(torch.log(torch.tensor(10000.0)) / d_h))
        div_w = torch.exp(torch.arange(0, d_w, 2) * -(torch.log(torch.tensor(10000.0)) / d_w))
        pos_h = torch.arange(height).unsqueeze(1)
        pos_w = torch.arange(width).unsqueeze(1)
        pe[0:d_h:2, :, :] = torch.sin(pos_h * div_h).transpose(0,1).unsqueeze(2)
        pe[1:d_h:2, :, :] = torch.cos(pos_h * div_h).transpose(0,1).unsqueeze(2)
        pe[d_h::2, :, :] = torch.sin(pos_w * div_w).transpose(0,1).unsqueeze(1)
        pe[d_h+1::2, :, :] = torch.cos(pos_w * div_w).transpose(0,1).unsqueeze(1)

        self.register_buffer('pe', pe.unsqueeze(0))  # (1, C, H, W)

    def forward(self, x):
        return x + self.pe

class CrossAttentionImages(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=8, height=7, width=7, dropout=0.0):
        """
        dim: number of channels for both query and key/value
        height, width: spatial dims of the feature maps
        """
        super().__init__()
        self.pos_enc = PositionalEncoding2D(in_dim, height, width)
        self.q_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.k_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.v_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.attn = nn.MultiheadAttention(embed_dim=in_dim, num_heads=num_heads, dropout=dropout)
        self.out_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, feat_q, feat_kv):
        """
        feat_q, feat_kv: (B, C, H, W)
        returns: (B, C) vector, same as ResNet’s final output
        """
        B, C, H, W = feat_q.shape
        # Add positional encoding
        feat_q = self.pos_enc(feat_q)
        feat_kv = self.pos_enc(feat_kv)
        # Project to Q, K, V and reshape to (H * W, B, C)
        q = self.q_proj(feat_q).view(B, C, -1).permute(2, 0, 1)
        k = self.k_proj(feat_kv).view(B, C, -1).permute(2, 0, 1)
        v = self.v_proj(feat_kv).view(B, C, -1).permute(2, 0, 1)
        # Cross-attention
        attn_out, _ = self.attn(query=q, key=k, value=v)
        # Reshape back
        attn_map = attn_out.permute(1, 2, 0).view(B, C, H, W)
        fused = self.out_proj(attn_map) + feat_q
        # Global avg pool to (B, C)
        feat = self.pool(fused).view(B, -1)
        return feat
