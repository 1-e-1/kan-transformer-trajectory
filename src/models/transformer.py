from __future__ import annotations

import torch
from torch import nn


class ControlTransformer(nn.Module):
    def __init__(self, d_model: int = 64, nhead: int = 4, num_layers: int = 3, dim_feedforward: int = 256, dropout: float = 0.0, fully_kan: bool = False, dense_kan_grid_size: int = 4, dense_kan_spline_order: int = 3):
        super().__init__()
        if fully_kan:
            from .full_kan import KANTransformerEncoder
            self.encoder=KANTransformerEncoder(d_model,nhead,num_layers,dim_feedforward,dropout,dense_kan_grid_size,dense_kan_spline_order)
            return
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))

    def forward(self, tokens: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        return hidden[:, 0]
