import math
from typing import Optional

import tripy as tp


class PositionEmbeddingSine(tp.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention Is All You Need paper, generalized to work on images.
    """

    def __init__(
        self,
        num_pos_feats,
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None,
    ):
        super().__init__()
        assert num_pos_feats % 2 == 0, "Expecting even model width"
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def __call__(self, x: tp.Tensor):
        # x: [B, C, H, W]
        B, _, H, W = x.shape
        y_embed = tp.arange(1, H + 1, dtype=tp.float32)
        y_embed = tp.reshape(y_embed, (1, -1, 1))
        y_embed = tp.repeat(y_embed, B, 0)
        y_embed = tp.repeat(y_embed, W, 2)  # [B, H, W]

        x_embed = tp.arange(1, W + 1, dtype=tp.float32)
        x_embed = tp.reshape(x_embed, (1, 1, -1))
        x_embed = tp.repeat(x_embed, B, 0)
        x_embed = tp.repeat(x_embed, H, 1)  # [B, H, W]

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = tp.arange(self.num_pos_feats, dtype=tp.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = tp.unsqueeze(x_embed, -1) / dim_t
        pos_y = tp.unsqueeze(y_embed, -1) / dim_t
        pos_x = tp.stack((tp.sin(pos_x[:, :, :, 0::2]), tp.cos(pos_x[:, :, :, 1::2])), dim=4)
        pos_y = tp.stack((tp.sin(pos_y[:, :, :, 0::2]), tp.cos(pos_y[:, :, :, 1::2])), dim=4)
        pos_x = tp.flatten(pos_x, 3)
        pos_y = tp.flatten(pos_y, 3)
        pos = tp.permute(pos, (0, 3, 1, 2))
        return pos

    def generate_static_embedding(self, inp_shape):
        import torch

        B, _, H, W = inp_shape
        y_embed = torch.arange(1, H + 1, dtype=torch.float32).view(1, -1, 1).repeat(B, 1, W)
        x_embed = torch.arange(1, W + 1, dtype=torch.float32).view(1, 1, -1).repeat(B, H, 1)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        print(f"Position embedding for {inp_shape}: {pos.shape}")
        return tp.Tensor(pos)
