import torch
from typing import Tuple
from jaxtyping import Float


def get_rays(
    focal: float,
    c2w: Float[torch.Tensor, "b 4 4"],
    height: int = 100,
    width: int = 100,
) -> Tuple[Float[torch.Tensor, "b w h 3"], Float[torch.Tensor, "b w h 3"]]:
    """Get rays origin and direction in the world coordinate system.

    Parameters
    ----------
    focal : [float] 
        Focal length
    c2w : [torch.Tensor]
        Camera to word matrix
    """
    i, j = torch.meshgrid(
        torch.arange(width, dtype=torch.float32),
        torch.arange(height, dtype=torch.float32),
        indexing='xy',
    )

    w_h = (width - 1) * 0.5
    h_h = (height - 1) * 0.5

    x = ((i - w_h) / focal).unsqueeze(-1)
    y = ((h_h - j) / focal).unsqueeze(-1)
    z = -torch.ones_like(x)

    # (B, W, H, 3)
    dirs = torch.cat(
        [x, y, z],
        dim=-1,
    ).unsqueeze(0).repeat(c2w.shape[0], 1, 1, 1).to(c2w.device)
    rays_d = torch.einsum(
        'bxyz,bkz->bxyk',
        dirs,
        c2w[:, :3, :3],
    )
    rays_o = c2w[:, None, None, :3, -1].expand(rays_d.shape)

    return rays_o, rays_d
