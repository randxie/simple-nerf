import torch

from utils_rays import get_rays


def test_get_rays_succeeded():
    focal = 1.0
    c2w = torch.eye(4).unsqueeze(0)
    rays_o, rays_d = get_rays(focal, c2w, height=2, width=2)

    assert rays_o.shape == (1, 2, 2, 3)
    assert rays_d.shape == (1, 2, 2, 3)
