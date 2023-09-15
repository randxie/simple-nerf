import numpy as np
import torch

from model import position_encode
from utils_rays import get_rays


def test_get_rays_succeeded():
    focal = 1.0
    c2w = torch.eye(4).unsqueeze(0)
    rays_o, rays_d = get_rays(focal, c2w, height=2, width=2)

    assert rays_o.shape == (1, 2, 2, 3)
    assert rays_d.shape == (1, 2, 2, 3)


def test_pos_enc_succeeded():
    out = position_encode(torch.Tensor([[1, 2, 3]]), l_emb=1)

    assert np.allclose(out.cpu().numpy(), np.array([1, 2, 3, 0, 0, 0, 1, 1,
                                                    1]), 1e-12)
