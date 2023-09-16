# Reference: Matt Williams's notebook https://github.com/bmild/nerf/blob/master/tiny_nerf.ipynb
import torch
from torch import nn
from typing import Callable, Tuple
from jaxtyping import Float
import matplotlib.pyplot as plt


def position_encode(x: Float[torch.Tensor, "N 3"],
                    l_emb: int = 6) -> Float[torch.Tensor, "N d"]:
    """
    Map each point in x from R to R^{2L}

    :param x: A matrix
    :param l_emb: The number of pos enc function
    :return: Encoded tensor
    """

    # l_emb
    pos_enc = torch.pow(torch.arange(l_emb, dtype=torch.float32),
                        2.0).to(x.device)
    x_sin = torch.sin(x.unsqueeze(-1) * pos_enc.unsqueeze(0)).view(
        x.shape[0], -1)
    x_cos = torch.cos(x.unsqueeze(-1) * pos_enc.unsqueeze(0)).view(
        x.shape[0], -1)

    return torch.cat([x, x_sin, x_cos], dim=-1)


# The x,y,z,theta,phi are in the voxel grid coordinate system
class MLP(nn.Module):
    """The MLP layer in NERF that predicts rgb and sigma"""

    def __init__(
        self,
        l_emb: int = 6,
        dim_hidden: int = 256,
        num_hidden: int = 8,
        dim_out: int = 4,
    ):
        super(MLP, self).__init__()

        # convert input to hidden size
        num_channels = 3
        dim_in = num_channels + 2 * l_emb * num_channels
        self.w_in = nn.Linear(dim_in, dim_hidden)

        self.w_hidden_1 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
            ) for _ in range(num_hidden // 2)
        ])
        self.w_hidden_cast = nn.Linear(dim_in + dim_hidden, dim_hidden)
        self.w_hidden_2 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
            ) for _ in range(num_hidden // 2)
        ])

        self.w_out = nn.Linear(dim_hidden, dim_out)

    def forward(
        self,
        x_in: Float[torch.Tensor, "b i"],
    ) -> Float[torch.Tensor, "b i"]:
        x = self.w_in(x_in)

        for layer in self.w_hidden_1:
            x = layer(x)
        x = self.w_hidden_cast(torch.cat([x_in, x], dim=-1))

        for layer in self.w_hidden_2:
            x = layer(x)
        x = self.w_out(x)

        return x


class Batcher(object):

    def __init__(self, batch_size: int, out_dim: int = 4):
        super(Batcher, self).__init__()
        self.batch_size = batch_size
        self.out_dim = out_dim

    def apply(
        self,
        fn: Callable[[Float[torch.Tensor, "N 3"]], torch.Tensor],
        x: Float[torch.Tensor, "N 4"],
    ):
        output = torch.zeros(x.shape[0],
                             self.out_dim,
                             device=x.device,
                             dtype=x.dtype)
        for i in range(0, x.shape[0], self.batch_size):
            output[i:i + self.batch_size] = fn(x[i:i + self.batch_size])

        return output


class NaiveNERF(nn.Module):

    def __init__(self):
        super(NaiveNERF, self).__init__()
        self.mlp = MLP()

    def forward(
        self,
        rays_o: Float[torch.Tensor, "b i j 3"],
        rays_d: Float[torch.Tensor, "b i j 3"],
        near: float,
        far: float,
        n_samples: int,
    ):
        B, W, H, _ = rays_o.shape
        step_size = (far - near) / n_samples
        depth_vals = torch.linspace(near, far, n_samples, dtype=rays_d.dtype)

        # adding random noise will improve the learning because you have varying z-loc
        depth_vals += torch.rand(n_samples, dtype=depth_vals.dtype) * step_size
        depth_vals = depth_vals.to(rays_d.device)

        # [B, H, W, N_Samples, 3]
        pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * depth_vals.unsqueeze(-1)

        # [H*W, 3]
        pts_flat = pts.view(B * H * W * n_samples, 3)

        # add position encode
        pts_flat = position_encode(pts_flat)

        # batching the points
        pts_pred = Batcher(1024 * 32).apply(self.mlp, pts_flat)

        # [B, H, W, N_Samples, 4]
        pts_pred = pts_pred.view(B, H, W, n_samples, 4)

        # do volume rendering
        # [B, H, W, N, 3]
        rgb = torch.sigmoid(pts_pred[..., :3])

        # [B, H, W, N]
        sigma = torch.relu(pts_pred[..., 3])

        # [N]
        steps = torch.cat(
            [
                depth_vals[1:] - depth_vals[:-1], 1e10 *
                torch.ones(1, device=depth_vals.device, dtype=rays_d.dtype)
            ],
            dim=0,
        )
        # [B, H, W, N]
        alpha = 1 - torch.exp(-sigma * steps)
        # [B, H, W, N]
        Ti = torch.cumprod(1 - alpha + 1e-10, dim=-1)
        # [B, H, W, N]
        weights = alpha * Ti

        # [B, H, W, 3]
        rgb_map = torch.sum(rgb * weights.unsqueeze(-1), dim=-2)

        # [B, H, W]
        depth_map = torch.sum(weights * depth_vals, dim=-1)

        # [B, H, W]
        acc_map = torch.sum(weights, dim=-1)

        return rgb_map, depth_map, acc_map
