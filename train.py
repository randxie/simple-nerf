import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from model import NaiveNERF
from utils_rays import get_rays


class TinyNERFData(Dataset):

    def __init__(self):
        self.data = np.load('data/tiny_nerf_data.npz')
        self.height, self.width = self.data['images'].shape[1:3]
        self.focal = float(self.data['focal'])

    def __len__(self):
        return self.data['images'].shape[0]

    def __getitem__(self, idx):
        img = self.data['images'][idx]
        pose = self.data['poses'][idx]
        return img, pose


if __name__ == "__main__":
    data = TinyNERFData()
    width, height = data.width, data.height
    focal = data.focal

    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size

    # fix seed for reproducibility
    generator = torch.Generator()
    generator.manual_seed(42)
    dtrain, dval = random_split(data, [train_size, test_size],
                                generator=generator)

    # create the model and optimizer
    nerf_mdl = NaiveNERF()
    optimizer = torch.optim.Adam(nerf_mdl.parameters(), lr=5e-4)

    # number of samples along each ray's direction
    n_samples = 64
    n_iters = 1000
    for it in range(n_iters):
        nerf_mdl.train()
        for img_b, pose_b in DataLoader(dtrain, batch_size=1, shuffle=True):
            # pose should be a (B, 4, 4) matrix
            rays_o, rays_d = get_rays(focal, pose_b)

            # Run the model and predict the novel view
            # (B, H, W, 3)
            img_pred = nerf_mdl(rays_o, rays_d, n_samples)
            loss = torch.mean(torch.square(img_b - img_pred))
            loss.backward()
            optimizer.zero_grad()

        # run eval
        nerf_mdl.eval()
        if it % 10 == 0:
            for img_b, pose_b in DataLoader(dval, batch_size=1, shuffle=False):
                pass
