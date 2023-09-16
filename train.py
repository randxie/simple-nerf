import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from model import NaiveNERF
from utils_rays import get_rays
import matplotlib.pyplot as plt
from tqdm import tqdm


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

    train_size = int(0.95 * len(data))
    test_size = len(data) - train_size

    # fix seed for reproducibility
    generator = torch.Generator()
    generator.manual_seed(42)
    dtrain, dval = random_split(data, [train_size, test_size],
                                generator=generator)

    # create the model and optimizer
    device = "cuda:0"
    nerf_mdl = NaiveNERF()
    nerf_mdl.to(device)
    optimizer = torch.optim.Adam(nerf_mdl.parameters(), lr=1e-4)

    # number of samples along each ray's direction
    n_samples = 64
    n_iters = 2000
    iter_nums = []
    psnrs = []
    for it in range(n_iters):
        for (img_b,
             pose_b) in tqdm(DataLoader(dtrain, batch_size=2, shuffle=True)):
            # pose should be a (B, 4, 4) matrix
            rays_o, rays_d = get_rays(
                focal,
                pose_b,
                height=height,
                width=width,
            )

            # load to GPU
            img_b = img_b.to(device)
            pose_b = pose_b.to(device)
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)

            # Run the model and predict the novel view
            # (B, H, W, 3)
            img_pred, depth_pred, acc_pred = nerf_mdl(
                rays_o,
                rays_d,
                near=2.0,
                far=6.0,
                n_samples=64,
            )
            loss = torch.mean(torch.square(img_b - img_pred))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if it % 10 == 0:
                plt.figure(figsize=(10, 4))
                plt.subplot(121)
                plt.imshow(img_pred[0].detach().cpu().numpy())
                plt.title(f'Pred Iteration: {it}')
                plt.subplot(122)
                plt.imshow(img_b[0].detach().cpu().numpy())
                plt.title(f'GT Iteration: {it}')
                plt.savefig(f'./results/train_{it}.png')
                plt.close()

        # run eval
        with torch.no_grad():
            for img_b, pose_b in DataLoader(dval, batch_size=1, shuffle=True):
                rays_o, rays_d = get_rays(focal, pose_b)
                # load to GPU
                img_b = img_b.to(device)
                pose_b = pose_b.to(device)
                rays_o = rays_o.to(device)
                rays_d = rays_d.to(device)

                img_pred, depth_pred, acc_pred = nerf_mdl(
                    rays_o,
                    rays_d,
                    near=2.0,
                    far=6.0,
                    n_samples=64,
                )
                loss = torch.mean(torch.square(img_b - img_pred))

                psnr = -10. * torch.log(loss) / np.log(10.0)
                iter_nums.append(it)
                psnrs.append(float(psnr.detach().cpu().numpy()))

                plt.figure(figsize=(10, 4))
                plt.subplot(131)
                plt.imshow(img_pred[0].detach().cpu().numpy())
                plt.title(f'Pred Iteration: {it}')
                plt.subplot(132)
                plt.imshow(img_b[0].detach().cpu().numpy())
                plt.title(f'GT Iteration: {it}')
                plt.subplot(133)
                plt.plot(iter_nums, psnrs)
                plt.title('PSNR')
                plt.savefig(f'./results/val_{it}.png')
                plt.close()
                break
