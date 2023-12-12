
import numpy as np
import torch

from torch.nn import functional as F

def rampfilter(size):
    n = np.concatenate((np.arange(1, size / 2 + 1, 2, dtype=int),
                        np.arange(size / 2 - 1, 0, -2, dtype=int)))
    f = np.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2
    return torch.tensor(2 * np.real(np.fft.fft(f)) )

class AngledFBP(torch.nn.Module):
    ''' 
    Filtered Backprojection

    Args:
    '''
    def __init__(self, sinogram, circle=True, filtered=True):
        super().__init__()

        self.circle = circle
        self.N, _, self.n_angles, self.M = sinogram.shape

        grid_z, grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, self.N),
                                                torch.linspace(-1, 1, self.M),
                                                torch.linspace(-1, 1, self.M))
        self.register_buffer('grid_z', torch.Tensor(grid_z).double())
        self.register_buffer('grid_y', torch.Tensor(grid_y).double())
        self.register_buffer('grid_x', torch.Tensor(grid_x).double())
        self.register_buffer('sinogram', torch.Tensor(sinogram).double())

        if filtered: self.apply_filter()

    @staticmethod
    def rampfilter(size):
        n = np.concatenate((np.arange(1, size / 2 + 1, 2, dtype=int),
                            np.arange(size / 2 - 1, 0, -2, dtype=int)))
        f = np.zeros(size)
        f[0] = 0.25
        f[1::2] = -1 / (np.pi * n) ** 2
        return torch.tensor(2 * np.real(np.fft.fft(f)))

    def apply_filter(self):
        # padding values
        final_width = max(64, int(2 ** (2 * torch.tensor(self.M)).float().log2().ceil()))
        pad_width = (final_width - self.M)

        filter = self.rampfilter(final_width)

        # pad sinogram
        self.sinogram = F.pad(self.sinogram, [0, pad_width], mode='constant', value=0) # (N, 1, n_angles, M + pad_width)

        # apply filter
        self.sinogram = torch.fft.fft(self.sinogram, dim=-1) * filter.double()
        self.sinogram = torch.real(torch.fft.ifft(self.sinogram, dim=-1))[..., :self.M] # (N, 1, n_angles, M)

    def forward(self, tilt_theta, theta, center=None):
        theta = torch.Tensor(theta).to(self.sinogram.device)
        tilt_theta = torch.tensor(tilt_theta).to(self.sinogram.device)

        if center is not None:
            # if center is iterable
            if hasattr(center, '__iter__'):
                center = torch.Tensor(center).to(self.sinogram.device)
            else:
                center = torch.Tensor([center] * self.N).to(self.sinogram.device)

        recon = torch.zeros(self.N, self.M, self.M, device=self.sinogram.device)
        for i, theta_i in enumerate(theta):
            recon_grid_x = self.grid_x * torch.cos(theta_i) - self.grid_y * torch.sin(theta_i) # (N, M, M)
            recon_grid_x = recon_grid_x.unsqueeze(-1) # (N, M, M, 1)

            if center is not None:
                recon_grid_x = recon_grid_x + (center.view(self.N, 1, 1, 1) - self.M / 2) / self.M * 2

            z_delta = self.grid_x * torch.sin(theta_i) + self.grid_y * torch.cos(theta_i) # (N, M, M)
            z_delta = z_delta * torch.sin(tilt_theta) * self.M / self.N

            recon_grid_z = self.grid_z + z_delta # (N, M, M)
            recon_grid_z = recon_grid_z.unsqueeze(-1) # (N, M, M, 1)

            recon_grid = torch.cat((recon_grid_x, recon_grid_z), dim=-1) # (N, M, M, 2)
            recon_grid = recon_grid.view(1, self.N * self.M, self.M, 2)

            recon_i = F.grid_sample(self.sinogram[:, :, i].view(1, 1, self.N, self.M), 
                                    recon_grid, 
                                    mode='bilinear', 
                                    padding_mode='zeros', 
                                    align_corners=True) # (1, 1, N * M, M)
            recon_i = recon_i.view(self.N, self.M, self.M) # (N, M, M)
            recon += recon_i

        recon[(self.grid_x ** 2 + self.grid_y ** 2) > 0.95] = 0
        return recon * np.pi / (2 * self.n_angles)