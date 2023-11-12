import torch
import tomopy
import numpy as np

from tqdm import tqdm
from scipy.interpolate import interp1d, interp2d

import cupy as cp
from cupyx.scipy.signal import convolve2d

from bayes_opt import BayesianOptimization

from torch import nn
from torch.nn import functional as F
from xitorch.interpolate import Interp1D

class EarlyStopping():
    def __init__(self, model=None, patient=5, epsilon=0):
        self.model = model
        self.patient = patient
        self.min_loss = np.inf
        self.counter = 0
        self.early_stop = False
        self.best_model = None
        self.epsilon = epsilon

    def __call__(self, loss):
        if loss > self.min_loss - self.epsilon:
            self.counter += 1
            if self.counter > self.patient:
                self.early_stop = True
                return True
        else:
            self.counter = 0
            self.min_loss = loss
            if self.model is not None:
                self.best_model = self.model.state_dict()
            return False
        
class SampledFBP(nn.Module):
    ''' 
    Sampled Filtered Backprojection

    Args:
        sinogram: (N, 1, n_angles, M) sinogram
        n_patch: number of patches
        patch_size: size of the patch
        circle: whether to use a circle mask
        filtered: whether to apply ramp filter
    '''
    def __init__(self, sinogram, n_patches, patch_size, circle=True, filtered=True):
        super().__init__()
        
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.circle = circle

        self.N, _, self.n_angles, self.M = sinogram.shape

        # generate a patch grid
        X = torch.arange(self.patch_size) - self.patch_size / 2
        patch_basegrid_X, patch_basegrid_y = torch.meshgrid(X, X)

        self.register_buffer('sinogram', torch.Tensor(sinogram).double())
        self.register_buffer('patch_basegrid_X', patch_basegrid_X)
        self.register_buffer('patch_basegrid_y', patch_basegrid_y)

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

    def sample_inside_circle(self, n):
        x, y = list(), list()

        while len(x) < n:
            x_ = torch.rand(n - len(x), device=self.patch_basegrid_X.device) * self.M
            y_ = torch.rand(n - len(x), device=self.patch_basegrid_X.device) * self.M

            mask = (x_ - self.M / 2) ** 2 + (y_ - self.M / 2) ** 2 <= (self.M / 2) ** 2

            if mask.sum() == 0: continue
            x.append(x_[mask])
            y.append(y_[mask])

        return torch.cat(x)[:n], torch.cat(y)[:n]

    def generate_patches(self):
        # duplicate patch grid for each sample
        patch_grid_X = self.patch_basegrid_X[None].repeat(self.n_patches, 1, 1)
        patch_grid_y = self.patch_basegrid_y[None].repeat(self.n_patches, 1, 1)

        # sample center points inside central circle
        X, y = self.sample_inside_circle(self.n_patches)

        # shift patch grid to the center points
        patch_grid_X = patch_grid_X + X.view(-1, 1, 1)
        patch_grid_y = patch_grid_y + y.view(-1, 1, 1)

        # convert the value range to (-1, 1)
        patch_grid_X = patch_grid_X / ((self.M-1) / 2) - 1
        patch_grid_y = patch_grid_y / ((self.M-1) / 2) - 1

        return patch_grid_X, patch_grid_y
    
    def generate_circle_mask(self, patch_grid_X, patch_grid_y):
        return (patch_grid_X ** 2 + patch_grid_y ** 2) <= 1

    def forward(self, theta):
        theta = torch.Tensor(theta).double().to(self.sinogram.device)

        patch_grid_X, patch_grid_y = self.generate_patches() # (n_patch, patch_size, patch_size)

        theta = theta[:, None, None]
        recon_grid_X = patch_grid_X.unsqueeze(1) * theta.cos() - patch_grid_y.unsqueeze(1) * theta.sin() 
        recon_grid_X = recon_grid_X.unsqueeze(-1) # (n_patch, n_angles, patch_size, patch_size, 1)
        recon_grid_y = torch.ones_like(recon_grid_X) * torch.linspace(-1, 1, self.n_angles, device=theta.device)[:, None, None, None]

        recon_grid = torch.cat((recon_grid_X, recon_grid_y), dim=-1)
        recon_grid = recon_grid.view(1, self.n_patches * self.n_angles * self.patch_size, self.patch_size, 2)
        recon_grid = recon_grid.repeat(self.N, 1, 1, 1)

        recon = F.grid_sample(self.sinogram, recon_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        recon = recon.view(self.N, self.n_patches, self.n_angles, self.patch_size, self.patch_size).sum(2)

        if self.circle:
            recon = recon * self.generate_circle_mask(patch_grid_X, patch_grid_y)
        return recon * np.pi / (2 * self.n_angles) # (N, n_patch, patch_size, patch_size)

class ThetaEstimator(nn.Module):
    def __init__(self, sinogram, assumed_angles, n_keypoints, n_patches=2, patch_size=128):
        '''
        Args:
            sinogram: (N, 1, n_angles, M) sinogram
            assumed_angles: (n_angles, ) array of assumed theta in radians
            n_keypoints: number of keypoints
            n_patches: number of patches
            patch_size: size of the patch, if smaller than 1 the patch_size will be M * patch_size
        '''

        super().__init__()

        self.n_sinograms, _, self.n_angles, self.M = sinogram.shape

        self.n_patches = n_patches
        self.patch_size = patch_size if patch_size > 1 else int(self.M * patch_size)

        self.n_keypoints = n_keypoints
        self.shift = nn.Parameter(torch.zeros(n_keypoints - 1, ))

        self.interp_f = Interp1D(torch.linspace(0, self.n_angles, self.n_keypoints).cuda(), assume_sorted=True)

        loss_kernel = torch.tensor([[1, 0, -1],
                                    [2, 0, -2],
                                    [1, 0, -1],]) # (3, 3)
        loss_kernel = loss_kernel.view(1, 1, 3, 3).double()
        loss_kernel = loss_kernel.repeat(self.n_patches, 1, 1, 1)

        self.fbp = SampledFBP(sinogram, n_patches, patch_size, True, True)

        self.register_buffer('interp_range', torch.arange(self.n_angles))
        self.register_buffer('loss_kernel', loss_kernel)
        self.register_buffer('assumed_angles', torch.Tensor(assumed_angles).double())
        
    def cuda(self):
        super().cuda()
        self.fbp = self.fbp.cuda()
        return self

    def interp_shift(self):
        '''
        Interpolate keypoints to get angle shift.

        return:
            shift: (n_angles, ) array of shift in radians
        '''
        return self.interp_f(self.interp_range, F.pad(self.shift, (1, 0), 'constant', 0))

    def forward(self):
        '''
        return:
            estimated_theta: (n_angles, ) array of theta in radians
        '''
        return self.assumed_angles + self.interp_shift()

    def gradient_magnitude(self, recon):
        v_gradient = F.conv2d(recon, self.loss_kernel, padding='same', groups=self.n_patches) ** 2
        h_gradient = F.conv2d(recon, self.loss_kernel.transpose(2, 3), padding='same', groups=self.n_patches) ** 2
        gm = ((v_gradient + h_gradient + 1e-9) ** 0.5)

        return gm

    def loss(self):
        return self.gradient_magnitude(self.fbp(self.forward())).mean() # smaller is better

def acutance(recon):
    # sobal kernel
    kernel = cp.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ], dtype=cp.float32)

    recon = cp.array(recon, dtype=cp.float32)
    gradient_magnitude = 0
    for r in recon:
        gradient_magnitude += ((convolve2d(r, kernel, mode='valid') ** 2 + convolve2d(r, kernel.T, mode='valid') ** 2) ** 0.5).mean().get()

    # mean of the gradient magnitude
    return gradient_magnitude / len(recon)

def padded_recon(sinogram, theta, center, pad_width):
    if center is not None: center = center + pad_width
    recon = tomopy.recon(np.pad(sinogram, ((0, 0), (0, 0), (pad_width, pad_width)), 'edge'),
                         theta,
                         center=center,
                         algorithm=tomopy.astra,
                         sinogram_order=True,
                         options={'proj_type': 'cuda', 'method': 'FBP_CUDA', 'extra_options': {'FilterType': 'hamming'}},
                         ncore=1)[:, pad_width:-pad_width, pad_width:-pad_width]
    recon = tomopy.circ_mask(recon, axis=0, ratio=1.0, val=0) # (N, M, M)
    return recon

def center_correction(sinogram, z_indices, total_z, theta, center_range, init_points=50, n_iter=300):
    '''
    Correct center shift in sinogram

    args:
        sinogram: (N, n_angles, M), z indices should be in ascending order
        z_indices: (N, ) z indices of the sinogram starting from 0
        total_z: total number of z slices
        theta: (n_angles,) array of the original theta in radians
        center_range: range of center shift
        init_points: number of initial points in Bayesian Optimization
        n_iter: number of iterations in Bayesian Optimization

    return:
        corrected_center: (total_z, ) array of corrected center
    '''
    def loss(start_center, end_center):
        X = [z_indices[0], z_indices[-1]]
        y = [start_center, end_center]
        center = interp1d(X, y, kind='linear')(z_indices) + width / 2
        return -acutance(padded_recon(sinogram, theta, center, pad_width))
        
    # make sure sinogram is 3D
    if len(sinogram.shape) == 2:
        sinogram = np.array(sinogram)[np.newaxis, ...]

    assumed_theta, n_angles, width = theta, len(theta), sinogram.shape[2]
    pad_width = width // 4 + 1
    
    # Bounded region of parameter space
    pbounds = {
        'start_center': center_range,
        'end_center': center_range,
    }
    optimizer = BayesianOptimization(f=loss, pbounds=pbounds, verbose=1, allow_duplicate_points=True)

    # probe the assumed center and theta
    optimizer.probe({'start_center': 0, 'end_center': 0})

    optimizer.maximize(
        init_points = init_points,
        n_iter = n_iter,
    )

    # interpolate the center shift for each z slice
    start_center, end_center = optimizer.max['params']['start_center'], optimizer.max['params']['end_center']
    center_shifts = interp1d([z_indices[0], z_indices[-1]], [start_center, end_center], kind='linear', fill_value='extrapolate')(np.arange(total_z))
    return center_shifts + width / 2

def theta_correction(sinogram, theta, center, n_keypoints, shift_range, init_points=50, n_iter=300):
    '''
    Correct theta shift in sinogram.

    args:
        sinogram: (N, n_angles, M)
        theta: (n_angles,) array of the original theta in radians
        center: (N, ) center of rotation
        n_keypoints: number of keypoints
        shift_range: range of shift in radians
        init_points: number of initial points in Bayesian Optimization
        n_iter: number of iterations in Bayesian Optimization

    return:
        corrected_theta: (n_angles,) array of corrected theta
    '''
    def interp_theta(keypoints):
        '''
        Interpolate keypoints to get theta.
        '''
        f = interp1d(np.linspace(0, n_angles, n_keypoints), keypoints, kind='cubic')
        return f(np.arange(n_angles)) + assumed_theta

    def loss(**args):
        theta = interp_theta(np.array(list(args.values())))
        return -acutance(padded_recon(sinogram, theta, center, pad_width))
    
    # make sure sinogram is 3D
    if len(sinogram.shape) == 2:
        sinogram = np.array(sinogram)[np.newaxis, ...]

    assumed_theta, n_angles, width = theta, len(theta), sinogram.shape[2]
    pad_width = width // 4 + 1
    
    # Bounded region of parameter space
    pbounds = {'input' + str(i): shift_range for i in range(n_keypoints)}
    optimizer = BayesianOptimization(f=loss, pbounds=pbounds, verbose=1, allow_duplicate_points=True)

    # probe the assumed theta
    optimizer.probe(params={'input' + str(i): 0 for i in range(n_keypoints)})

    optimizer.maximize(
        init_points = init_points,
        n_iter = n_iter,
    )

    return interp_theta(np.array(list(optimizer.max['params'].values())))

def theta_correction_gd(sinogram, theta, center=None, n_keypoints=10, steps=10, patient=2):
    '''
    Correct theta shift in sinogram using gradient descent.

    args:
        sinogram: (N, n_angles, M)
        theta: (n_angles,) array of the original theta in radians
        center: (N, ) center of rotation (default: M / 2)
        n_keypoints: number of keypoints (default: n_angles // 50)
        patient: number of epochs to wait before early stopping

    return:
        corrected_theta: (n_angles,) array of corrected theta
    '''

    @torch.compile
    def train_step(estimator, optimizer):
        optimizer.zero_grad()
        estimator.loss().backward()
        optimizer.step()

    # make sure sinogram is 3D
    if len(sinogram.shape) == 2:
        sinogram = np.array(sinogram)[np.newaxis, ...]

    N, n_angles, M = sinogram.shape
    pad_width = M // 4 + 1

    if center is not None:
        if np.isscalar(center):
            center = np.array([center] * N)

        # shift sinogram to the center using interp2d
        X, y = np.arange(M), np.arange(n_angles)
        sinogram = np.array([interp2d(X, y, sino, kind='linear', fill_value=0)(X + c - M / 2, y) for sino, c in zip(sinogram, center)])
    
    sinogram = sinogram[:, np.newaxis, ...] # (N, 1, n_angles, M)
    estimator = ThetaEstimator(sinogram, theta, n_keypoints).cuda()
    
    optimizer = torch.optim.Adam(estimator.parameters(), lr=1e-4)
    early_stopping = EarlyStopping(estimator, patient=patient)

    # register the initial loss to the early stopping
    initial_loss = acutance(padded_recon(sinogram[:,0], theta, None, pad_width))
    early_stopping(initial_loss)
    print('Initial loss: {:.8f}'.format(initial_loss))

    pbar = tqdm()
    while not early_stopping.early_stop:
        # estimator.update_sample_map()
        epoch_loss = 0
        for _ in range(steps):
            train_step(estimator, optimizer)

        epoch_loss = acutance(padded_recon(sinogram[:,0], estimator.forward().cpu().detach().numpy(), None, pad_width))
        early_stopping(epoch_loss)

        # update progress bar
        pbar.set_description('loss: {:.8f}'.format(epoch_loss))
        pbar.update(1)
    pbar.close()

    print('Best loss: {:.8f}'.format(early_stopping.min_loss))

    # restore the best model
    estimator.load_state_dict(early_stopping.best_model)
    return estimator.forward().cpu().detach().numpy()