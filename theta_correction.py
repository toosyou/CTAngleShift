import tomopy
import numpy as np

from scipy.interpolate import interp1d

import cupy as cp
from cupyx.scipy.signal import convolve2d

from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction

from tqdm import tqdm

from cuda_recon import FBP

SOBAL_KERNEL = cp.array([
                    [1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1],
                ], dtype=cp.float32)

@cp.fuse()
def l2_sum(x, y):
    return cp.sum((x ** 2 + y ** 2) ** 0.5)

class CircularAcutance:
    def __init__(self, size, n_sectors=6, auxiliary_ratio=0.1):
        self.size = size
        self.n_sectors = n_sectors

        gridx, gridy = cp.meshgrid(cp.arange(size), cp.arange(size))
        theta = cp.arctan2(gridy - (size-1) / 2, gridx - (size-1) / 2)
        
        sectors = cp.floor((theta + cp.pi) / (2 * cp.pi / n_sectors))
        sector_masks = [sectors == i for i in range(n_sectors)]
        self.weights = [mask * (1 - auxiliary_ratio) + auxiliary_ratio for mask in sector_masks]

    @cp.fuse()
    @staticmethod
    def l2(x, y):
        return (x ** 2 + y ** 2) ** 0.5

    def __call__(self, images):
        N, H, W = images.shape

        images = cp.array(images, dtype=cp.float32, copy=False)
        gradient_magnitude = np.zeros((self.n_sectors, ))
        for r in images:
            x = convolve2d(r, SOBAL_KERNEL, mode='same')
            y = convolve2d(r, SOBAL_KERNEL.T, mode='same')

            gm = self.l2(x, y)

            for i in range(self.n_sectors):
                gradient_magnitude[i] += cp.sum(gm * self.weights[i]).get()

        return gradient_magnitude / (N * H * W) * self.n_sectors

def acutance(images):
    '''
    Compute the acutance of a stack of images

    args:
        images: (N, H, W) array of images of the same size (H, W)

    return:
        acutance: scalar, the mean of the gradient magnitude
    '''
    N, H, W = images.shape

    images = cp.array(images, dtype=cp.float32, copy=False)
    gradient_magnitude = 0
    for r in images:
        x = convolve2d(r, SOBAL_KERNEL, mode='valid')
        y = convolve2d(r, SOBAL_KERNEL.T, mode='valid')

        gradient_magnitude += l2_sum(x, y).get()

    # mean of the gradient magnitude
    return gradient_magnitude / (N * H * W)

def padded_recon(sinogram, theta, center, pad_width):
    '''
    Pad the sinogram and perform filtered backprojection reconstruction

    args:
        sinogram: (N, n_angles, M) sinogram
        theta: (n_angles,) array of the theta in radians
        center: center of rotation, if None, it is automatically calculated
        pad_width: padding width 

    return:
        recon: (N, M, M) reconstructed images
    '''

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
    The function center_correction() uses Bayesian Optimization to correct shifts in the center of a given sinogram.

    args:
        sinogram: (N, n_angles, M) The z indices in the sinogram should be provided in ascending order.
        z_indices: (N,) The z indices of the images in the sinogram, starting from 0.
        total_z: Integer value giving the total number of z slices.
        theta: (n_sections, n_angles) or (n_angles,) The original theta value in radians.
        center_range: A tuple (low, high) to indicates the range of the center shift.
        init_points: Optional, defines the number of initial points in Bayesian Optimization. Default value is set to 50.
        n_iter: Optional, sets the number of iterations in Bayesian Optimization. Default value is set to 300.

    return:
        corrected_center: (total_z, ) array of corrected center
    '''
    def loss(start_center, end_center):
        X = [z_indices[0], z_indices[-1]]
        y = [start_center, end_center]
        center_offset = interp1d(X, y, kind='linear')(z_indices)
        return -acutance(fbp.run(theta, center_offset, to_host=False))
        
    # make sure sinogram is 3D
    if sinogram.ndim == 2:
        sinogram = np.array(sinogram)[np.newaxis, ...]

    N, n_angles, width = sinogram.shape
    
    fbp = FBP(sinogram)

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
    center_shifts = interp1d([z_indices[0], z_indices[-1]], [start_center, end_center], kind='linear', fill_value=np.nan if total_z == 1 else 'extrapolate')(np.arange(total_z))
    return center_shifts + width / 2

def theta_correction(sinogram, assumed_theta, center, n_keypoints, shift_range, n_sectors=1, n_iter=50):
    '''
    Correct theta shift in sinogram.

    args:
        sinogram: (N, n_angles, M)
        assumed_theta: (n_angles, ) array of the original theta in radians.
        center: (N, ) giving the center of rotation, if None, it is automatically calculated.
        n_keypoints: Integer value giving the number of keypoints.
        shift_range: A tuple of (low, high) to indicate the range of shift in radians.
        n_sectors: Integer value giving the number of sectors. Default value is set to 6.
        n_iter: Optional, sets the number of iterations in Bayesian Optimization. Default value is set to 300.

    return:
        corrected_theta: (n_angles,) array of corrected theta
    '''
    def interp_theta(keypoints):
        '''
        Interpolate keypoints to get theta.
        '''
        f = interp1d(np.linspace(0, n_angles-1, n_keypoints), keypoints, kind='cubic')
        return f(np.arange(n_angles)) + assumed_theta
    
    def calcualte_acutance(keypoints):
        '''
        Calculate acutance of a given theta.
        '''
        keypoints = np.array([keypoints[f'input{i}'] for i in range(n_keypoints)])
        theta = interp_theta(keypoints)
        return circular_acutance(fbp.run(theta, center_offset, to_host=False)) * -1
    
    # make sure sinogram is 3D
    if sinogram.ndim == 2:
        sinogram = np.array(sinogram)[np.newaxis, ...]

    assumed_theta = np.array(assumed_theta).reshape(-1)

    N, n_angles, width = sinogram.shape
    center_offset = 0 if center is None else center - width / 2

    fbp, circular_acutance = FBP(sinogram), CircularAcutance(width, n_sectors=n_sectors)

    # Bounded region of parameter space
    pbounds = {'input' + str(i): shift_range for i in range(n_keypoints)}

    optimizers = [BayesianOptimization(f=None, pbounds=pbounds, verbose=1, allow_duplicate_points=True) for _ in range(n_sectors)]

    # probe the assumed center and theta
    probe_params = {'input' + str(i): 0 for i in range(n_keypoints)}
    for optimizer, probe_loss in zip(optimizers, calcualte_acutance(probe_params)):
        optimizer.register(params=probe_params, target=probe_loss)

    utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

    for _ in (pbar := tqdm(range(n_iter), desc='theta correction - loss: -')):
        suggestions = [optimizer.suggest(utility) for optimizer in optimizers]
        losses = np.array([calcualte_acutance(suggestion) for suggestion in suggestions])

        for i, optimizer in enumerate(optimizers):
            for suggestion, loss in zip(suggestions, losses):
                optimizer.register(params=suggestion, target=loss[i])
        
        pbar.set_description(f'theta correction - loss: {losses.sum() / n_sectors:.6f}')

    return np.array([interp_theta(np.array(list(optimizer.max['params'].values()))) for optimizer in optimizers])
