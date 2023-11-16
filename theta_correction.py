import tomopy
import numpy as np

from scipy.interpolate import interp1d

import cupy as cp
from cupyx.scipy.signal import convolve2d

from bayes_opt import BayesianOptimization

def acutance(images):
    '''
    Compute the acutance of a stack of images

    args:
        images: (N, H, W) array of images of the same size (H, W)

    return:
        acutance: scalar, the mean of the gradient magnitude
    '''

    # sobal kernel
    kernel = cp.array([
                    [1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1],
                ], dtype=cp.float32)

    images = cp.array(images, dtype=cp.float32)
    gradient_magnitude = 0
    for r in images:
        gradient_magnitude += ((convolve2d(r, kernel, mode='valid') ** 2 + convolve2d(r, kernel.T, mode='valid') ** 2) ** 0.5).mean().get()

    # mean of the gradient magnitude
    return gradient_magnitude / len(images)

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
        theta: (n_angles,) The original theta value in radians.
        center_range: A tuple (low, high) to indicates the range of the center shift.
        init_points: Optional, defines the number of initial points in Bayesian Optimization. Default value is set to 50.
        n_iter: Optional, sets the number of iterations in Bayesian Optimization. Default value is set to 300.

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
        theta: (n_angles,) array of the original theta in radians.
        center: (N, ) giving the center of rotation, if None, it is automatically calculated.
        n_keypoints: Integer value giving the number of keypoints.
        shift_range: A tuple of (low, high) to indicate the range of shift in radians.
        init_points: Optional, defines the number of initial points in Bayesian Optimization. Default value is set to 50.
        n_iter: Optional, sets the number of iterations in Bayesian Optimization. Default value is set to 300.

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
