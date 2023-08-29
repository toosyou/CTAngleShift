import tomopy
import numpy as np

from scipy.interpolate import interp1d

import cupy as cp
from cupyx.scipy.signal import convolve2d

from bayes_opt import BayesianOptimization

def theta_correction(sinogram, theta, n_keypoints, shift_range, init_points=50, n_iter=300):
    '''
    Correct theta shift in sinogram.

    args:
        sinogram: (n_angles, N, M)
        theta: (n_angles,) array of the original theta in radians
        n_keypoints: number of keypoints
        shift_range: range of shift in radians
        init_points: number of initial points in Bayesian Optimization
        n_iter: number of iterations in Bayesian Optimization

    return:
        corrected_theta: (n_angles,) array of corrected theta
    '''
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
        
    def interp_theta(keypoints):
        '''
        Interpolate keypoints to get theta.
        '''
        f = interp1d(np.linspace(0, n_angles, n_keypoints), keypoints, kind='cubic')
        return f(np.arange(n_angles)) + assumed_theta

    def loss(**args):
        theta = interp_theta(np.array(list(args.values())))
        
        recon = tomopy.recon(sinogram,
                         theta,
                         center=None,
                         algorithm=tomopy.astra,
                         sinogram_order=True,
                         options={'proj_type': 'cuda', 'method': 'FBP_CUDA'},
                         ncore=1)
        recon = tomopy.circ_mask(recon, axis=0, ratio=0.95, val=0) # (N, M, M)
        return -acutance(recon)
        
    # make sure sinogram is 3D
    if len(sinogram.shape) == 2:
        sinogram = np.array(sinogram)[np.newaxis, ...]

    assumed_theta, n_angles = theta, len(theta)
    
    # Bounded region of parameter space
    pbounds = {'input' + str(i): shift_range for i in range(n_keypoints)}
    optimizer = BayesianOptimization(f=loss, pbounds=pbounds, verbose=1)

    # probe the assumed theta
    optimizer.probe(params={'input' + str(i): 0 for i in range(n_keypoints)})

    optimizer.maximize(
        init_points = init_points,
        n_iter = n_iter,
    )

    return interp_theta(np.array(list(optimizer.max['params'].values())))