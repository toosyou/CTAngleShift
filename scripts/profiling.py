import time
from functools import wraps

import tomopy
import numpy as np

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ztilt

import cuda_recon

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

@timeit
def tomopy_padded_recon(sinogram, theta, center, pad_width):
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

@timeit
def python_cuda_recon(tilted_fbp, theta):
    return tilted_fbp.run(0, theta)

@timeit
def c_cuda_recon(fbp, theta):
    return fbp.run(theta)

def main():
    sample_sinogram = np.load('./notebooks/sample_sinogram.npy')
    n_angles, N, M = sample_sinogram.shape
    sample_sinogram = np.swapaxes(sample_sinogram, 0, 1) # (N, n_angles, M)
    angles = np.linspace(0, np.pi, n_angles)
    center = None
    pad_width = M // 4 + 1

    fbp = cuda_recon.FBP(sample_sinogram)
    c_cuda_recon(fbp, angles)

    tomopy_padded_recon(sample_sinogram, angles, center, pad_width)

    tilted_fbp = ztilt.TiltedFBP(sample_sinogram, circle=True, filtered=True)
    python_cuda_recon(tilted_fbp, angles)

if __name__ == '__main__':
    main()