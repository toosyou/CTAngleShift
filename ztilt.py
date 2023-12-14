import math
import numpy as np

from numba import cuda

@cuda.jit('void(float64[:, :, :], float64, float64[:], float64[:], float64[:, :, :])')
def cuda_bp_kernel(sinogram, ztilt_theta_sin, theta_sin, theta_cos, recon):
    '''
    Compute the backprojection of a tilted sinogram on GPU

    args:
        sinogram: (N, n_angles, M)
        ztilt_theta_sin: scalar, the tilt angle of the z axis in radians
        theta_sin: (n_angles,) sin of the angles in radians
        theta_cos: (n_angles,) cos of the angles in radians
        recon: (N, M, M) output reconstruction
    '''
    z, x, y = cuda.grid(3)
    if z >= recon.shape[0] or x >= recon.shape[1] or y >= recon.shape[2]: return

    N, n_angles, M = sinogram.shape

    recon[z, x, y] = 0
    center = M / 2
    for i, (tsin, tcos) in enumerate(zip(theta_sin, theta_cos)):
        sy = (x - center) * tcos - (y - center) * tsin + center

        z_delta = (x - center) * tsin + (y - center) * tcos
        sz = z + z_delta * ztilt_theta_sin
        
        if sy >= 0 and sy < M and sz >= 0 and sz < N:
            # 3d interpolate
            sy_bottom = max(0, math.floor(sy))
            sz_bottom = max(0, math.floor(sz))
            sy_top = min(M-1, math.ceil(sy))
            sz_top = min(N-1, math.ceil(sz))

            sy_bottom, sz_bottom = int(sy_bottom), int(sz_bottom)
            sy_top, sz_top = int(sy_top), int(sz_top)
            
            c000 = sinogram[sz_bottom, i, sy_bottom]
            c100 = sinogram[sz_top, i, sy_bottom]
            c010 = sinogram[sz_bottom, i, sy_top]
            c001 = sinogram[sz_bottom, i+1, sy_bottom]
            c101 = sinogram[sz_top, i+1, sy_bottom]
            c011 = sinogram[sz_bottom, i+1, sy_top]
            c110 = sinogram[sz_top, i, sy_top]
            c111 = sinogram[sz_top, i+1, sy_top]
            
            wy = sy - sy_bottom
            wz = sz - sz_bottom
            wx = i+1 - i
            
            c00 = c000 * (1 - wy) + c010 * wy
            c01 = c001 * (1 - wy) + c011 * wy
            c10 = c100 * (1 - wy) + c110 * wy
            c11 = c101 * (1 - wy) + c111 * wy
            
            c0 = c00 * (1 - wz) + c01 * wz
            c1 = c10 * (1 - wz) + c11 * wz
            
            c = c0 * (1 - wx) + c1 * wx

            # update the reconstruction
            recon[z, x, y] += c

def rampfilter(size):
    n = np.concatenate((np.arange(1, size / 2 + 1, 2, dtype=int),
                        np.arange(size / 2 - 1, 0, -2, dtype=int)))
    f = np.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2
    return 2 * np.real(np.fft.fft(f))

def tilted_FBP(sinogram, ztilt_theta, theta, circle=True, filtered=True):
    '''
    Compute the backprojection of a tilted sinogram

    args:
        sinogram: (N, n_angles, M) sinogram
        ztilt_theta: scalar, the tilt angle of the z axis in radians
        theta: (n_angles,) array of the theta in radians
        circle: bool, whether to crop the reconstruction to a circle
        filtered: bool, whether to apply ramp filter
    '''
    
    N, n_angles, M = sinogram.shape

    if filtered:
        # padding values
        final_width = max(64, int(2 ** np.ceil(np.log2(2 * M))))
        pad_width = final_width - M

        # pad sinogram
        sinogram = np.pad(sinogram, [(0, 0), (0, 0), (0, pad_width)], mode='constant', constant_values=0)

        # apply filter
        sinogram = np.fft.fft(sinogram, axis=-1) * rampfilter(final_width)
        sinogram = np.real(np.fft.ifft(sinogram, axis=-1))[:, :, :M] 

    # move to device
    sinogram = cuda.to_device(np.ascontiguousarray(sinogram))
    theta_sin, theta_cos = cuda.to_device(np.sin(theta)), cuda.to_device(np.cos(theta))

    # prepare output
    recon = cuda.device_array((N, M, M))

    # prepare grid
    threadsperblock = (1, 32, 32)
    blockspergrid_x = math.ceil(N / threadsperblock[0])
    blockspergrid_y = math.ceil(M / threadsperblock[1])
    blockspergrid_z = math.ceil(M / threadsperblock[2])
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    # run kernel
    cuda_bp_kernel[blockspergrid, threadsperblock](sinogram, np.sin(ztilt_theta), theta_sin, theta_cos, recon)
    recon = recon.copy_to_host()

    if circle:
        # crop to circle for the last two dimensions
        center = M / 2
        x, y = np.meshgrid(np.arange(M) - center, np.arange(M) - center)
        recon[:, ~(x ** 2 + y ** 2 <= center ** 2)] = 0

    return recon
    
if __name__ == '__main__':
    import time

    # random sinogram
    N = 10
    n_angles = 601
    M = 2560

    sinogram = np.random.rand(N, n_angles, M)
    ztilt_theta = 0.5
    theta = np.linspace(0, np.pi, n_angles)

    # run cuda
    start = time.time()
    recon = tilted_FBP(sinogram, ztilt_theta, theta)
    end = time.time()
    print('cuda time:', end - start)