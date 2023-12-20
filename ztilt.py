import math
import numpy as np

from numba import cuda

@cuda.jit('void(float64[:, :, :], float64, float64[:], float64[:], float64[:, :, :], boolean, boolean)')
def cuda_bp_kernel(sinogram, ztilt_theta_sin, theta_sin, theta_cos, recon, circle=True, only_middle=False):
    '''
    Compute the backprojection of a tilted sinogram on GPU. 
    The sinogram is assumed to be aligned properly so that the center of rotation is M / 2.

    args:
        sinogram: (N, n_angles, M)
        ztilt_theta_sin: scalar, the tilt angle of the z axis in radians
        theta_sin: (n_angles,) sin of the angles in radians
        theta_cos: (n_angles,) cos of the angles in radians
        recon: (N, M, M) output reconstruction
        circle: bool, whether to crop the reconstruction to a circle
        only_middle: bool, whether to only compute the middle layer of the reconstruction
    '''
    z, x, y = cuda.grid(3)
    if z >= recon.shape[0] or x >= recon.shape[1] or y >= recon.shape[2]: return
    if only_middle and z != recon.shape[0] // 2: return

    N, n_angles, M = sinogram.shape

    recon[z, x, y] = 0
    center = M / 2
    x_centered = x - center
    y_centered = y - center
    if circle and x_centered ** 2 + y_centered ** 2 > center ** 2: return

    for i, (tsin, tcos) in enumerate(zip(theta_sin, theta_cos)):
        sy = x_centered * tcos - y_centered * tsin + center

        z_delta = x_centered * tsin + y_centered * tcos
        sz = z + z_delta * ztilt_theta_sin
        
        if sy >= 0 and sy < M and sz >= 0 and sz < N:
            # 3d interpolate
            sy_bottom = int(max(0, math.floor(sy)))
            sz_bottom = int(max(0, math.floor(sz)))
            sy_top = int(min(M-1, math.ceil(sy)))
            sz_top = int(min(N-1, math.ceil(sz)))

            c0 = sinogram[sz_bottom, i, sy_bottom]
            c1 = sinogram[sz_top, i, sy_bottom]
            c2 = sinogram[sz_bottom, i, sy_top]
            c3 = sinogram[sz_top, i, sy_top]

            wy = sy - sy_bottom
            wz = sz - sz_bottom

            c00 = c0 * (1 - wy) + c2 * wy
            c10 = c1 * (1 - wy) + c3 * wy
            c = c00 * (1 - wz) + c10 * wz

            # update the reconstruction
            recon[z, x, y] += c
    recon[z, x, y] *= math.pi / (2 * n_angles)
    
class TiltedFBP:
    def __init__(self, sinogram, circle=True, filtered=True):
        '''
        Compute the backprojection of a tilted sinogram

        args:
            sinogram: (N, n_angles, M) sinogram
            circle: bool, whether to crop the reconstruction to a circle
            filtered: bool, whether to apply ramp filter
            pad: bool, whether to pad the sinogram before reconstruction
        '''
        self.N, self.n_angles, self.M = sinogram.shape

        self.sinogram = sinogram
        self.circle = circle
        self.filtered = filtered
        self.stream = None
        self.recon = None

        self.threadsperblock = (1, 32, 32)
        blockspergrid_x = math.ceil(self.N / self.threadsperblock[0])
        blockspergrid_y = math.ceil(self.M / self.threadsperblock[1])
        blockspergrid_z = math.ceil(self.M / self.threadsperblock[2])
        self.blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

        if filtered: self.apply_filter()

        self.to_device()

    @staticmethod
    def rampfilter(size):
        n = np.concatenate((np.arange(1, size / 2 + 1, 2, dtype=int),
                            np.arange(size / 2 - 1, 0, -2, dtype=int)))
        f = np.zeros(size)
        f[0] = 0.25
        f[1::2] = -1 / (np.pi * n) ** 2
        return 2 * np.real(np.fft.fft(f))

    def apply_filter(self):
        # padding values
        final_width = max(64, int(2 ** np.ceil(np.log2(2 * (self.M * 1.5)))))
        left_pad_width = self.M // 4 + 1
        right_pad_width = final_width - self.M - left_pad_width

        # pad sinogram
        self.sinogram = np.pad(self.sinogram, [(0, 0), (0, 0), (left_pad_width, right_pad_width)], mode='edge')

        # apply filter
        self.sinogram = np.fft.fft(self.sinogram, axis=-1) * self.rampfilter(final_width)
        self.sinogram = np.real(np.fft.ifft(self.sinogram, axis=-1))[:, :, left_pad_width: left_pad_width+self.M]

    def to_device(self):
        self.stream = cuda.stream()  # Create a new CUDA stream
        self.sinogram = cuda.to_device(np.ascontiguousarray(self.sinogram), self.stream)
        self.recon = cuda.device_array((self.N, self.M, self.M), stream=self.stream)

    def run(self, ztilt_theta, theta, value_range=None, copy_to_host=True, only_middle=False):
        '''
        Compute the backprojection of a tilted sinogram

        args:
            ztilt_theta: scalar, the tilt angle of the z axis in radians
            theta: (n_angles,) array of the theta in radians
            value_range: A tuple of (low, high) to indicate the range of values. If None, the range is set to the min and max of the reconstruction.
            copy_to_host: bool, whether to copy the reconstruction back to host
            only_middle: bool, whether to only compute the middle layer of the reconstruction

        return:
            recon: (N, M, M) reconstructed images
        '''
        theta_sin = cuda.to_device(np.sin(theta), self.stream)
        theta_cos = cuda.to_device(np.cos(theta), self.stream)

        # run kernel
        cuda_bp_kernel[self.blockspergrid, self.threadsperblock, self.stream](self.sinogram, np.sin(ztilt_theta), theta_sin, theta_cos, self.recon, self.circle, only_middle)
        # This will initiate the copy back to host, but will return control to CPU immediately
        if copy_to_host:
            recon_host_future = self.recon.copy_to_host(stream=self.stream)

        # Make sure we wait for the stream to complete before accessing on the host
        self.stream.synchronize()

        if copy_to_host:
            if value_range is not None:
                vmin, vmax = value_range
                recon_host_future = (recon_host_future - vmin) / (vmax - vmin)
                return np.clip(recon_host_future, 0, 1)
            return recon_host_future
        return self.recon

def ztilt_correction(sinogram, theta, ztilt_range=(-0.0017, 0.0017), init_points=10, n_iter=50):
    '''
    Correct ztilt angle in sinogram.

    args:
        sinogram: (K, N, n_angles, M)
        theta: (n_angles,) array of the original theta in radians.
        ztilt_range: A tuple of (low, high) to indicate the range of ztilt angle in radians.
        init_points: Optional, defines the number of initial points in Bayesian Optimization. Default value is set to 10.
        n_iter: Optional, sets the number of iterations in Bayesian Optimization. Default value is set to 100.

    return:
        corrected_tilt: scalar of corrected tilt angle in radians
    '''
    import cupy as cp
    from cupyx.scipy.signal import convolve2d

    from bayes_opt import BayesianOptimization

    def loss(ztilt):
        center_recon = [tfbp.run(ztilt, theta, copy_to_host=False, only_middle=True) for tfbp in tilted_fbps]

        kernel = cp.array([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1],
                            ], dtype=cp.float32)
    
        center_recon = [cp.asarray(r) for r in center_recon]
        center_recon = [r[N // 2] for r in center_recon]

        acutances = [((convolve2d(r, kernel, mode='valid') ** 2 + convolve2d(r, kernel.T, mode='valid') ** 2) ** 0.5).mean().get() for r in center_recon]
        return -np.mean(acutances)

    K, N, n_angles, M = sinogram.shape

    tilted_fbps = [TiltedFBP(sino) for sino in sinogram]
    
    # Bounded region of parameter space
    pbounds = {
        'ztilt': ztilt_range,
    }
    optimizer = BayesianOptimization(f=loss, pbounds=pbounds, verbose=1, allow_duplicate_points=True)

    # probe the assumed center and theta
    optimizer.probe({'ztilt': 0})

    optimizer.maximize(
        init_points = init_points,
        n_iter = n_iter,
    )

    return optimizer.max['params']['ztilt']

if __name__ == '__main__':
    import time

    # random sinogram
    N = 10
    n_angles = 601
    M = 2560

    sinogram = np.random.rand(N, n_angles, M)
    ztilt_theta = 0.5
    theta = np.linspace(0, np.pi, n_angles)

    # time both itialization and computation
    start = time.time()
    tiltedfbp = TiltedFBP(sinogram)
    initial_time = time.time()
    recon = tiltedfbp.run(ztilt_theta, theta)
    end = time.time()
    print('initialization time:', initial_time - start)
    print('computation time:', end - initial_time)