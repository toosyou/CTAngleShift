import math
import numpy as np

from numba import cuda

@cuda.jit('void(float64[:, :, :], float64, float64[:], float64[:], float64[:, :, :], boolean)')
def cuda_bp_kernel(sinogram, ztilt_theta_sin, theta_sin, theta_cos, recon, circle=True):
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

    # Apply CUDA stream
    stream = cuda.stream()  # Create a new CUDA stream
    sinogram = cuda.to_device(np.ascontiguousarray(sinogram), stream)
    theta_sin, theta_cos = cuda.to_device(np.sin(theta), stream), cuda.to_device(np.cos(theta), stream)

    # prepare output
    recon = cuda.device_array((N, M, M), stream=stream)

    # prepare grid
    threadsperblock = (1, 32, 32)
    blockspergrid_x = math.ceil(N / threadsperblock[0])
    blockspergrid_y = math.ceil(M / threadsperblock[1])
    blockspergrid_z = math.ceil(M / threadsperblock[2])
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    # run kernel
    cuda_bp_kernel[blockspergrid, threadsperblock, stream](sinogram, np.sin(ztilt_theta), theta_sin, theta_cos, recon)
    # This will initiate the copy back to host, but will return control to CPU immediately
    recon_host_future = recon.copy_to_host(stream=stream)

    # Make sure we wait for the stream to complete before accessing on the host
    stream.synchronize()

    # Now it's safe to access
    recon = recon_host_future

    if circle:
        # crop to circle for the last two dimensions
        center = M / 2
        x, y = np.meshgrid(np.arange(M) - center, np.arange(M) - center)
        recon[:, ~(x ** 2 + y ** 2 <= center ** 2)] = 0

    return recon
    
class TiltedFBP:
    def __init__(self, sinogram, circle=True, filtered=True):
        '''
        Compute the backprojection of a tilted sinogram

        args:
            sinogram: (N, n_angles, M) sinogram
            circle: bool, whether to crop the reconstruction to a circle
            filtered: bool, whether to apply ramp filter
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

    def apply_filter(self):
        # padding values
        final_width = max(64, int(2 ** np.ceil(np.log2(2 * self.M))))
        pad_width = final_width - self.M

        # pad sinogram
        self.sinogram = np.pad(self.sinogram, [(0, 0), (0, 0), (0, pad_width)], mode='constant', constant_values=0)

        # apply filter
        self.sinogram = np.fft.fft(self.sinogram, axis=-1) * rampfilter(final_width)
        self.sinogram = np.real(np.fft.ifft(self.sinogram, axis=-1))[:, :, :self.M]

    def to_device(self):
        self.stream = cuda.stream()  # Create a new CUDA stream
        self.sinogram = cuda.to_device(np.ascontiguousarray(self.sinogram), self.stream)
        self.recon = cuda.device_array((self.N, self.M, self.M), stream=self.stream)

    def run(self, ztilt_theta, theta, copy_to_host=True):
        theta_sin = cuda.to_device(np.sin(theta), self.stream)
        theta_cos = cuda.to_device(np.cos(theta), self.stream)

        # run kernel
        cuda_bp_kernel[self.blockspergrid, self.threadsperblock, self.stream](self.sinogram, np.sin(ztilt_theta), theta_sin, theta_cos, self.recon, self.circle)
        # This will initiate the copy back to host, but will return control to CPU immediately
        if copy_to_host:
            recon_host_future = self.recon.copy_to_host(stream=self.stream)

        # Make sure we wait for the stream to complete before accessing on the host
        self.stream.synchronize()

        # Now it's safe to access
        if copy_to_host:
            self.recon = recon_host_future

        return self.recon

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