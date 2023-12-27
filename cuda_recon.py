import sys

import numpy as np
import cupy as cp

bp_kernel = cp.RawKernel(r'''
extern "C" __global__
void bp_kernel(const cudaTextureObject_t sinogram, 
                const float *theta_sin,
                const float *theta_cos, 
                const float center_offset, 
                const int n_angles, 
                const int M, float *recon){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
                         
    if (x >= M || y >= M) return;

    float r = (float)(M-1) / 2.0f;
    float x_centered = x - r;
    float y_centered = y - r;
    
    // crop to a circle
    if (x_centered * x_centered + y_centered * y_centered > r * r) return; 

    float value = 0.;
    for (int i = 0; i < n_angles; i++){
        const float sy = x_centered * theta_cos[i] - y_centered * theta_sin[i] + r + center_offset;
        value += tex2D<float>(sinogram, sy + 0.5f, (float)i + 0.5f);
    }
    recon[x * M + y] = value / (2.0f * (float)n_angles);
}        
''', 'bp_kernel')

texture_readback = cp.RawKernel(r'''
extern "C" __global__
void texture_readback(const cudaTextureObject_t sinogram,
                        const int n_angles,
                        const int M,
                        float* output){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= n_angles || y >= M) return;
    output[x * M + y] = tex2D<float>(sinogram, y + 0.5f, x + 0.5f);
}''', 'texture_readback')

class FBP:
    def __init__(self, sinogram):
        self.N, self.n_angles, self.M = sinogram.shape

        sinogram = self.apply_filter(sinogram)

        self.sinograms = [self.create_cuda_texture(s) for s in sinogram]

    def create_cuda_texture(self, sinogram):
        channel_desc = cp.cuda.texture.ChannelFormatDescriptor(32, 0, 0, 0, cp.cuda.runtime.cudaChannelFormatKindFloat)
        sinogram_tex = cp.array(sinogram, dtype=cp.float32)

        resource_desc = cp.cuda.texture.ResourceDescriptor(cp.cuda.runtime.cudaResourceTypePitch2D,
                                                           arr=sinogram_tex, chDesc=channel_desc, 
                                                           width=self.M, height=self.n_angles, pitchInBytes=self.M // 128 * 128 * 4)

        texture_desc = cp.cuda.texture.TextureDescriptor([cp.cuda.runtime.cudaAddressModeBorder, 
                                                            cp.cuda.runtime.cudaAddressModeBorder],
                                                         cp.cuda.runtime.cudaFilterModeLinear,
                                                         cp.cuda.runtime.cudaReadModeElementType,
                                                         0)

        texture_obj = cp.cuda.texture.TextureObject(resource_desc, texture_desc)
        return texture_obj

    @staticmethod
    def rampfilter(size):
        n = np.concatenate((np.arange(1, size / 2 + 1, 2, dtype=int),
                            np.arange(size / 2 - 1, 0, -2, dtype=int)))
        f = np.zeros(size)
        f[0] = 0.25
        f[1::2] = -1 / (np.pi * n) ** 2
        return 2 * np.real(np.fft.fft(f))

    def apply_filter(self, sinogram):
        # padding values
        final_width = max(64, int(2 ** np.ceil(np.log2(2 * (self.M * 1.5)))))
        left_pad_width = self.M // 4 + 1
        right_pad_width = final_width - self.M - left_pad_width

        # pad sinogram
        sinogram = np.pad(sinogram, [(0, 0), (0, 0), (left_pad_width, right_pad_width)], mode='edge')

        # apply filter
        sinogram = np.fft.fft(sinogram, axis=-1) * self.rampfilter(final_width)
        sinogram = np.real(np.fft.ifft(sinogram, axis=-1))[:, :, left_pad_width: left_pad_width+self.M]

        return sinogram
    
    def read_back_test(self):
        test_sinogram = cp.zeros((self.n_angles, self.M), dtype=cp.float32)
        texture_readback((self.n_angles // 32, self.M // 32), (32, 32), (self.sinograms[0], self.n_angles, self.M, test_sinogram))
        return test_sinogram.get()

    def run(self, theta, center_offset=None, to_host=True):
        if center_offset is None: center_offset = [0.0] * self.N
        elif np.isscalar(center_offset): center_offset = [center_offset] * self.N

        theta = cp.array(theta, dtype=cp.float32)
        theta_sin, theta_cos = cp.sin(theta), cp.cos(theta)

        recons = list()
        for sinogram, offset in zip(self.sinograms, center_offset):
            recon = cp.zeros((self.M, self.M), dtype=cp.float32)
            grid_size = np.ceil(self.M / 32).astype(int)
            bp_kernel((grid_size, grid_size), (32, 32), 
                      (sinogram, theta_sin, theta_cos, 
                       np.float32(offset), self.n_angles, self.M, recon))
            recons.append(recon)

        recons = cp.stack(recons)
        if to_host: recons = recons.get()
        return recons
    
if __name__ == "__main__":
    sample_sinogram = np.load('./notebooks/sample_sinogram.npy')
    n_angles, N, M = sample_sinogram.shape
    sample_sinogram = np.swapaxes(sample_sinogram, 0, 1) # (N, n_angles, M)
    angles = np.linspace(0, np.pi, n_angles)
    center = M // 2 - 18
    pad_width = M // 4 + 1

    fbp = FBP(sample_sinogram)
    recon = fbp.run(angles, center)

    print(recon.shape)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    plt.imshow(recon[0], cmap='gray')
    plt.axis('off')  # to hide the axis
    plt.savefig('recon.png', bbox_inches='tight', pad_inches=0)
    plt.close()
