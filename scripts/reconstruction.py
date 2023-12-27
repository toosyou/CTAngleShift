import os
import argparse
import multiprocessing as mp

import cv2
import pywt
import tomopy
import numpy as np
import numpy.ma as ma

from numpy.fft import fft, ifft, fftshift, ifftshift
from PIL import Image
from glob import glob
from tqdm import tqdm
from scipy.ndimage import median_filter
from loguru import logger

from multiprocessing import Pool

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from theta_correction import theta_correction, center_correction

def read_background(proj_dir):
    '''
    Reads background images from a directory.

    args:
        proj_dir: Directory path containing background images. The background images should be named as DF*.tif and FF*.tif.

    returns:
        df: Median image of DF*.tif files.
        ff: Median image of FF*.tif files.
    '''
    df_filenames = glob(proj_dir + '/DF*.tif')
    ff_filenames = glob(proj_dir + '/FF*.tif')

    df = np.median([np.array(Image.open(fn)) for fn in df_filenames], axis=0)
    ff = np.median([np.array(Image.open(fn)) for fn in ff_filenames], axis=0)

    return df, ff

def read_normed(filename, df_projection, ff_projection):
    '''
    Reads and processes a projection image.

    args:
        filename: File path of the projection image.
        df_projection: Dark field projection.
        ff_projection: Flat field projection.

    returns:
        p: Processed projection image.
    '''

    p = np.array(Image.open(filename), dtype='float32')
    p = (p - df_projection) / (ff_projection - df_projection)
    
    return p

def read_normed_mp(args):
    fn, df_projection, ff_projection = args
    return read_normed(fn, df_projection, ff_projection)

def align_projection(projections, align_matrix=None, centers=None):
    '''
    Aligns a projection image inplace using an alignment matrix or centers in stacks.

    args:
        projections: Projection images of shape (n_angles, z, M).

    returns:
        Aligned projection images.
    '''

    assert align_matrix is not None or centers is not None, 'Either align_matrix or centers should be given.'

    _, z, M = projections.shape

    if align_matrix is None:
        theta = np.arctan((centers[-1] - centers[0]) / (z - 1))
        align_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])

        xshift = M / 2 - (align_matrix @ np.array([centers[0], 0, 1]))[0]
        yshift = -(align_matrix @ np.array([centers[0], 0, 1]))[1]
        align_matrix[0, 2] = xshift
        align_matrix[1, 2] = yshift
        align_matrix = np.linalg.inv(align_matrix)[:2]

    STACK_SIZE = 500
    for i in range(0, projections.shape[0], STACK_SIZE):
        end = min(i + STACK_SIZE, projections.shape[0])
        p = cv2.warpAffine(np.moveaxis(projections[i: end], 0, -1), # (H, W, N) 
                            align_matrix, 
                            (projections.shape[2], projections.shape[1]), 
                            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                            borderMode=cv2.BORDER_REPLICATE)
        projections[i: end] = np.moveaxis(p, -1, 0)

    return projections

def find_align_matrix(projections, df_projection, ff_projection):
    '''
    Finds the alignment matrix for projection images.

    args:
        proj_files: List of projection file paths.
        df_projection: Dark field projection.
        ff_projection: Flat field projection.

    returns:
        align_matrix: Alignment matrix.
        angles: Array of angles for alignment.
    '''
    def align(im1, im2, motion_model=cv2.MOTION_EUCLIDEAN):
        '''
        Aligns two images using ECC algorithm.

        args:
            im1: (H, W) First input image.
            im2: (H, W) Second input image.
            motion_model: Motion model type. Default is cv2.MOTION_EUCLIDEAN.

        returns:
            warp_matrix: Transformation matrix.
            im2_aligned: Aligned second image.
        '''

        im1, im2 = im1.astype('float32'), im2.astype('float32')
        
        # Find size of image1
        sz = im1.shape

        # Define the motion model
        warp_mode = motion_model
        
        # Define 2x3 or 3x3 matrices and initialize the matrix to identity
        if warp_mode == cv2.MOTION_HOMOGRAPHY :
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else :
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        # Specify the number of iterations.
        number_of_iterations = 5000

        # Specify the threshold of the increment
        
        termination_eps = 1e-10

        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

        # Run the ECC algorithm. The results are stored in warp_matrix.
        (cc, warp_matrix) = cv2.findTransformECC(im1, im2, warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=1)
        
        if warp_mode == cv2.MOTION_HOMOGRAPHY :
            # Use warpPerspective for Homography
            im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else :
            # Use warpAffine for Translation, Euclidean and Affine
            im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        return warp_matrix, im2_aligned

    n_angles, height, width = projections.shape

    # load the first and a few last projections
    hcut, wcut = height // 4, width // 4

    first = projections[0]
    lasts = projections[-3:][::-1]
    align_result = [align(first, np.flip(i_last, axis=-1)) for i_last in lasts] # [(warp_matrix, aligned_image), ...]

    diffs = [np.abs(first - aligned_image)[hcut: -hcut, wcut: -wcut].mean() for _, aligned_image in align_result]

    warp_matrix = align_result[np.argmin(diffs)][0]

    angles = np.linspace(0, np.pi, n_angles)
    angles *= np.pi / angles[-1-np.argmin(diffs)] # make sure the most aligned image is at 180 degree

    theta = (np.arccos(warp_matrix[0, 0]) + np.arcsin(warp_matrix[0, 1])) / 2
    align_matrix = np.array([[np.cos(-theta), np.sin(-theta), -warp_matrix[0, 2] / 2], 
                                [-np.sin(-theta), np.cos(-theta), warp_matrix[1, 2] / 2]])
    
    return align_matrix, angles

def read_projections(proj_files, df_projection, ff_projection, align=False):
    '''
    Reads aligned projection images.

    args:
        proj_files: List of projection file paths.
        df_projection: Dark field projection.
        ff_projection: Flat field projection.
        align: Whether to align the projection images. (default: False)

    returns:
        projections: Aligned projection images.
        angles: Array of angles for the projections.
    '''

    n_angles, height, width = len(proj_files), df_projection.shape[0], ff_projection.shape[1]
    angles = np.linspace(0, np.pi, n_angles)

    # number of process
    num_processes = mp.cpu_count()
    projections = np.zeros((n_angles, height, width), dtype='float32')

    with Pool(num_processes) as pool:
        args_list = [(fn, df_projection, ff_projection) for i, fn in enumerate(proj_files)]
        for i, p in enumerate(tqdm(pool.imap(read_normed_mp, args_list), total=n_angles, desc='Reading projections')):
            projections[i] = p

    if align:
        align_matrix, angles = find_align_matrix(projections, df_projection, ff_projection)
        projections = align_projection(projections, align_matrix)

    return projections, angles

def fft_wavelet_ring_removal(sinogram, level, wname='db5', sigma=1.5):
    '''
    Suppress horizontal stripe in a sinogram using the Fourier-Wavelet based
    method by Munch et al. [2]_.

    Parameters
    ----------
    sinogram : 2d array (n_angles, width)
        The two-dimensional array representig the image or the sinogram to de-stripe.

    level : int
        The highest decomposition level.

    wname : str, optional
        The wavelet type. Default value is ``db5``

    sigma : float, optional
        The damping factor in the Fourier space. Default value is ``1.5``

    Returns
    -------
    out : 2d array
        The resulting filtered image.

    References
    ----------
    .. [2] B. Munch, P. Trtik, F. Marone, M. Stampanoni, Stripe and ring artifact removal with
        combined wavelet-Fourier filtering, Optics Express 17(10):8567-8591, 2009.
    '''

    nrow, ncol = sinogram.shape

    # wavelet decomposition.
    cH = []; cV = []; cD = []

    for i in range(0, level):
        sinogram, (cHi, cVi, cDi) = pywt.dwt2(sinogram, wname)
        cH.append(cHi)
        cV.append(cVi)
        cD.append(cDi)

    # FFT transform of horizontal frequency bands
    for i in range(level):
        # FFT
        fcV=fftshift(fft(cV[i], axis=0))
        my, mx = fcV.shape

        # damping of vertical stripe information
        yy2  = (np.arange(-np.floor(my/2), -np.floor(my/2) + my))**2
        damp = - np.expm1( - yy2 / (2.0*(sigma**2)) )
        fcV  = fcV * np.tile(damp.reshape(damp.size, 1), (1,mx))

        #inverse FFT
        cV[i] = np.real( ifft( ifftshift(fcV), axis=0) )

    # wavelet reconstruction
    for i in  range(level-1, -1, -1):
        sinogram = sinogram[0:cH[i].shape[0], 0:cH[i].shape[1]]
        sinogram = pywt.idwt2((sinogram, (cH[i], cV[i], cD[i])), wname)

    return sinogram[0:nrow, 0:ncol]
    
def fft_wavelet_ring_removal_mp(args):
    i, sinogram, level = args
    return i, fft_wavelet_ring_removal(sinogram, level)

def ring_removal(projections, adv=False):
    '''
    Removes ring artifacts from projections.

    args:
        projections: Projection images.
        adv: Advanced ring removal flag. If set to True, Fourier-Wavelet based ring removal will be used.

    returns:
        Projections with ring artifacts removed.
    '''

    def simple_ring_removal(projections):
        angle_mean = projections.mean(axis=0)
        angle_mean_median = median_filter(angle_mean, size=(1, 11)) # (H, W)
        projections -= (angle_mean - angle_mean_median)
        return projections
    
    if adv:
        with Pool(mp.cpu_count()) as pool:
            args_list = [(i, projections[:, i, :], 3) for i in range(projections.shape[1])]
            for i, p in tqdm(pool.imap(fft_wavelet_ring_removal_mp, args_list), total=projections.shape[1], desc='Ring removal'):
                projections[:, i, :] = p
        return projections
    else:
        return simple_ring_removal(projections)

def negative_log(projections):
    '''
    Computes negative logarithm of projections.

    args:
        projections: Input projection images.

    returns:
        Projections after negative logarithm transformation.
    '''

    projections = np.where(projections > 0, projections, ma.array(projections, mask=projections <= 0).min(keepdims=True))
    return -np.log(projections)

def find_correct_centers(projections, angles, n_center_sample=5, shift_range=(-10, 10), init_points=50, n_iter=150):
    '''
    Finds correct centers for reconstruction using Bayesian Optimization.

    args:
        projections: Projection images.
        angles: Array of angles.
        n_center_sample: Number of center samples.
        shift_range: Range for center shift.
        init_points: Optional, defines the number of initial points in Bayesian Optimization. Default value is set to 50.
        n_iter: Optional, sets the number of iterations in Bayesian Optimization. Default value is set to 150.

    returns:
        Corrected centers for each projection.
    '''

    sample_indices = np.linspace(0, projections.shape[1], n_center_sample+2)[1:-1].astype(int)
    sample_sinogram = projections[:, sample_indices, :]

    return center_correction(sample_sinogram.transpose(1, 0, 2), sample_indices, projections.shape[1], angles, shift_range, init_points, n_iter)

def find_correct_angles(projections, angles, center=None, n_angle_sample=5, n_keypoints=8, shift_range=(-0.015, 0.015), init_points=50, n_iter=400):
    '''
    Finds correct angles for reconstruction.

    args:
        projections: Projection images.
        angles: Array of angles.
        center: Center positions.
        n_angle_sample: Number of angle samples.
        n_keypoints: Number of keypoints.
        shift_range: Range for angle shift.
        init_points: Optional, defines the number of initial points in Bayesian Optimization. Default value is set to 50.
        n_iter: Optional, sets the number of iterations in Bayesian Optimization. Default value is set to 300.

    returns:
        Corrected angles for reconstruction.
    '''

    sample_indices = np.linspace(0, projections.shape[1], n_angle_sample+2)[1:-1].astype(int)
    sample_sinogram = projections[:, sample_indices, :]

    if center is not None:
        center = center[sample_indices]

    return theta_correction(sample_sinogram.transpose(1, 0, 2), angles, center, n_keypoints, shift_range, init_points, n_iter)

def reconstruct(projections, angles, center=None, value_range=None):
    '''
    Reconstructs the 3D volume from projections.

    args:
        projections: Projection images.
        angles: Array of angles.
        center: Center positions. If None, the center will be automatically calculated.
        value_range: Range of values for normalization. If given, the reconstructed volume will be normalized to [0, 1].

    returns:
        Reconstructed 3D volume.
    '''

    pad_width = projections.shape[-1] // 4 + 1
    if center is not None: center = center + pad_width
    recon = tomopy.recon(np.pad(projections, ((0, 0), (0, 0), (pad_width, pad_width)), 'edge'),
                            angles,
                            center=center,
                            algorithm=tomopy.astra,
                            sinogram_order=False,
                            options={'proj_type': 'cuda', 'method': 'FBP_CUDA', 'extra_options': {'FilterType': 'hamming'}},
                            ncore=1)[:, pad_width:-pad_width, pad_width:-pad_width]
    recon = tomopy.circ_mask(recon, axis=0, ratio=1.0, val=0) # (N, M, M)

    if value_range is not None:
        vmin, vmax = value_range
        recon = (recon - vmin) / (vmax - vmin)
        recon = np.clip(recon, 0, 1)

    return recon

def find_value_range(projections, angles, center=None, n_range_sample=10):
    '''
    Finds value range for reconstruction.

    args:
        projections: Projection images.
        angles: Array of angles.
        center: Center positions.
        n_range_sample: Number of value range samples.

    returns:
        Minimum and maximum value range for reconstruction.
    '''

    sample_indices = np.linspace(0, projections.shape[1], n_range_sample+2)[1:-1].astype(int)
    sample_sinogram = projections[:, sample_indices, :]

    if center is not None:
        center = center[sample_indices]

    recon = reconstruct(negative_log(sample_sinogram), angles, center)

    valid = recon[recon != 0]
    vmin = np.percentile(valid[valid <= np.percentile(valid, 1)], 1)
    vmax = np.percentile(valid[valid >= np.percentile(valid, 99)], 99)

    return vmin, vmax

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Reconstruct sinogram to tomogram.')

    parser.add_argument('proj_dir', type=str, help='directory of projections')
    parser.add_argument('-r', dest='ring_removal', type=int, choices=[0, 1, 2], default=1, 
                        help='ring removal option: 0(no ring removal), 1(fast ring removal, default) or 2(Fourier-Wavelet based ring removal)')
    parser.add_argument('-a', dest='angle_shift', action='store_true', help='do angle shift correction')
    parser.add_argument('-c', dest='center_shift', action='store_true', help='do center shift correction')
    parser.add_argument('-b', dest='batch_size', type=int, help='batch size for reconstruction (default = 100)', default=100)
    parser.add_argument('-o', dest='output_dir', type=str, help='directory of output', default='./recons')

    args = parser.parse_args()

    # create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Finding projection filenames")
    proj_files = sorted(glob(args.proj_dir + '/tomo*.tif'))

    logger.info("Reading background")
    df_projection, ff_projection = read_background(args.proj_dir)

    logger.info("Reading projections")
    projections, angles = read_projections(proj_files, df_projection, ff_projection, align=not args.center_shift)

    centers = None
    if args.angle_shift and args.center_shift:
        for i in range(3):
            logger.info("Angle and center shift correction iteration {}/3".format(i))
            angles = find_correct_angles(projections, angles, centers, init_points=30, n_iter=150)
            centers = find_correct_centers(projections, angles, init_points=30, n_iter=100)
    elif args.center_shift:
        logger.info("Finding correct center")
        centers = find_correct_centers(projections, angles)
    elif args.angle_shift:
        logger.info("Finding correct angles")
        angles = find_correct_angles(projections, angles)

    if args.center_shift:
        logger.info("Aligning projections using found centers")
        projections = align_projection(projections, centers=centers)

    if args.ring_removal != 0:
        logger.info("Performing ring removal")
        projections = ring_removal(projections, args.ring_removal == 2)

    logger.info("Finding value range")
    vmin, vmax = find_value_range(projections, angles)

    # batch reconstruction
    logger.info("Starting batch reconstruction")
    for i in tqdm(range(0, projections.shape[1], args.batch_size), desc='Reconstructing'):
        end = min(i + args.batch_size, projections.shape[1])
        recon = reconstruct(negative_log(projections[:, i: end, :]), 
                            angles, None,
                            (vmin, vmax))
        recon = (recon * 65535).astype('uint16')

        # save to files
        for j, r in enumerate(recon):
            filename = os.path.join(args.output_dir, f'tomo{i+j:05d}.tif')
            Image.fromarray(r).save(filename)

    logger.info("Reconstruction finished, output dir: {}".format(os.path.abspath(args.output_dir)))

if __name__ == '__main__':
    main()