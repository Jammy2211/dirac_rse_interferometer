import numpy as np
from os import path

from autoarray import numba_util

import autolens as al


@numba_util.jit()
def w_tilde_data_interferometer_from(
    visibilities_real: np.ndarray,
    noise_map_real: np.ndarray,
    uv_wavelengths: np.ndarray,
    grid_radians_slim: np.ndarray,
    native_index_for_slim_index,
) -> np.ndarray:
    """
    The matrix w_tilde is a matrix of dimensions [image_pixels, image_pixels] that encodes the PSF convolution of
    every pair of image pixels given the noise map. This can be used to efficiently compute the curvature matrix via
    the mappings between image and source pixels, in a way that omits having to perform the PSF convolution on every
    individual source pixel. This provides a significant speed up for inversions of imaging datasets.

    When w_tilde is used to perform an inversion, the mapping matrices are not computed, meaning that they cannot be
    used to compute the data vector. This method creates the vector `w_tilde_data` which allows for the data
    vector to be computed efficiently without the mapping matrix.

    The matrix w_tilde_data is dimensions [image_pixels] and encodes the PSF convolution with the `weight_map`,
    where the weights are the image-pixel values divided by the noise-map values squared:

    weight = image / noise**2.0

    Parameters
    ----------
    image_native
        The two dimensional masked image of values which `w_tilde_data` is computed from.
    noise_map_native
        The two dimensional masked noise-map of values which `w_tilde_data` is computed from.
    kernel_native
        The two dimensional PSF kernel that `w_tilde_data` encodes the convolution of.
    native_index_for_slim_index
        An array of shape [total_x_pixels*sub_size] that maps pixels from the slimmed array to the native array.

    Returns
    -------
    ndarray
        A matrix that encodes the PSF convolution values between the imaging divided by the noise map**2 that enables
        efficient calculation of the data vector.
    """

    image_pixels = len(native_index_for_slim_index)

    w_tilde_data = np.zeros(image_pixels)

    weight_map_real = visibilities_real / noise_map_real**2.0

    for ip0 in range(image_pixels):
        value = 0.0

        y = grid_radians_slim[ip0, 1]
        x = grid_radians_slim[ip0, 0]

        for vis_1d_index in range(uv_wavelengths.shape[0]):
            value += weight_map_real[vis_1d_index] ** -2.0 * np.cos(
                2.0
                * np.pi
                * (
                    y * uv_wavelengths[vis_1d_index, 0]
                    + x * uv_wavelengths[vis_1d_index, 1]
                )
            )

        w_tilde_data[ip0] = value

    return w_tilde_data



"""
__Example__

We now load an interferometer dataset and input the quantities which use the funciton above to make
the w tilde matrix.

__Mask__

We define the ‘real_space_mask’ which defines the grid the image the strong lens is evaluated using.

Basiclaly, the lens model is evaluated in real space and then mapped to Fourier Space via the NUFFT. This
matrix therefore defines the dimensions of certain matrices which enter our likelihood function and calculation.
"""
real_space_mask = al.Mask2D.circular(
    shape_native=(100, 100),
    pixel_scales=0.2,
    radius=3.0,
)


"""
__Interferometer Dataset__

We load an example interferometer dataset which will be used to help us develop the likelihood function.
"""
dataset_type = "sma"

dataset_path = path.join("dataset", dataset_type)

dataset = al.Interferometer.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerDFT,
)

"""
__W Tilde__

The code below calls the w_tilde function above to create the w_tilde matrix, which we need to speed up via JAX.

Note that we have to put `np.array()` in front of all objects.

This is because of what we discussed in the telecom, where JAX arrays are typed (e.g. `Array2D`) and need special
behaviour to ensure JAX worked on the JAX'ed array. This is what `np.array()` achieves.

The source code now requires these `np.array()`'s to be in place even though we're currenlty using numba.
"""
w_tilde = w_tilde_data_interferometer_from(
    visibilities_real=np.array(dataset.data.real),
    noise_map_real=np.array(dataset.noise_map.real),
    uv_wavelengths=np.array(dataset.uv_wavelengths),
    grid_radians_slim=np.array(dataset.grid.in_radians),
    native_index_for_slim_index=np.array(real_space_mask.derive_indexes.native_for_slim).astype("int")
)

print(w_tilde)



