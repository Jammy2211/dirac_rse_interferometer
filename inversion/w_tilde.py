"""
__Log Likelihood Function: W Tilde__

This script describes how a pixelization can be computed using a different linear algebra calculation, but
one which produces an identical likelihood at the end.

This is called the `w_tilde` formalism, and for interferometer datasets it avoids storing the `operated_mapping_matrix`
in memory, meaning that in the regime of 1e6 or more visibilities this extremely large matrix does not need to be
stored in memory.

This can make the likelihood function significantly faster, for example with speed ups of hundreds of times or more
for tens or millions of visibilities. In fact, the run time does not scale with the number of visibilities at all,
meaning datasets of any size can be fitted in seconds.

It directly follows on from the `pixelization/log_likelihood_function.py` notebook and you should read through that
script before reading this script.

__Prerequisites__

You must read through the following likelihood functions first:

 - `pixelization/log_likelihood_function.py` the likelihood function for a pixelization.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import matplotlib.pyplot as plt
import numpy as np
from os import path

import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset__

Following the `pixelization/log_likelihood_function.py` script, we load and mask an `Imaging` dataset and
set oversampling to 1.
"""
real_space_mask = ag.Mask2D.circular(shape_native=(8, 8), pixel_scales=0.05, radius=4.0)

dataset_name = "simple"
dataset_path = path.join("dataset", "interferometer", dataset_name)

dataset = ag.Interferometer.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
    transformer_class=ag.TransformerDFT,
)

"""
__W Tilde__

We now compute the `w_tilde` matrix.

The `w_tilde` matrix is applied to the `curvature_matrix`, and allows us to efficiently compute the curvature matrix
without computing the `transformed_mapping_matrix` matrix. 

The functions used to do this has been copy and pasted from the `inversion` module of PyAutoArray source code below,
so you can see the calculation in full detail.

REMINDER: for the `real_space_mask` above with shape (800, 800) the `w_tilde` matrix will TAKE A LONG
TIME TO COMPUTE.
"""
from autoarray import numba_util


@numba_util.jit()
def w_tilde_curvature_interferometer_from(
    noise_map_real: np.ndarray,
    uv_wavelengths: np.ndarray,
    grid_radians_slim: np.ndarray,
) -> np.ndarray:
    """
    The matrix w_tilde is a matrix of dimensions [image_pixels, image_pixels] that encodes the NUFFT of every pair of
    image pixels given the noise map. This can be used to efficiently compute the curvature matrix via the mappings
    between image and source pixels, in a way that omits having to perform the NUFFT on every individual source pixel.
    This provides a significant speed up for inversions of interferometer datasets with large number of visibilities.

    The limitation of this matrix is that the dimensions of [image_pixels, image_pixels] can exceed many 10s of GB's,
    making it impossible to store in memory and its use in linear algebra calculations extremely. The method
    `w_tilde_preload_interferometer_from` describes a compressed representation that overcomes this hurdles. It is
    advised `w_tilde` and this method are only used for testing.

    Parameters
    ----------
    noise_map_real
        The real noise-map values of the interferometer data.
    uv_wavelengths
        The wavelengths of the coordinates in the uv-plane for the interferometer dataset that is to be Fourier
        transformed.
    grid_radians_slim
        The 1D (y,x) grid of coordinates in radians corresponding to real-space mask within which the image that is
        Fourier transformed is computed.

    Returns
    -------
    ndarray
        A matrix that encodes the NUFFT values between the noise map that enables efficient calculation of the curvature
        matrix.
    """

    w_tilde = np.zeros((grid_radians_slim.shape[0], grid_radians_slim.shape[0]))

    for i in range(w_tilde.shape[0]):
        for j in range(i, w_tilde.shape[1]):
            y_offset = grid_radians_slim[i, 1] - grid_radians_slim[j, 1]
            x_offset = grid_radians_slim[i, 0] - grid_radians_slim[j, 0]

            for vis_1d_index in range(uv_wavelengths.shape[0]):
                w_tilde[i, j] += noise_map_real[vis_1d_index] ** -2.0 * np.cos(
                    2.0
                    * np.pi
                    * (
                        y_offset * uv_wavelengths[vis_1d_index, 0]
                        + x_offset * uv_wavelengths[vis_1d_index, 1]
                    )
                )

    for i in range(w_tilde.shape[0]):
        for j in range(i, w_tilde.shape[1]):
            w_tilde[j, i] = w_tilde[i, j]

    return w_tilde


"""
We now compute the `w_tilde` matrices.
"""
w_tilde = w_tilde_curvature_interferometer_from(
    noise_map_real=np.array(dataset.noise_map.real),
    uv_wavelengths=np.array(dataset.uv_wavelengths),
    grid_radians_slim=np.array(dataset.grid.in_radians),
)

"""
__Mapping Matrix__

The `w_tilde` matrix is applied directly to the `mapping_matrix` to compute the `curvature_matrix`.

Below, we perform the likelihood function steps described in the `pixelization/log_likelihood_function.py` script,
to create the `mapping_matrix` we will apply the `w_tilde` matrix to.
"""
pixelization = ag.Pixelization(
    mesh=ag.mesh.Rectangular(shape=(30, 30)),
    regularization=ag.reg.Constant(coefficient=1.0),
)

galaxy = ag.Galaxy(redshift=0.5, pixelization=pixelization)

grid_rectangular = ag.Mesh2DRectangular.overlay_grid(
    shape_native=galaxy.pixelization.mesh.shape, grid=dataset.grids.pixelization
)

mapper_grids = ag.MapperGrids(
    mask=real_space_mask,
    source_plane_data_grid=dataset.grids.pixelization,
    source_plane_mesh_grid=grid_rectangular,
)

mapper = ag.Mapper(
    mapper_grids=mapper_grids,
    regularization=None,
)

mapping_matrix = ag.util.mapper.mapping_matrix_from(
    pix_indexes_for_sub_slim_index=mapper.pix_indexes_for_sub_slim_index,
    pix_size_for_sub_slim_index=mapper.pix_sizes_for_sub_slim_index,  # unused for rectangular
    pix_weights_for_sub_slim_index=mapper.pix_weights_for_sub_slim_index,  # unused for rectangular
    pixels=mapper.pixels,
    total_mask_pixels=mapper.source_plane_data_grid.mask.pixels_in_mask,
    slim_index_for_sub_slim_index=mapper.slim_index_for_sub_slim_index,
    sub_fraction=np.array(mapper.over_sampler.sub_fraction),
)

"""
__Curvature Matrix__

We can now compute the `curvature_matrix` using the `w_tilde` matrix and `mapping_matrix`, which amazingly uses
simple matrix multiplication.
"""


def curvature_matrix_via_w_tilde_from(
    w_tilde: np.ndarray, mapping_matrix: np.ndarray
) -> np.ndarray:
    """
    Returns the curvature matrix `F` (see Warren & Dye 2003) from `w_tilde`.

    The dimensions of `w_tilde` are [image_pixels, image_pixels], meaning that for datasets with many image pixels
    this matrix can take up 10's of GB of memory. The calculation of the `curvature_matrix` via this function will
    therefore be very slow, and the method `curvature_matrix_via_w_tilde_curvature_preload_imaging_from` should be used
    instead.

    Parameters
    ----------
    w_tilde
        A matrix of dimensions [image_pixels, image_pixels] that encodes the convolution or NUFFT of every image pixel
        pair on the noise map.
    mapping_matrix
        The matrix representing the mappings between sub-grid pixels and pixelization pixels.

    Returns
    -------
    ndarray
        The curvature matrix `F` (see Warren & Dye 2003).
    """

    return np.dot(mapping_matrix.T, np.dot(w_tilde, mapping_matrix))


curvature_matrix = curvature_matrix_via_w_tilde_from(
    w_tilde=w_tilde, mapping_matrix=mapping_matrix
)

"""
If you compare the `curvature_matrix` computed using the `w_tilde` matrix to the `curvature_matrix` computed using the
`operated_mapping_matrix` matrix in the other example scripts, you'll see they are identical.

__Data Vector__

The `data_vector` was computed in the `pixelization/log_likelihood_function.py` script using 
the `transformed_mapping_matrix`.

Fortunately, there is also an easy way to compute the `data_vector` which bypasses the need to compute the
`transformed_mapping_matrix` matrix, again using simple matrix multiplication.
"""
data_vector = np.dot(mapping_matrix.T, dataset.w_tilde.dirty_image)

"""
__Reconstruction__

The `reconstruction` is computed using the `curvature_matrix` and `data_vector` as per usual.
"""
regularization_matrix = ag.util.regularization.constant_regularization_matrix_from(
    coefficient=galaxy.pixelization.regularization.coefficient,
    neighbors=mapper.source_plane_mesh_grid.neighbors,
    neighbors_sizes=mapper.source_plane_mesh_grid.neighbors.sizes,
)

curvature_reg_matrix = np.add(curvature_matrix, regularization_matrix)

reconstruction = np.linalg.solve(curvature_reg_matrix, data_vector)

"""
__Likelihood Step: Fast Chi Squared__

In the `pixelization/log_likelihood_function.py` example the mapped reconstructed visibilities were another quantity 
computed which used the `transformed_mapping_matrix` matrix, which is another step that must skip computing this matrix.

The w-tilde matrix again provides a trick which skips the need to compute the `transformed_mapping_matrix` matrix,
with the code for this shown below.
"""
print(mapping_matrix.shape)
print(w_tilde.shape)

chi_squared_term_1 = np.linalg.multi_dot(
    [
        reconstruction.T,  # NOTE: shape = (M, )
        curvature_matrix,  # NOTE: shape = (M, M)
        reconstruction,  # NOTE: shape = (M, )
    ]
)
chi_squared_term_2 = -2.0 * np.linalg.multi_dot(
    [reconstruction.T, data_vector]  # NOTE: shape = (M, )  # NOTE: i.e. dirty_image
)
chi_squared_term_3 = np.add(  # NOTE: i.e. noise_normalization
    np.sum(dataset.data.real**2.0 / dataset.noise_map.real**2.0),
    np.sum(dataset.data.imag**2.0 / dataset.noise_map.imag**2.0),
)

chi_squared = chi_squared_term_1 + chi_squared_term_2 + chi_squared_term_3

print(chi_squared)

"""
__Log Likelihood__

Finally, we verify that the log likelihood computed using the `curvature_matrix` and `data_vector` computed using the
`w_tilde` matrix is identical to the log likelihood computed using the `operated_mapping_matrix` matrix in the
other example scripts.
"""
regularization_term = np.matmul(
    reconstruction.T, np.matmul(regularization_matrix, reconstruction)
)

log_curvature_reg_matrix_term = np.linalg.slogdet(curvature_reg_matrix)[1]
log_regularization_matrix_term = np.linalg.slogdet(regularization_matrix)[1]


noise_normalization_real = np.sum(np.log(2 * np.pi * dataset.noise_map.real**2.0))
noise_normalization_imag = np.sum(np.log(2 * np.pi * dataset.noise_map.imag**2.0))
noise_normalization = noise_normalization_real + noise_normalization_imag

log_evidence = float(
    -0.5
    * (
        chi_squared
        + regularization_term
        + log_curvature_reg_matrix_term
        - log_regularization_matrix_term
        + noise_normalization
    )
)

print(log_evidence)

"""
__Repeated Pattern in W_Tilde__

The `w_tilde` matrix has a repeated pattern, which can be used to perform the above calculations using far less
memory, at the expense of code complexity. 

First, let us consider the pattern of the `w_tilde` matrix, which is seen in the following 7 values: 
"""
print(w_tilde[0, 1])
print(w_tilde[1, 2])
print(w_tilde[2, 3])
print(w_tilde[3, 4])
print(w_tilde[4, 5])
print(w_tilde[5, 6])
print(w_tilde[6, 7])

"""
However, the pattern breaks for the next value, which is:
"""
print(w_tilde[7, 8])

"""
What do the first 7 values have in common?

Let us think about the `real_space_mask` of the interferometer dataset, which I have made a really basic cartoon of
below:

![w_tilde](https://github.com/Jammy2211/autogalaxy_workspace/blob/main/scripts/advanced/log_likelihood_function/interferometer/pixelization/w_tilde_cartoon.png?raw=true)

What elements 0 -> 6 of the `w_tilde` matrix have in common is that they are next to one another in the real-space,
to the right, in the mask.

The element 6 -> 7 breaks this pattern, as it is at the end of the mask and there is no pixel to the right of it,
so it "jumps" to the next row.

We can now reinspect how the `w_tilde` matrix is computed, and see that the pattern of the `w_tilde` matrix is
determined by the real-space mask:
"""


@numba_util.jit()
def w_tilde_curvature_interferometer_from(
    noise_map_real: np.ndarray,
    uv_wavelengths: np.ndarray,
    grid_radians_slim: np.ndarray,
) -> np.ndarray:
    w_tilde = np.zeros((grid_radians_slim.shape[0], grid_radians_slim.shape[0]))

    for i in range(w_tilde.shape[0]):
        for j in range(i, w_tilde.shape[1]):
            """

            !!!LOOK HERE!!!!

            """

            y_offset = (
                grid_radians_slim[i, 1] - grid_radians_slim[j, 1]
            )  # The y-offset is 0 for pixels 0 -> 6, but becomes non-zero for 6 -> 7
            x_offset = (
                grid_radians_slim[i, 0] - grid_radians_slim[j, 0]
            )  # The x-offset is the same for pixels 0 -> 6 and 6 -> 7

            for vis_1d_index in range(uv_wavelengths.shape[0]):
                w_tilde[i, j] += noise_map_real[vis_1d_index] ** -2.0 * np.cos(
                    2.0
                    * np.pi
                    * (
                        y_offset * uv_wavelengths[vis_1d_index, 0]
                        + x_offset * uv_wavelengths[vis_1d_index, 1]
                    )
                )

    for i in range(w_tilde.shape[0]):
        for j in range(i, w_tilde.shape[1]):
            w_tilde[j, i] = w_tilde[i, j]

    return w_tilde


"""
The `y_offset` and `x_offset` values are what determine the repeated pattern of the `w_tilde` matrix, and therefore
mean it has far fewer unique values than the number of pixels in the real-space mask.

This could make our calculation way more efficient: as if we can exploit it we would not store [image_pixels, image_pixels]
values (where image_pixels is the number of pixels in the real-space mask and can easily reach 100,000, or 100GB+ memory),
but instead far fewer values.

This could also, maybe, speed up the matrix multiplication calculation, as we would be performing far fewer operations.

__W Tilde 1D__

The function below shows how we compute `w_tilde_curvature_preload`, which is a 2D array of dimensions
[2*shape_masked_pixels_y, 2*shape_masked_pixels_x, 2], where `shape_masked_pixels` is the (y,x) size corresponding to the
extent of unmasked pixels that go vertically and horizontally across the mask.

print(real_space_mask.shape_native_masked_pixels)

The idea behind this is we don't need to store all [image_pixels, image_pixels] values of the `w_tilde` matrix, but
instead only the unique values of the `w_tilde` matrix that are computed for each unique (y,x) offset between pairs of
pixels in the real-space mask.

Another complication is that the `y_offset` and `x_offset` values can be negative, for example if we pair a pixel
to its neighbor to the left.

That is why it has shape [2*shape_masked_pixels_y, 2*shape_masked_pixels_x, 2], with a factor of 2* in front of the
shape of the real-space mask. This is so that negative offsets can be stored in the negative half of the 2D array.

The function also has four inner four loops, which store the values of the `w_tilde` matrix for each unique (y,x) offset
between pairs of pixels in the real-space mask.
"""
from typing import Tuple


@numba_util.jit()
def w_tilde_curvature_preload_interferometer_from(
    noise_map_real: np.ndarray,
    uv_wavelengths: np.ndarray,
    shape_masked_pixels_2d: Tuple[int, int],
    grid_radians_2d: np.ndarray,
) -> np.ndarray:
    """
    The matrix w_tilde is a matrix of dimensions [unmasked_image_pixels, unmasked_image_pixels] that encodes the
    NUFFT of every pair of image pixels given the noise map. This can be used to efficiently compute the curvature
    matrix via the mapping matrix, in a way that omits having to perform the NUFFT on every individual source pixel.
    This provides a significant speed up for inversions of interferometer datasets with large number of visibilities.
    The limitation of this matrix is that the dimensions of [image_pixels, image_pixels] can exceed many 10s of GB's,
    making it impossible to store in memory and its use in linear algebra calculations extremely. This methods creates
    a preload matrix that can compute the matrix w_tilde via an efficient preloading scheme which exploits the
    symmetries in the NUFFT.
    To compute w_tilde, one first defines a real space mask where every False entry is an unmasked pixel which is
    used in the calculation, for example:
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI     This is an imaging.Mask2D, where:
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI     x = `True` (Pixel is masked and excluded from lens)
        IxIxIxIoIoIoIxIxIxIxI     o = `False` (Pixel is not masked and included in lens)
        IxIxIxIoIoIoIxIxIxIxI
        IxIxIxIoIoIoIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
    Here, there are 9 unmasked pixels. Indexing of each unmasked pixel goes from the top-left corner right and
    downwards, therefore:
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxI0I1I2IxIxIxIxI
        IxIxIxI3I4I5IxIxIxIxI
        IxIxIxI6I7I8IxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
    In the standard calculation of `w_tilde` it is a matrix of
    dimensions [unmasked_image_pixels, unmasked_pixel_images], therefore for the example mask above it would be
    dimensions [9, 9]. One performs a double for loop over `unmasked_image_pixels`, using the (y,x) spatial offset
    between every possible pair of unmasked image pixels to precompute values that depend on the properties of the NUFFT.
    This calculation has a lot of redundancy, because it uses the (y,x) *spatial offset* between the image pixels. For
    example, if two image pixel are next to one another by the same spacing the same value will be computed via the
    NUFFT. For the example mask above:

    - The value precomputed for pixel pair [0,1] is the same as pixel pairs [1,2], [3,4], [4,5], [6,7] and [7,9].

    - The value precomputed for pixel pair [0,3] is the same as pixel pairs [1,4], [2,5], [3,6], [4,7] and [5,8].

    - The values of pixels paired with themselves are also computed repeatedly for the standard calculation (e.g. 9
      times using the mask above).

    The `w_tilde_preload` method instead only computes each value once. To do this, it stores the preload values in a
    matrix of dimensions [shape_masked_pixels_y, shape_masked_pixels_x, 2], where `shape_masked_pixels` is the (y,x)
    size of the vertical and horizontal extent of unmasked pixels, e.g. the spatial extent over which the real space
    grid extends.
    Each entry in the matrix `w_tilde_preload[:,:,0]` provides the precomputed NUFFT value mapping an image pixel
    to a pixel offset by that much in the y and x directions, for example:

    - w_tilde_preload[0,0,0] gives the precomputed values of image pixels that are offset in the y direction by 0 and
      in the x direction by 0 - the values of pixels paired with themselves.

    - w_tilde_preload[1,0,0] gives the precomputed values of image pixels that are offset in the y direction by 1 and
      in the x direction by 0 - the values of pixel pairs [0,3], [1,4], [2,5], [3,6], [4,7] and [5,8]

    - w_tilde_preload[0,1,0] gives the precomputed values of image pixels that are offset in the y direction by 0 and
      in the x direction by 1 - the values of pixel pairs [0,1], [1,2], [3,4], [4,5], [6,7] and [7,9].

    Flipped pairs:

    The above preloaded values pair all image pixel NUFFT values when a pixel is to the right and / or down of the
    first image pixel. However, one must also precompute pairs where the paired pixel is to the left of the host
    pixels. These pairings are stored in `w_tilde_preload[:,:,1]`, and the ordering of these pairings is flipped in the
    x direction to make it straight forward to use this matrix when computing w_tilde.

    Parameters
    ----------
    noise_map_real
        The real noise-map values of the interferometer data
    uv_wavelengths
        The wavelengths of the coordinates in the uv-plane for the interferometer dataset that is to be Fourier
        transformed.
    shape_masked_pixels_2d
        The (y,x) shape corresponding to the extent of unmasked pixels that go vertically and horizontally across the
        mask.
    grid_radians_2d
        The 2D (y,x) grid of coordinates in radians corresponding to real-space mask within which the image that is
        Fourier transformed is computed.
    Returns
    -------
    ndarray
        A matrix that precomputes the values for fast computation of w_tilde.
    """

    y_shape = shape_masked_pixels_2d[0]
    x_shape = shape_masked_pixels_2d[1]

    curvature_preload = np.zeros((y_shape * 2, x_shape * 2))

    #  For the second preload to index backwards correctly we have to extracted the 2D grid to its shape.
    grid_radians_2d = grid_radians_2d[0:y_shape, 0:x_shape]

    grid_y_shape = grid_radians_2d.shape[0]
    grid_x_shape = grid_radians_2d.shape[1]

    for i in range(y_shape):
        for j in range(x_shape):
            y_offset = grid_radians_2d[0, 0, 0] - grid_radians_2d[i, j, 0]
            x_offset = grid_radians_2d[0, 0, 1] - grid_radians_2d[i, j, 1]

            for vis_1d_index in range(uv_wavelengths.shape[0]):
                curvature_preload[i, j] += noise_map_real[
                    vis_1d_index
                ] ** -2.0 * np.cos(
                    2.0
                    * np.pi
                    * (
                        x_offset * uv_wavelengths[vis_1d_index, 0]
                        + y_offset * uv_wavelengths[vis_1d_index, 1]
                    )
                )

    for i in range(y_shape):
        for j in range(x_shape):
            if j > 0:
                y_offset = (
                    grid_radians_2d[0, -1, 0]
                    - grid_radians_2d[i, grid_x_shape - j - 1, 0]
                )
                x_offset = (
                    grid_radians_2d[0, -1, 1]
                    - grid_radians_2d[i, grid_x_shape - j - 1, 1]
                )

                for vis_1d_index in range(uv_wavelengths.shape[0]):
                    curvature_preload[i, -j] += noise_map_real[
                        vis_1d_index
                    ] ** -2.0 * np.cos(
                        2.0
                        * np.pi
                        * (
                            x_offset * uv_wavelengths[vis_1d_index, 0]
                            + y_offset * uv_wavelengths[vis_1d_index, 1]
                        )
                    )

    for i in range(y_shape):
        for j in range(x_shape):
            if i > 0:
                y_offset = (
                    grid_radians_2d[-1, 0, 0]
                    - grid_radians_2d[grid_y_shape - i - 1, j, 0]
                )
                x_offset = (
                    grid_radians_2d[-1, 0, 1]
                    - grid_radians_2d[grid_y_shape - i - 1, j, 1]
                )

                for vis_1d_index in range(uv_wavelengths.shape[0]):
                    curvature_preload[-i, j] += noise_map_real[
                        vis_1d_index
                    ] ** -2.0 * np.cos(
                        2.0
                        * np.pi
                        * (
                            x_offset * uv_wavelengths[vis_1d_index, 0]
                            + y_offset * uv_wavelengths[vis_1d_index, 1]
                        )
                    )

    for i in range(y_shape):
        for j in range(x_shape):
            if i > 0 and j > 0:
                y_offset = (
                    grid_radians_2d[-1, -1, 0]
                    - grid_radians_2d[grid_y_shape - i - 1, grid_x_shape - j - 1, 0]
                )
                x_offset = (
                    grid_radians_2d[-1, -1, 1]
                    - grid_radians_2d[grid_y_shape - i - 1, grid_x_shape - j - 1, 1]
                )

                for vis_1d_index in range(uv_wavelengths.shape[0]):
                    curvature_preload[-i, -j] += noise_map_real[
                        vis_1d_index
                    ] ** -2.0 * np.cos(
                        2.0
                        * np.pi
                        * (
                            x_offset * uv_wavelengths[vis_1d_index, 0]
                            + y_offset * uv_wavelengths[vis_1d_index, 1]
                        )
                    )

    return curvature_preload


curvature_preload = w_tilde_curvature_preload_interferometer_from(
    noise_map_real=np.array(dataset.noise_map.real),
    uv_wavelengths=np.array(dataset.uv_wavelengths),
    shape_masked_pixels_2d=np.array(
        dataset.transformer.grid.mask.shape_native_masked_pixels
    ),
    grid_radians_2d=np.array(
        dataset.transformer.grid.mask.derive_grid.all_false.in_radians.native
    ),
)

"""
We can now use the `curvature_preload` matrix to compute the `w_tilde` matrix with its original dimensions
of [image_pixels, image_pixels] using the function below.

This is a lot faster than the original calculation, as we are only storing the unique values of the `w_tilde` matrix
and avoid repeating the same calculation for every pair of pixels in the real-space mask.
"""


@numba_util.jit()
def w_tilde_via_preload_from(w_tilde_preload, native_index_for_slim_index):
    """
    Use the preloaded w_tilde matrix (see `w_tilde_preload_interferometer_from`) to compute
    w_tilde (see `w_tilde_interferometer_from`) efficiently.

    Parameters
    ----------
    w_tilde_preload
        The preloaded values of the NUFFT that enable efficient computation of w_tilde.
    native_index_for_slim_index
        An array of shape [total_unmasked_pixels*sub_size] that maps every unmasked sub-pixel to its corresponding
        native 2D pixel using its (y,x) pixel indexes.

    Returns
    -------
    ndarray
        A matrix that encodes the NUFFT values between the noise map that enables efficient calculation of the curvature
        matrix.
    """

    slim_size = len(native_index_for_slim_index)

    w_tilde_via_preload = np.zeros((slim_size, slim_size))

    for i in range(slim_size):
        i_y, i_x = native_index_for_slim_index[i]

        for j in range(i, slim_size):
            j_y, j_x = native_index_for_slim_index[j]

            y_diff = j_y - i_y
            x_diff = j_x - i_x

            w_tilde_via_preload[i, j] = w_tilde_preload[y_diff, x_diff]

    for i in range(slim_size):
        for j in range(i, slim_size):
            w_tilde_via_preload[j, i] = w_tilde_via_preload[i, j]

    return w_tilde_via_preload


w_matrix = w_tilde_via_preload_from(
    w_tilde_preload=curvature_preload,
    native_index_for_slim_index=real_space_mask.derive_indexes.native_for_slim,
)

"""
The following function is how we compute `curvature_matrix` using the `w_tilde` matrix computed using the preload
method.
"""


@numba_util.jit()
def curvature_matrix_via_w_tilde_curvature_preload_interferometer_from(
    curvature_preload: np.ndarray,
    pix_indexes_for_sub_slim_index: np.ndarray,
    pix_size_for_sub_slim_index: np.ndarray,
    pix_weights_for_sub_slim_index: np.ndarray,
    native_index_for_slim_index: np.ndarray,
    pix_pixels: int,
) -> np.ndarray:
    """
    Returns the curvature matrix `F` (see Warren & Dye 2003) by computing it using `w_tilde_preload`
    (see `w_tilde_preload_interferometer_from`) for an interferometer inversion.

    To compute the curvature matrix via w_tilde the following matrix multiplication is normally performed:

    curvature_matrix = mapping_matrix.T * w_tilde * mapping matrix

    This function speeds this calculation up in two ways:

    1) Instead of using `w_tilde` (dimensions [image_pixels, image_pixels] it uses `w_tilde_preload` (dimensions
       [image_pixels, 2]). The massive reduction in the size of this matrix in memory allows for much fast computation.

    2) It omits the `mapping_matrix` and instead uses directly the 1D vector that maps every image pixel to a source
       pixel `native_index_for_slim_index`. This exploits the sparsity in the `mapping_matrix` to directly
       compute the `curvature_matrix` (e.g. it condenses the triple matrix multiplication into a double for loop!).

    Parameters
    ----------
    curvature_preload
        A matrix that precomputes the values for fast computation of w_tilde, which in this function is used to bypass
        the creation of w_tilde altogether and go directly to the `curvature_matrix`.
    pix_indexes_for_sub_slim_index
        The mappings from a data sub-pixel index to a pixelization's mesh pixel index.
    pix_size_for_sub_slim_index
        The number of mappings between each data sub pixel and pixelization pixel.
    pix_weights_for_sub_slim_index
        The weights of the mappings of every data sub pixel and pixelization pixel.
    native_index_for_slim_index
        An array of shape [total_unmasked_pixels*sub_size] that maps every unmasked sub-pixel to its corresponding
        native 2D pixel using its (y,x) pixel indexes.
    pix_pixels
        The total number of pixels in the pixelization's mesh that reconstructs the data.

    Returns
    -------
    ndarray
        The curvature matrix `F` (see Warren & Dye 2003).
    """

    preload = curvature_preload[0, 0]

    curvature_matrix = np.zeros((pix_pixels, pix_pixels))

    image_pixels = len(native_index_for_slim_index)

    for ip0 in range(image_pixels):
        ip0_y, ip0_x = native_index_for_slim_index[ip0]

        for ip0_pix in range(pix_size_for_sub_slim_index[ip0]):
            sp0 = pix_indexes_for_sub_slim_index[ip0, ip0_pix]

            ip0_weight = pix_weights_for_sub_slim_index[ip0, ip0_pix]

            for ip1 in range(ip0 + 1, image_pixels):
                ip1_y, ip1_x = native_index_for_slim_index[ip1]

                for ip1_pix in range(pix_size_for_sub_slim_index[ip1]):
                    sp1 = pix_indexes_for_sub_slim_index[ip1, ip1_pix]

                    ip1_weight = pix_weights_for_sub_slim_index[ip1, ip1_pix]

                    y_diff = ip1_y - ip0_y
                    x_diff = ip1_x - ip0_x

                    curvature_matrix[sp0, sp1] += (
                        curvature_preload[y_diff, x_diff] * ip0_weight * ip1_weight
                    )

    curvature_matrix_new = np.zeros((pix_pixels, pix_pixels))

    for i in range(pix_pixels):
        for j in range(pix_pixels):
            curvature_matrix_new[i, j] = curvature_matrix[i, j] + curvature_matrix[j, i]

    curvature_matrix = curvature_matrix_new

    for ip0 in range(image_pixels):
        for ip0_pix in range(pix_size_for_sub_slim_index[ip0]):
            for ip1_pix in range(pix_size_for_sub_slim_index[ip0]):
                sp0 = pix_indexes_for_sub_slim_index[ip0, ip0_pix]
                sp1 = pix_indexes_for_sub_slim_index[ip0, ip1_pix]

                ip0_weight = pix_weights_for_sub_slim_index[ip0, ip0_pix]
                ip1_weight = pix_weights_for_sub_slim_index[ip0, ip1_pix]

                if sp0 > sp1:
                    curvature_matrix[sp0, sp1] += preload * ip0_weight * ip1_weight

                    curvature_matrix[sp1, sp0] += preload * ip0_weight * ip1_weight

                elif sp0 == sp1:
                    curvature_matrix[sp0, sp1] += preload * ip0_weight * ip1_weight

    return curvature_matrix


curvature_matrix_fast = curvature_matrix_via_w_tilde_curvature_preload_interferometer_from(
    curvature_preload=dataset.w_tilde.curvature_preload,
    pix_indexes_for_sub_slim_index=mapper.pix_sub_weights.mappings,
    pix_size_for_sub_slim_index=mapper.pix_sub_weights.sizes,
    pix_weights_for_sub_slim_index=mapper.pix_sub_weights.weights,
    native_index_for_slim_index=dataset.transformer.real_space_mask.derive_indexes.native_for_slim,
    pix_pixels=mapper.pixels,
)


print(curvature_matrix_fast - curvature_matrix)

"""
Finish.
"""
