import numpy as np
from os import path
from typing import Tuple

from autoarray import numba_util
import autolens as al

"""
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
__Mapping Matrix__

The following task describes what the `mapping_matrix` is:

https://github.com/Jammy2211/autolens_workspace/blob/release/notebooks/advanced/log_likelihood_function/pixelization/log_likelihood_function.ipynb

This tasks describes it using a `Rectangular` pixelizaiton, but we ideally will have a JAX calculaiton which instead
uses a `Delaunay` mesh, which performs better for analysis.

We first set up a `TracerToInversion` which has the quantities we need to compute the `mapping_matrix`
"""
mass = al.mp.Isothermal(
    centre=(0.0, 0.0),
    einstein_radius=1.6,
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
)

lens_galaxy = al.Galaxy(redshift=0.5, mass=mass)

pixelization = al.Pixelization(
    image_mesh=al.image_mesh.Overlay(shape=(30, 30)),
    mesh=al.mesh.Delaunay(),
    regularization=al.reg.Constant(coefficient=1.0),
)

source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

tracer_to_inversion = al.TracerToInversion(tracer=tracer, dataset=dataset)

"""
The module we need to try and convert to JAX is the following, which is the Delaunay computation:

https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/pixelization/mappers/delaunay.py

First we map a `Mapper`, which is a `MapperDelaunay` object.
"""
inversion = tracer_to_inversion.inversion

mapper = inversion.cls_list_from(cls=al.AbstractMapper)[0]

"""
The `MapperDelaunay` object has a property `pix_sub_weights`:
"""
print(mapper.pix_sub_weights)

"""
This has three key quantities, which all `Mappers` have (e.g. `MapperRectangular`, `MapperVoronoi` etc.):

The first is the `mappings`, which are the mappings of every data pixel to its corresponding pixelization mesh pixels.

Lets look at the first enter `mappings` in the Delaunay `Mapper`, which is telling us what Delaunay triangles the
first image pixel in our mask map too.

This prints 3 integers, which are the indexes of the Delaunay triangles the first image pixel maps too. For a Delaunay
mesh, there are always 3 mappings per data pixel.
"""
print(mapper.pix_sub_weights.mappings[0])

"""
There is also a `sizes` property, which gives the number of mappings of every data pixel to pixelization mesh pixels.

For the Delaunay `Mapper`, this is always 3 mappings per data pixel, but for other meshes, (e.g Voronoi) this can vary
per mesh pixel and there is a quantity we compute.
"""
print(mapper.pix_sub_weights.sizes[0])

"""
The `Delaunay` mappings are computed by the following function:
"""
@numba_util.jit()
def pix_indexes_for_sub_slim_index_delaunay_from(
    source_plane_data_grid,
    simplex_index_for_sub_slim_index,
    pix_indexes_for_simplex_index,
    delaunay_points,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    The indexes mappings between the sub pixels and Voronoi mesh pixels.
    For Delaunay tessellation, most sub pixels should have contribution of 3 pixelization pixels. However,
    for those ones not belonging to any triangle, we link its value to its closest point.

    The returning result is a matrix of (len(sub_pixels, 3)) where the entries mark the relevant source pixel indexes.
    A row like [A, -1, -1] means that sub pixel only links to source pixel A.
    """

    pix_indexes_for_sub_slim_index = -1 * np.ones(
        shape=(source_plane_data_grid.shape[0], 3)
    )

    for i in range(len(source_plane_data_grid)):
        simplex_index = simplex_index_for_sub_slim_index[i]
        if simplex_index != -1:
            pix_indexes_for_sub_slim_index[i] = pix_indexes_for_simplex_index[
                simplex_index_for_sub_slim_index[i]
            ]
        else:
            pix_indexes_for_sub_slim_index[i][0] = np.argmin(
                np.sum((delaunay_points - source_plane_data_grid[i]) ** 2.0, axis=1)
            )

    pix_indexes_for_sub_slim_index_sizes = np.sum(
        pix_indexes_for_sub_slim_index >= 0, axis=1
    )

    return pix_indexes_for_sub_slim_index, pix_indexes_for_sub_slim_index_sizes


"""
The function is called as follows, where we can immediately see why this will be hard to convert to JAX,
because it uses the `scipy.spatial` `Delaunay` object.
"""
import scipy

source_plane_data_grid = mapper.source_plane_data_grid

delaunay = scipy.spatial.Delaunay(np.asarray([source_plane_data_grid[:, 0], source_plane_data_grid[:, 1]]).T)

simplex_index_for_sub_slim_index = delaunay.find_simplex(
    source_plane_data_grid.over_sampled
)
pix_indexes_for_simplex_index = delaunay.simplices

mappings, sizes = pix_indexes_for_sub_slim_index_delaunay_from(
    source_plane_data_grid=np.array(source_plane_data_grid.over_sampled),
    simplex_index_for_sub_slim_index=simplex_index_for_sub_slim_index,
    pix_indexes_for_simplex_index=pix_indexes_for_simplex_index,
    delaunay_points=delaunay.points,
)

"""
The other property is `weights`, which are the interpolation weights of every data pixel's pixelization pixel mapping.
"""
print(mapper.pix_sub_weights.weights[0])


"""
These are computed as follows:
"""
@numba_util.jit()
def delaunay_triangle_area_from(
    corner_0: Tuple[float, float],
    corner_1: Tuple[float, float],
    corner_2: Tuple[float, float],
) -> float:
    """
    Returns the area within a Delaunay triangle where the three corners are located at the (x,y) coordinates given by
    the inputs `corner_a` `corner_b` and `corner_c`.

    This function actually returns the area of any triangle, but the term `delaunay` is included in the title to
    separate it from the `rectangular` and `voronoi` methods in `mesh_util.py`.

    Parameters
    ----------
    corner_0
        The (x,y) coordinates of the triangle's first corner.
    corner_1
        The (x,y) coordinates of the triangle's second corner.
    corner_2
        The (x,y) coordinates of the triangle's third corner.

    Returns
    -------
    The area of the triangle given the input (x,y) corners.
    """

    x1 = corner_0[0]
    y1 = corner_0[1]
    x2 = corner_1[0]
    y2 = corner_1[1]
    x3 = corner_2[0]
    y3 = corner_2[1]

    return 0.5 * np.abs(x1 * y2 + x2 * y3 + x3 * y1 - x2 * y1 - x3 * y2 - x1 * y3)


@numba_util.jit()
def pixel_weights_delaunay_from(
    source_plane_data_grid,
    source_plane_mesh_grid,
    slim_index_for_sub_slim_index: np.ndarray,
    pix_indexes_for_sub_slim_index,
) -> np.ndarray:
    """
    Returns the weights of the mappings between the masked sub-pixels and the Delaunay pixelization.

    Weights are determiend via a nearest neighbor interpolation scheme, whereby every data-sub pixel maps to three
    Delaunay pixel vertexes (in the source frame). The weights of these 3 mappings depends on the distance of the
    coordinate to each vertex, with the highest weight being its closest neighbor,

    Parameters
    ----------
    source_plane_data_grid
        A 2D grid of (y,x) coordinates associated with the unmasked 2D data after it has been transformed to the
        `source` reference frame.
    source_plane_mesh_grid
        The 2D grid of (y,x) centres of every pixelization pixel in the `source` frame.
    slim_index_for_sub_slim_index
        The mappings between the data's sub slimmed indexes and the slimmed indexes on the non sub-sized indexes.
    pix_indexes_for_sub_slim_index
        The mappings from a data sub-pixel index to a pixelization pixel index.
    """

    pixel_weights = np.zeros(pix_indexes_for_sub_slim_index.shape)

    for sub_slim_index in range(slim_index_for_sub_slim_index.shape[0]):
        pix_indexes = pix_indexes_for_sub_slim_index[sub_slim_index]

        if pix_indexes[1] != -1:
            vertices_of_the_simplex = source_plane_mesh_grid[pix_indexes]

            sub_gird_coordinate_on_source_place = source_plane_data_grid[sub_slim_index]

            area_0 = delaunay_triangle_area_from(
                corner_0=vertices_of_the_simplex[1],
                corner_1=vertices_of_the_simplex[2],
                corner_2=sub_gird_coordinate_on_source_place,
            )
            area_1 = delaunay_triangle_area_from(
                corner_0=vertices_of_the_simplex[0],
                corner_1=vertices_of_the_simplex[2],
                corner_2=sub_gird_coordinate_on_source_place,
            )
            area_2 = delaunay_triangle_area_from(
                corner_0=vertices_of_the_simplex[0],
                corner_1=vertices_of_the_simplex[1],
                corner_2=sub_gird_coordinate_on_source_place,
            )

            norm = area_0 + area_1 + area_2

            weight_abc = np.array([area_0, area_1, area_2]) / norm

            pixel_weights[sub_slim_index] = weight_abc

        else:
            pixel_weights[sub_slim_index][0] = 1.0

    return pixel_weights

weights = pixel_weights_delaunay_from(
    source_plane_data_grid=np.array(source_plane_data_grid.over_sampled),
    source_plane_mesh_grid=np.array(mapper.source_plane_mesh_grid),
    slim_index_for_sub_slim_index=mapper.slim_index_for_sub_slim_index,
    pix_indexes_for_sub_slim_index=mappings,
)

"""
__Mapping Matrix__

The calculaiton of the `mapping_matrix` needs to be JAX-ified, whcih is performed in `mapper_util`:

https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/pixelization/mappers/mapper_util.py

It is called `mapping_matrix_from` and I call it below:
"""
@numba_util.jit()
def mapping_matrix_from(
    pix_indexes_for_sub_slim_index: np.ndarray,
    pix_size_for_sub_slim_index: np.ndarray,
    pix_weights_for_sub_slim_index: np.ndarray,
    pixels: int,
    total_mask_pixels: int,
    slim_index_for_sub_slim_index: np.ndarray,
    sub_fraction: np.ndarray,
) -> np.ndarray:
    """
    Returns the mapping matrix, which is a matrix representing the mapping between every unmasked sub-pixel of the data
    and the pixels of a pixelization. Non-zero entries signify a mapping, whereas zeros signify no mapping.

    For example, if the data has 5 unmasked pixels (with `sub_size=1` so there are not sub-pixels) and the pixelization
    3 pixels, with the following mappings:

    data pixel 0 -> pixelization pixel 0
    data pixel 1 -> pixelization pixel 0
    data pixel 2 -> pixelization pixel 1
    data pixel 3 -> pixelization pixel 1
    data pixel 4 -> pixelization pixel 2

    The mapping matrix (which is of dimensions [data_pixels, pixelization_pixels]) would appear as follows:

    [1, 0, 0] [0->0]
    [1, 0, 0] [1->0]
    [0, 1, 0] [2->1]
    [0, 1, 0] [3->1]
    [0, 0, 1] [4->2]

    The mapping matrix is actually built using the sub-grid of the grid, whereby each pixel is divided into a grid of
    sub-pixels which are all paired to pixels in the pixelization. The entries in the mapping matrix now become
    fractional values dependent on the sub-pixel sizes.

    For example, for a 2x2 sub-pixels in each pixel means the fractional value is 1.0/(2.0^2) = 0.25, if we have the
    following mappings:

    data pixel 0 -> data sub pixel 0 -> pixelization pixel 0
    data pixel 0 -> data sub pixel 1 -> pixelization pixel 1
    data pixel 0 -> data sub pixel 2 -> pixelization pixel 1
    data pixel 0 -> data sub pixel 3 -> pixelization pixel 1
    data pixel 1 -> data sub pixel 0 -> pixelization pixel 1
    data pixel 1 -> data sub pixel 1 -> pixelization pixel 1
    data pixel 1 -> data sub pixel 2 -> pixelization pixel 1
    data pixel 1 -> data sub pixel 3 -> pixelization pixel 1
    data pixel 2 -> data sub pixel 0 -> pixelization pixel 2
    data pixel 2 -> data sub pixel 1 -> pixelization pixel 2
    data pixel 2 -> data sub pixel 2 -> pixelization pixel 3
    data pixel 2 -> data sub pixel 3 -> pixelization pixel 3

    The mapping matrix (which is still of dimensions [data_pixels, pixelization_pixels]) appears as follows:

    [0.25, 0.75, 0.0, 0.0] [1 sub-pixel maps to pixel 0, 3 map to pixel 1]
    [ 0.0,  1.0, 0.0, 0.0] [All sub-pixels map to pixel 1]
    [ 0.0,  0.0, 0.5, 0.5] [2 sub-pixels map to pixel 2, 2 map to pixel 3]

    For certain pixelizations each data sub-pixel maps to multiple pixelization pixels in a weighted fashion, for
    example a Delaunay pixelization where there are 3 mappings per sub-pixel whose weights are determined via a
    nearest neighbor interpolation scheme.

    In this case, each mapping value is multiplied by this interpolation weight (which are in the array
    `pix_weights_for_sub_slim_index`) when the mapping matrix is constructed.

    Parameters
    ----------
    pix_indexes_for_sub_slim_index
        The mappings from a data sub-pixel index to a pixelization pixel index.
    pix_size_for_sub_slim_index
        The number of mappings between each data sub pixel and pixelization pixel.
    pix_weights_for_sub_slim_index
        The weights of the mappings of every data sub pixel and pixelization pixel.
    pixels
        The number of pixels in the pixelization.
    total_mask_pixels
        The number of datas pixels in the observed datas and thus on the grid.
    slim_index_for_sub_slim_index
        The mappings between the data's sub slimmed indexes and the slimmed indexes on the non sub-sized indexes.
    sub_fraction
        The fractional area each sub-pixel takes up in an pixel.
    """

    mapping_matrix = np.zeros((total_mask_pixels, pixels))

    for sub_slim_index in range(slim_index_for_sub_slim_index.shape[0]):
        slim_index = slim_index_for_sub_slim_index[sub_slim_index]

        for pix_count in range(pix_size_for_sub_slim_index[sub_slim_index]):
            pix_index = pix_indexes_for_sub_slim_index[sub_slim_index, pix_count]
            pix_weight = pix_weights_for_sub_slim_index[sub_slim_index, pix_count]

            mapping_matrix[slim_index][pix_index] += (
                sub_fraction[slim_index] * pix_weight
            )

    return mapping_matrix

mapping_matrix = mapping_matrix_from(
    pix_indexes_for_sub_slim_index=mapper.pix_sub_weights.mappings,
    pix_size_for_sub_slim_index=mapper.pix_sub_weights.sizes,  # unused for Voronoi
    pix_weights_for_sub_slim_index=mapper.pix_sub_weights.weights,  # unused for Voronoi
    pixels=mapper.pixels,
    total_mask_pixels=mapper.source_plane_data_grid.mask.pixels_in_mask,
    slim_index_for_sub_slim_index=mapper.slim_index_for_sub_slim_index,
    sub_fraction=np.array(mapper.over_sampler.sub_fraction),
)

"""
__Voronoi__

The exact same task above can be performed using a `Voronoi` mesh, which uses natuural neighbor interpolation.

The module for this is here:

https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/pixelization/mappers/voronoi.py

This Voronoi mesh is "better" for lensing analysis, but may be even more difficult to JAX-ify.

It uses a C++ library to compute the Voronoi mappings and weights:

https://github.com/Jammy2211/PyAutoArray/tree/main/autoarray/util/nn
"""
