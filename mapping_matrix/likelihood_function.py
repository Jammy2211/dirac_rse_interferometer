import numpy as np
from os import path

from . import likelihood_function_funcs

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
We need to convert as many calculations performed in `pix_sub_weights` to JAX as possible, with the following
things to note:

- The beginning of this function uses the `scipy.spatial` `Delaunay` object which could be extremely difficult to
  convert to JAX. If you cannot find a JAX implementation, it is fine to split the function in half and focus on
  JAX-ifying the rest of the calculation. JAXBind could allow us to implement this function as a stop gap until
  we find a JAX implementation of Delaunay: https://arxiv.org/abs/2403.08847
  
- The functions `pix_indexes_for_sub_slim_index_delaunay_from` and `pixel_weights_delaunay_from` should be more easy
  to JAX-ify, assuming you have the input arrays from the `Delaunay` object.
"""
print(mapper.pix_sub_weights)

"""
__Mapping Matrix__

The calculaiton of the `mapping_matrix` needs to be JAX-ified, whcih is performed in `mapper_util`:

https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/pixelization/mappers/mapper_util.py

It is called `mapping_matrix_from` and I call it below:
"""
mapping_matrix = al.util.mapper.mapping_matrix_from(
    pix_indexes_for_sub_slim_index=mapper.pix_indexes_for_sub_slim_index,
    pix_size_for_sub_slim_index=mapper.pix_sizes_for_sub_slim_index,  # unused for Voronoi
    pix_weights_for_sub_slim_index=mapper.pix_weights_for_sub_slim_index,  # unused for Voronoi
    pixels=mapper.pixels,
    total_mask_pixels=mapper.source_plane_data_grid.mask.pixels_in_mask,
    slim_index_for_sub_slim_index=mapper.slim_index_for_sub_slim_index,
    sub_fraction=np.array(mapper.over_sampler.sub_fraction),
)

"""
__Border Relocator__

The example:

https://github.com/Jammy2211/autolens_workspace/blob/release/notebooks/advanced/log_likelihood_function/pixelization/log_likelihood_function.ipynb

Has a step called "Likelihood Step 5: Border Relocation"

This is performed here:

https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/pixelization/mesh/triangulation.py

In the function `mapper_grids_from`.

This means the functions in `BorderRelocator` need to be converted to JAX:

https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/pixelization/border_relocator.py
"""