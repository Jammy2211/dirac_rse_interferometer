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
__W Tilde__

The first task you completed was to convert the computation of two `w_tilde` matrix to JAX:

- `w_tilde_data_interferometer`.
- `w_tilde_curvature_interferometer`.

These matrices, alongside some other matrices which are used in the likelihood function, is found in the `w_tilde`
property of the `Interferometer` dataset and used in the likelihood function below we attempt to convert to JAX.

These arrays are computed in memory and stored before the likelihood function is called and therefore do not
curretly need converting to JAX.

https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/dataset/interferometer/w_tilde.py

See the function `w_tilde` here:

https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/dataset/interferometer/dataset.py
"""
print(dataset.w_tilde)

"""
__Mapping__

The code below does not need to run in JAX, and is used to create a `mapping_matrix` which is the starting
point for the interferometer code we currently seek to JAX-ify.

The example script notebooks/advanced/log_likelihood_function/pixelization/log_likelihood_function.ipynb explains
what a `mapping_matrix` is, which is identical for both imaging and interferomeer datasets.
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

inversion = tracer_to_inversion.inversion

"""
__JAX Future Functions__

These quantities below will need to be JAX-ified for the final likelihood funtion, but JAX-ing them could be
more complicated and so for now we will compute them outside of JAX using the functioning source code and come back
to them later.
"""
mapping_matrix = inversion.mapping_matrix
mapper = inversion.cls_list_from(cls=al.AbstractMapper)[0]

print(mapping_matrix)

"""
__JAX Function 1__

The first function we need to JAX-ify is the `data_vector` here:

https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/inversion/interferometer/w_tilde.py

It is the `data_vector` function.
"""
dirty_image = dataset.w_tilde.dirty_image

data_vector = likelihood_function_funcs.data_vector_from(mapping_matrix=mapping_matrix, dirty_image=dirty_image)

"""
__JAX Function 2__

We next need to JAX-ify the `curvature_matrix` function here:

https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/inversion/interferometer/w_tilde.py

In the source code this function has multiple ways to be computed, but we only need the one called
`inversion_util.curvature_matrix_via_w_tilde_from` which is found here:

https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/inversion/inversion_util.py

This uses the `w_tilde` matrix your previous task converted to JAX.
"""
w_tilde = dataset.w_tilde.w_matrix

curvature_matrix = likelihood_function_funcs.curvature_matrix_via_w_tilde_from(
    w_tilde=w_tilde, mapping_matrix=mapping_matrix
)

r"""
__JAX Function 3__

A regularization matrix is next applied to the `curvature_matrix`, the calculation of which requires JAX-ifying.

For now, lets do a simple regularization matrix which is constant.

Regularization adds a linear regularization term $G_{\rm L}$ to the $\chi^2$ we solve for giving us a new merit 
function $G$ (equation 11 WD03):

 $G = \chi^2 + \lambda \, G_{\rm L}$
 
where $\lambda$ is the `regularization_coefficient` which describes the magnitude of smoothness that is applied. A 
higher $\lambda$ will regularize the source more, leading to a smoother source reconstruction.
 
Different forms for $G_{\rm L}$ can be defined which regularize the source reconstruction in different ways. The 
`Constant` regularization scheme used in this example applies gradient regularization (equation 14 WD03):

 $G_{\rm L} = \sum_{\rm  i}^{I} \sum_{\rm  n=1}^{N}  [s_{i} - s_{i, v}]$

This regularization scheme is easier to express in words -- the summation goes to each source pixel,
determines all source pixels with which it shares a direct vertex (e.g. its neighbors) and penalizes solutions 
where the difference in reconstructed flux of these two neighboring source pixels is large.

The summation does this for all pixels, thus it favours solutions where neighboring source
pixels reconstruct similar values to one another (e.g. it favours a smooth source reconstruction).

We now define the `regularization matrix`, $H$, which allows us to include this smoothing when we solve for $s$. $H$
has dimensions `(total_source_pixels, total_source_pixels)`.

This relates to $G_{\rm L}$ as (equation 13 WD03):

 $H_{ik} = \frac{1}{2} \frac{\partial G_{\rm L}}{\partial s_{i} \partial s_{k}}$

$H$ has the `regularization_coefficient` $\lambda$ folded into it such $\lambda$'s control on the degree of smoothing
is accounted for.

This function is found here:

https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/regularization/regularization_util.py
"""
regularization_matrix = likelihood_function_funcs.constant_regularization_matrix_from(
    coefficient=source_galaxy.pixelization.regularization.coefficient,
    neighbors=mapper.source_plane_mesh_grid.neighbors, # Will need JAX-ifying in future.
    neighbors_sizes=mapper.source_plane_mesh_grid.neighbors.sizes,
)


"""
__JAX Function 4__

$H$ enters the linear algebra system we solve for as follows (WD03 equation (12)):

 $s = [F + H]^{-1} D$
"""
curvature_reg_matrix = np.add(curvature_matrix, regularization_matrix)

"""
__JAX Function 5__

We can now solve the linear system above using NumPy linear algebra. 

Note that the for loop used above to prevent a LinAlgException is no longer required.

My understanding is that `np.solve` is fully supported and implmented in JAX so this should be an easy conversion.
"""
reconstruction = likelihood_function_funcs.reconstruction_positive_negative_from(
    curvature_reg_matrix=curvature_reg_matrix,
    data_vector=data_vector
)

"""
__JAX Function 6__

We now quantify the goodness-of-fit of our lens model and source reconstruction. 

We compute the `chi_squared` of the fit.

With the `w_tilde` matrix formalism we can use a trick to compute the chi sqaured in a fast way, which bypass
mapping the `reconstruction` to fourier space via a fast Fourier transform.

The second task brings in a number of other functions that in the source code calculation are used before
and after the w_tilde matrix multiplication above.

These all need to be JAX-ified and profiled to understand how they scale with JAX.

They look simple to JAX-ify -- they just use `np.multiply` and `np.dot` which are natively supported by JAX.
"""
# NOTE:
chi_squared_term_1 = np.linalg.multi_dot(
    [
        mapping_matrix,  # NOTE: shape = (N, )
        w_tilde,  # NOTE: shape = (N, N)
        mapping_matrix,
    ]
)

# NOTE:
chi_squared_term_2 = -np.multiply(2.0, np.dot(mapping_matrix, dirty_image)) # Need to double check dirty_image is the right input.

chi_squared = chi_squared_term_1 + chi_squared_term_2

r"""
__JAX Function 7__

The second term, $s^{T} H s$, corresponds to the $\lambda $G_{\rm L}$ regularization term we added to our merit 
function above.

This is the term which sums up the difference in flux of all reconstructed source pixels, and reduces the likelihood of 
solutions where there are large differences in flux (e.g. the source is less smooth and more likely to be 
overfitting noise).

We compute it below via matrix multiplication, noting that the `regularization_coefficient`, $\lambda$, is built into 
the `regularization_matrix` already.
"""
regularization_term = np.matmul(
    reconstruction.T, np.matmul(regularization_matrix, reconstruction)
)

print(regularization_term)

r"""
__JAX Function 8__

Up to this point, it is unclear why we chose a value of `regularization_coefficient=1.0`. 

We cannot rely on the `chi_squared` and `regularization_term` above to optimally choose its value, because increasing 
the `regularization_coefficient` smooths the solution more and therefore:
 
 - Decreases `chi_squared` by fitting the data worse, producing a lower `log_likelihood`.
 
 - Increases the `regularization_term` by penalizing the differences between source pixel fluxes more, again reducing
 the inferred `log_likelihood`.

If we set the regularization coefficient based purely on these two terms, we would set a value of 0.0 and be back where
we started over-fitting noise!

The terms $\left[ \mathrm{det} (F + H) \right]$ and $ - { \mathrm{ln}} \, \left[ \mathrm{det} (H) \right]$ address 
this problem. 

They quantify how complex the source reconstruction is, and penalize solutions where *it is more complex*. Reducing 
the `regularization_coefficient` makes the source reconstruction more complex (because a source that is 
smoothed less uses more flexibility to fit the data better).

These two terms therefore counteract the `chi_squared` and `regularization_term`, so as to attribute a higher
`log_likelihood` to solutions which fit the data with a more smoothed and less complex source (e.g. one with a higher 
`regularization_coefficient`).

In **HowToLens** -> `chapter 4` -> `tutorial_4_bayesian_regularization` we expand on this further and give a more
detailed description of how these different terms impact the `log_likelihood_function`. 
"""
log_curvature_reg_matrix_term = np.linalg.slogdet(curvature_reg_matrix)[1]
log_regularization_matrix_term = np.linalg.slogdet(regularization_matrix)[1]

print(log_curvature_reg_matrix_term)
print(log_regularization_matrix_term)

"""
__JAX Function 9__

Our likelihood function assumes the imaging data consists of independent Gaussian noise in every image pixel.

The final term ins the likelihood function is therefore a `noise_normalization` term, which consists of the sum
of the log of every noise-map value squared. 

Given the `noise_map` is fixed, this term does not change during the lens modeling process and has no impact on the 
model we infer.

Returns the noise-map normalization term of the noise-map, summing the noise_map value in every pixel as:

[Noise_Term] = sum(log(2*pi*[Noise]**2.0))
"""
noise_normalization = likelihood_function_funcs.noise_normalization_complex_from(
    noise_map=dataset.noise_map,
)

"""
__Jax Function 10__

We can now, finally, compute the `log_likelihood` of the lens model, by combining the five terms computed above using
the likelihood function defined above.
"""
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
