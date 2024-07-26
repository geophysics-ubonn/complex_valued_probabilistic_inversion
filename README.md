This package implements the complex valued probablistic inversion introduced by Hase et al. 2024 (https://doi.org/10.1093/gji/ggae045).
The inversion is performed by using a Gauss Newton scheme that was adapted to the optimization of the real valued cost function in complex variables.

For more information on the theoretical background, we refer the user to the paper, where the concepts are introduced in detail.

FIRST THINGS FIRST:
Install the package by running
  pip install .
from the main directory.

Import the package via:
  import complex_inversion_tools as cit

All functionality is implemented in the main Class:
  Complex_Inversion_Mangager
  
The inversion uses the complex coordinate representation of complex vectors. For the complex vector $\tilde{\mathbf{x}} \in \mathbb{C}^M$ we define $(\tilde{\mathbf{x}}, \bar{\tilde{\mathbf{x}}})^T \in \mathbb{C}^{2M}$ to be its representation in conjugate coordinates, with $\bar{\tilde{x}}$ denoting the complex conjugation and $^T$ denoting the transpose. Input vectors (for example for model and data) have to be formulated accordingly.
