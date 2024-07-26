### Description
This package implements the complex valued probablistic inversion introduced by Hase et al. 2024 (https://doi.org/10.1093/gji/ggae045). The inversion is performed by using a Gauss Newton scheme that was adapted to the optimization of the real valued cost function in complex variables. The framework allows for accurate consideration of complex data errors and for the application of individual regularization to the real and imaginary part of the inferred model. For more information on the theoretical background, we refer the user to the paper, where the concepts are introduced in detail.

### Install and Import
* Install the package by running
  ```pip install .```
from the main directory of the repository.

* Import the package via:
  ```import complex_inversion_tools as cit```

* All functionality is implemented in the main Class:
  ```Complex_Inversion_Mangager```

### Basic usage
While having been developed for complex resistivity imaging, the implementation provided in this package is genral purpose and can theoretically be used for other inverse problems aswell. The inversion uses the complex coordinate representation of complex vectors. For the complex vector $\tilde{\mathbf{x}} \in \mathbb{C}^M$ we define $(\tilde{\mathbf{x}}, \bar{\tilde{\mathbf{x}}})^T \in \mathbb{C}^{2M}$ to be its representation in conjugate coordinates, with $\bar{\tilde{x}}$ denoting the complex conjugation and $^T$ denoting the transpose. Input vectors (for example for model and data) have to be formulated accordingly. The basic workflow is as follows:

1. Set up your data vector in Conjugate Coordinates (CC)
2. Set up your initial model vector in CC
3. Set up the data covariance matrix according to Equations (5), (6) and (7) in the paper. Here you can include different variances for real and imaginary data parts. Also covariances are possible. Calculate the inverse of the data covariance matrix to get the data weighting matrix that has to be provided to the inversion.
4. Set up the regularization operator according to Equation (12) in the paper.
5. Define a function ```solve_forward_problem``` that takes a model in CC and returns the forward response in CC, aswell as the complex Jacobian matrix (by default not in CC).
6. Initialize the ```Complex_Inversion_Manager``` object and provide all the above.
7. If you want, overwrite the attributes  ```Complex_Inversion_Manager.invVr``` and ```Complex_Inversion_Manager.invVi``` with the inverse variances (1-D Vector) of the real and imaginary data parts. This allows for correct calculation and printing of the individual RMSE values.
8. Run the inversion by calling ```Complex_Inversion_Manager.inversion()```
9. Get some ice cream and enjoy the result of your hard work ;)

### Examplex
One of the supplied examples demonstrates the application to complex resistivity imaging. We use pyGIMLi for the complex forward calculation, creating the sensitivities and setting up the regularization operator. You can download pyGIMLi at: https://pygimli.org

### Further information:

Check out the complete paper:
* Hase, Joost, Maximilian Weigand, and Andreas Kemna. "A probabilistic solution to geophysical inverse problems in complex variables and its application to complex resistivity imaging." Geophysical Journal International 237.1 (2024): 456-464. https://doi.org/10.1093/gji/ggae045

And other literature if you want to go into more depth:
* Kemna, Andreas. "Tomographic inversion of complex resistivity." Ruhr-Universität Bochum 169 (2000).
* Orozco, Adrián Flores, Andreas Kemna, and Egon Zimmermann. "Data error quantification in spectral induced polarization imaging." Geophysics 77.3 (2012): E227-E237. https://doi.org/10.1190/geo2010-0194.1
* Picinbono, Bernard. "Second-order complex random vectors and normal distributions." IEEE Transactions on Signal Processing 44.10 (1996): 2637-2640.
* Sorber, Laurent, Marc Van Barel, and Lieven De Lathauwer. "Unconstrained optimization of real functions in complex variables." SIAM Journal on Optimization 22.3 (2012): 879-898. https://doi.org/10.1137/110832124
