# Mock Project for Bayesian Optimization

## Description

Bayesian optimization (BO) with Gaussian processes (GPs) is a powerful method 
    for finding the global optimum (either maximum or minimum) of a 
    multidimensional function using the least number of function evaluations.
The algorithm suggests points to evaluate at, successively, whilst updating its inner
    representation of the hidden function.
Values belonging to these points are noisy, and the algorithm handles this noise
    effectively, through the incorporation of uncertainty to the inner 
    GP representation.

## Implementation

This code is an implementation of the BO in Python 3 using NumPy and SciPy.

## API

First, one should define the underlying GP model;

```python
from pybopt import GaussianProcess

gp = GaussianProcess(
    length_scale=...,  # the characteristic length inside the Gaussian kernel
    n_of_features=...,  # the number of hidden function input dimensions
    kernel_scaler=...  # scales the kernels in the covariance matrix
)
```

The `length_scale` parameter can be a single float or a numpy array of floats,
    if every feature has a different length scale.
Next, we can define the BO object;

```python
from pybopt import BayesianOptimizer

bo = BayesianOptimizer(
    gaussian_process=gp,
    bounds_min=...,  # lower bounds for every feature
    bounds_max=...,  # upper bounds for every feature
    solver_n_points=...,  # number of guess points for GP optimum search
    solver_bfgs_n_iter=...  # number of BFGS iterations for GP optimum search
)
```

Arguments starting with `solver_` refer to options setting the behavior of the maximum
    search of the GP.
The following methods are available for the GP and BO objects;

- `gp.calculate_kernel` calculates the kernel for two point collections,
- `gp.add_point` adds new observations to the gaussian process,
- `gp.evaluate_at` evaluates the gaussian process at certain test point values,
- `gp.calculate_flatness` calculates the flatness of the target function,
- `bo.upper_confidence_bound` calculates the upper confidence bound (UCB) 
    utility function for a set of test points,
- `bo.probability_of_improvement` calculates the probability of improvement (POI) 
    utility function for a set of test points,
- `bo.expected_improvement` calculates the expected improvement (EI) 
    utility function for a set of test points,
- `bo.suggest` makes a suggestion for the next point to evaluate at.

## Showcases

This implementation was showcased using the following scripts:

### test_1d.py

This script shows a simple 1D BO optimization procedure.
The function to be optimized is `(1.45 * x ** 2 - x) * math.sin(10 * x)` on the
    [0, 1] interval, which has a maximum of y = 0.1578 at x = 0.8494.
A decoy maximum can be found at x = 0.4516 with a y value of 0.1529.
The function is evaluated noisily; a random number is drawn uniformly from the
    interval [0.05, 0.3], which will serve as a standard deviation.
Then, a random number is drawn from a normal distribution with a mean of the
    function evaluation as the current test point and a standard deviation of
    the previously generated value.
This new random number will be the observed, noisy value.
The BO algorithm uses the upper confidence bound utility function with a lambda
    value of 3.
It runs for 100 iterations.

### test_2d.py

This is a more complicated function to be optimized in two dimensions.
The function is a product of two subfunctions, each dependent only on
    one of two the input variables.
The maximum is searched for on the interval [-6, 6] x [-6, 6].
In this box, among the relevant extrema, there are four larger and 
    two smaller maxima, and also four deeper and two shallower minima.
The script shows the exploration process by showing the real target
    function, the underlying mean and variance of the GP, and the utility
    function values (again, using upper confidence bound).