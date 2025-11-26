from typing import Tuple, Dict, Any
import numpy as np

from scipy.optimize import minimize, OptimizeResult
from scipy.stats.distributions import norm

from .gaussian_process import GaussianProcess


class BayesianOptimizer:

    def __init__(
            self,
            gaussian_process: GaussianProcess,
            bounds_min: np.ndarray,
            bounds_max: np.ndarray,
            solver_n_points: int = 1000,
            solver_bfgs_n_iter: int = 10
    ):

        self.gaussian_process = gaussian_process
        self.bounds_min = bounds_min
        self.bounds_max = bounds_max
        self.solver_n_points = solver_n_points
        self.solver_bfgs_n_iter = solver_bfgs_n_iter

    def upper_confidence_bound(self, test_points: np.ndarray, ucb_lambda: float) -> np.ndarray:
        """
        Calculates the upper confidence bound (UCB) for a set of test points,
        i.e. mean + lambda * standard deviation.

        :param test_points: The locations at which the UCB is to be calculated.
        :param ucb_lambda: The coefficient of the standard deviation.
        :return: The UCB values.
        """
        f_mean, f_var = self.gaussian_process.evaluate_at(test_points)
        return f_mean + ucb_lambda * np.sqrt(f_var)

    def probability_of_improvement(self, test_points: np.ndarray, chi: float) -> np.ndarray:
        """
        Calculates the probability of improvement (POI) for a set of test points,
        i.e. Phi((mean - best observed - chi) / standard deviation), where Phi is the
        cumulative density function of the standard normal distribution.

        :param test_points: The locations at which the POI is to be calculated.
        :param chi: Fine-tunes the exploration-exploitation tradeoff.
            Larger values correspond to more exploration.
        :return: The POI values.
        """
        f_mean, f_var = self.gaussian_process.evaluate_at(test_points)
        best_observed = np.max(self.gaussian_process.values)
        z_score = (f_mean - best_observed - chi) / np.sqrt(f_var)
        poi = norm.cdf(z_score)

        return poi

    def expected_improvement(self, test_points: np.ndarray, chi: float) -> np.ndarray:
        """
        Calculates the expected improvement (EI) for a set of test points,
        i.e. standard deviation * [ z * Phi(z) + phi(z) ],
        where Phi is the cumulative density function of the standard normal distribution,
        phi is the probability density function ~,
        and z is (mean - best observed - chi) / standard deviation.

        :param test_points: The locations at which the EI is to be calculated.
        :param chi: Fine-tunes the exploration-exploitation tradeoff.
            Larger values correspond to more exploration.
        :return: The EI values.
        """
        f_mean, f_var = self.gaussian_process.evaluate_at(test_points)
        best_observed = np.max(self.gaussian_process.values)
        z_score = (f_mean - best_observed - chi) / np.sqrt(f_var)

        ei = np.sqrt(f_var) * (z_score * norm.cdf(z_score) + norm.pdf(z_score))

        return ei

    def suggest(self, utility_type: str, utility_kwargs: Dict[str, Any]) -> Tuple[np.ndarray, float]:
        """
        Makes a suggestion for the next point to evaluate at using the
        specified utility function and utility function arguments.
        It is based on the maximization of the inner gaussian process regressor
        using Monte Carlo evaluations and a subsequent, gradient-based algorithm (L-BFGS-B).

        :param utility_type: The name of the utility function to be used.
            Available options are `"upper_confidence_bound"`, `"probability_of_improvement"` and
            `"expected_improvement"`.
        :param utility_kwargs: The keyword arguments of the selected utility function as a dictionary.
        :return: The position and the value of the found local optimum.
        """

        def calculate_utility(points: np.ndarray) -> np.ndarray:

            if utility_type == "upper_confidence_bound":
                return self.upper_confidence_bound(points, **utility_kwargs)
            elif utility_type == "probability_of_improvement":
                return self.probability_of_improvement(points, **utility_kwargs)
            elif utility_type == "expected_improvement":
                return self.expected_improvement(points, **utility_kwargs)
            else:
                raise Exception(f"Unknown utility function type: \"{utility_type}\"!")

        if len(self.gaussian_process.points) == 0:
            out = np.random.uniform(0, 1, size=len(self.bounds_min))
            out = self.bounds_min + (self.bounds_max - self.bounds_min) * out
            return out, 0.

        test_points = np.random.uniform(0, 1, size=(self.solver_n_points, len(self.bounds_min)))
        test_points = self.bounds_min[None, :] + (self.bounds_max - self.bounds_min)[None, :] * test_points

        utility = calculate_utility(test_points)
        best_point = test_points[np.argmax(utility)]

        opt_result: OptimizeResult = minimize(
            fun=lambda x: -1. * calculate_utility(x[None, :])[0],
            x0=best_point,
            method="L-BFGS-B",
            bounds=[(x, y) for x, y in zip(self.bounds_min, self.bounds_max)],
            options=dict(maxiter=self.solver_bfgs_n_iter)
        )

        return opt_result.x, -1. * opt_result.fun
