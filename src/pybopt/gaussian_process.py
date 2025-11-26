from typing import Union, Tuple

import numpy as np

class GaussianProcess:

    def __init__(
            self,
            length_scale: Union[float, np.ndarray],
            n_of_features: int,
            kernel_scaler: float = 1.0,
    ) -> None:
        """
        Initializes a gaussian process.

        :param length_scale: The scaler for the features.
             Either a single float or a vector of scaler values for every feature.
        :param n_of_features: The number of features.
        :param kernel_scaler: The coefficient of the gaussian kernels.
             It is used to scale the correlations with respect to the diagonal variances.
        """

        if type(length_scale) is float:
            length_scale = length_scale * np.ones((1, 1, n_of_features))  # (1, 1, f)
        elif len(length_scale.shape) == 1:
            length_scale = np.copy(length_scale)[None, None, :]  # (1, 1, f)
        else:
            raise TypeError("Invalid shape for the parameter length_scale!")

        self.length_scale = length_scale
        self.kernel_scaler = kernel_scaler

        self.points = np.zeros((0, n_of_features))
        self.values = np.zeros((0, ))
        self.variances = np.zeros((0, ))

        self._k_mx = np.zeros((0, 0, ))
        self._k_inv_mx = np.zeros((0, 0, ))

    def calculate_kernel(self, points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
        """
        Calculates the kernel for two point collections.

        :param points1: The first collection of points with a shape of `(Np1, f)`.
        :param points2: The second collection of points with a shape of `(Np2, f)`.
        :return: The correlation matrix with a shape of `(Np1, Np2)`.
        """

        dmx_squared = points1[:, None, :] - points2[None, :, :]
        dmx_squared = (dmx_squared / self.length_scale) ** 2 / 2
        dmx_squared = np.sum(dmx_squared, axis=2)

        correlation_mx = self.kernel_scaler * np.exp(- dmx_squared)

        return correlation_mx  # (Np1, Np2)

    def add_point(self, new_points: np.ndarray, values: np.ndarray, variances: np.ndarray) -> None:
        """
        Adds new observations to the gaussian process.

        :param new_points: The locations of the observations. Should have a shape of `(Np_new, f)`,
            where Np_new is the number of points, while f is the number of features.
            Note that if `self.length_scale` is a vector instead of a single float, it should have a shape
            of `(f, )`.
        :param values: The evaluated hidden function at each `point` with a shape of `(Np_new, )`.
        :param variances: The variances of the evaluations with a shape of `(Np_new, )`.
        """

        points_np = np.array(self.points)  # (Np, f)

        kernel_self = self.calculate_kernel(new_points, new_points)  # (Np_new, Np_new)
        kernel_cross = self.calculate_kernel(new_points, points_np)  # (Np_new, Np)

        # Update self._k_mx with the new kernel parts; kernel_self and kernel_cross.
        # Before the update, self._k_mx has a shape of (Np, Np).
        self._k_mx = np.concatenate([
            np.concatenate([self._k_mx, kernel_cross], axis=0),  # (Np + Np_new, Np)
            np.concatenate([kernel_cross, kernel_self], axis=1).T  # (Np_new, Np + Np_new).T
        ], axis=1)  # (Np + Np_new, Np + Np_new)

        # Update points, values and variances
        self.points = np.concatenate([self.points, new_points], axis=0)
        self.values = np.concatenate([self.values, values], axis=0)
        self.variances = np.concatenate([self.variances, variances], axis=0)

        # Update the inverse kernel matrix.
        self._k_inv_mx = np.linalg.inv(self._k_mx + np.diag(self.variances))

    def evaluate_at(self, test_points: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Evaluates the gaussian process at certain test point values.
        The algorithm is constructed from the equations found in the following book:
        "C. E. Rasmussen & C. K. I. Williams,
        Gaussian Processes for Machine Learning,
        the MIT Press, 2006, ISBN 026218253X".

        :param test_points: The points at which the GP should be evaluated.
            Should have a shape of `(Npt, f)` with Npt number of test points and f number of features.
        :return: The GP means and variances, both with a shape of `(Npt, )`.
        """

        k_star_mx = self.calculate_kernel(test_points, self.points)  # (Npt, Np)
        k_star_star_mx = self.calculate_kernel(test_points, test_points)  # (Npt, Npt)

        f_mean = k_star_mx @ self._k_inv_mx @ self.values  # (Npt, Np) (Np, Np) (Np, ) = (Npt, )
        f_var = k_star_star_mx - k_star_mx @ self._k_inv_mx @ k_star_mx.T  # (Npt, Npt)

        return f_mean, np.diag(f_var)

    def calculate_flatness(self) -> float:
        """
        Calculates the flatness of the target function.
        Larger values mean more flat surfaces with respect to the mean variability of the datapoints.
        Mathematically, it is the ratio of the mean variance and the variance of
            the measurements under a square root.
        If the variance cannot be calculated, it returns -1.

        :return: The flatness descriptor.
        """

        if len(self.values) < 2:
            return -1.

        flatness = np.sqrt(
            np.mean(self.variances) / np.var(self.values)
        )
        return flatness

