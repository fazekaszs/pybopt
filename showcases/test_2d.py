import math
import random
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from pybopt import GaussianProcess, BayesianOptimizer


def hidden_function(x: float, y: float, without_noise: bool = True) -> Tuple[float, float]:

    f1 = 2 * math.sin(x) / (x ** 2 + 1)
    f2 = 3 * math.sin(3 * y - 1) / (y ** 2 + 1) - y ** 2 / 4.0 + 4
    f = f1 * f2

    if without_noise:
        return f, 0.

    var_f = random.uniform(0.1, 1.0)
    f += random.gauss(0.0, math.sqrt(var_f))

    return f, var_f


def update_plot(
    ax,
    test_points_x,
    test_points_y,
    f_mean,
    f_var,
    utility
):

    ax[0, 1].cla()
    ax[1, 0].cla()
    ax[1, 1].cla()

    ax[0, 1].imshow(
        f_mean.reshape((len(test_points_x), len(test_points_y))),
        cmap="viridis", extent=(-6, 6, -6, 6)
    )
    ax[0, 1].set_title("Mean GP Values")

    ax[1, 0].imshow(
        f_var.reshape((len(test_points_x), len(test_points_y))),
        cmap="viridis", extent=(-6, 6, -6, 6)
    )
    ax[1, 0].set_title("Variance GP Values")

    ax[1, 1].imshow(
        utility.reshape((len(test_points_x), len(test_points_y))),
        cmap="viridis", extent=(-6, 6, -6, 6)
    )
    ax[1, 1].set_title("Utility Values")

def main():

    gp = GaussianProcess(length_scale=1.0, kernel_scaler=1.0, n_of_features=2)
    bo = BayesianOptimizer(
        gaussian_process=gp,
        bounds_min=np.array([-6., -6.]),
        bounds_max=np.array([6., 6.]),
        solver_n_points=100,
        solver_bfgs_n_iter=3
    )
    ucb_lambda = 5.0

    test_points_x = np.arange(-6, 6, 0.36)
    test_points_y = np.arange(-6, 6, 0.36)
    test_points = np.meshgrid(test_points_x, test_points_y)
    test_points = np.concatenate([
        test_points[0].flatten()[:, None], test_points[1].flatten()[:, None]
    ], axis=1)

    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(7, 7)

    real_values = np.array(list(map(lambda x: hidden_function(x[0], x[1])[0], test_points)))
    ax[0, 0].imshow(
        real_values.reshape((len(test_points_x), len(test_points_y))),
        cmap="viridis", extent=(-6, 6, -6, 6)
    )
    ax[0, 0].set_title("Target Function")

    for idx in range(1000):

        if len(bo.gaussian_process.points) < 3:

            (new_x, new_y), _ = bo.suggest("upper_confidence_bound", dict(ucb_lambda=ucb_lambda))
            f, var_f = hidden_function(new_x, new_y, without_noise=False)
            bo.gaussian_process.add_point(
                new_points=np.array([[new_x, new_y], ]),
                values=f * np.ones((1, )),
                variances=var_f * np.ones((1, )),
            )
            continue

        if idx % 10 == 0:
            f_mean, f_var = bo.gaussian_process.evaluate_at(test_points)
            utility = bo.upper_confidence_bound(test_points, ucb_lambda)
            update_plot(ax, test_points_x, test_points_y, f_mean, f_var, utility)
            plt.pause(0.1)

        (chosen_x, chosen_y), _ = bo.suggest("upper_confidence_bound", dict(ucb_lambda=ucb_lambda))
        f, var_f = hidden_function(chosen_x, chosen_y, without_noise=False)

        print(f"\rChosen x: {chosen_x:.2f}, y: {chosen_y:.2f} - f: {f:.2f} (var f: {var_f:.2f})", end="")

        bo.gaussian_process.add_point(
            new_points=np.array([[chosen_x, chosen_y], ]),
            values=f * np.ones((1, )),
            variances=var_f * np.ones((1, )),
        )

    plt.show()


if __name__ == "__main__":
    main()
