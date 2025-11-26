import math
import random
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from pybopt import GaussianProcess, BayesianOptimizer


def hidden_function(x: float, without_noise: bool = True) -> Tuple[float, float]:

    mu = (1.45 * x ** 2 - x) * math.sin(10 * x)

    if without_noise:
        return mu, 0.

    sigma = random.uniform(0.05, 0.3)

    return random.gauss(mu, sigma), sigma ** 2


def main():

    gp = GaussianProcess(
        length_scale=0.1,
        n_of_features=1
    )
    bo = BayesianOptimizer(
        gaussian_process=gp,
        bounds_min=np.array([0., ]),
        bounds_max=np.array([1., ]),
        solver_n_points=100
    )
    fig, ax = plt.subplots()
    ucb_lambda = 3.0

    for idx in range(100):

        position, utility = bo.suggest("upper_confidence_bound", dict(ucb_lambda=ucb_lambda))
        real_y, real_y_var = hidden_function(position[0], without_noise=False)
        bo.gaussian_process.add_point(
            new_points=position[None, ...],
            values=np.array([real_y, ]),
            variances=np.array([real_y_var, ])
        )

        ax.cla()

        plot_x = np.arange(0, 1, 0.01)
        pred_y, pred_y_var = bo.gaussian_process.evaluate_at(plot_x[..., None])
        utility = bo.upper_confidence_bound(plot_x[..., None], ucb_lambda=ucb_lambda)

        ax.plot(
            plot_x,
            list(map(lambda x: hidden_function(x)[0], plot_x)),
            color="black", label="Target Function"
        )
        ax.plot(
            plot_x, pred_y, color="red", label="Mean GP Values"
        )
        ax.fill_between(
            plot_x,
            pred_y + np.sqrt(pred_y_var),
            pred_y - np.sqrt(pred_y_var),
            color="red", alpha=0.5, label="One Sigma Interval"
        )
        ax.plot(
            plot_x,
            utility,
            color="green", label="Utility Function"
        )
        ax.errorbar(
            bo.gaussian_process.points[:, 0],
            bo.gaussian_process.values,
            yerr=np.sqrt(bo.gaussian_process.variances),
            marker=".", ls="none", color="blue", label="Guesses"
        )
        ax.set_xlabel("Variable Positions")
        ax.set_ylabel("Function Value")
        ax.legend(loc="upper right")

        plt.pause(0.5)

    plt.show()


if __name__ == "__main__":
    main()
