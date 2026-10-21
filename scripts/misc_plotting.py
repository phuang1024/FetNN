"""Misc plotting.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_constr_losses():
    """Plot loss funcs corr. to different types of constraints.
    """
    plt.figure()

    # Equality, MSE.
    x_min = 0
    x_max = 3
    xs = np.linspace(x_min, x_max, 50)
    ys = (xs - 1) ** 2

    plt.subplot(221)
    plt.plot(xs, ys, label="$L_{eq} = (V_{th} - 1)^2$")
    plt.xlabel("$V_{th}$ (Volts)")
    plt.ylabel("Loss")
    plt.title("Equality: $V_{th} = 1$")
    plt.legend()

    # Minimization, linear.
    x_min = 0
    x_max = 50
    xs = np.linspace(x_min, x_max, 50)
    ys = xs

    plt.subplot(222)
    plt.plot(xs, ys, label="$L_{min} = R_{sp}$", color="green")
    plt.xlabel("$R_{sp}$ ($m\\Omega$)")
    plt.ylabel("Loss")
    plt.title("Minimization: minimize $R_{sp}$")
    plt.legend()

    # One sided barrier.
    x_min = 5
    x_max = 30
    xs = np.linspace(15 + 1e-3, x_max, 100)
    ys = -np.log(xs - 15)

    plt.subplot(223)
    plt.plot(xs, ys, label="$L_{ineq1} = -\\log(BV - 15)$", color="red")
    plt.axvspan(x_min, 15, color="red", alpha=0.3, hatch="/")

    plt.xlim(x_min, x_max)
    plt.xlabel("$BV$ (Volts)")
    plt.ylabel("Loss")
    plt.title("One-sided inequality: $BV > 15$")
    plt.legend()

    # Two sided barrier.
    x_min = 2
    x_max = 3
    xs = np.linspace(2.35 + 1e-3, 2.75 - 1e-3, 100)
    ys = -np.log(xs - 2.35) - np.log(2.75 - xs)

    plt.subplot(224)
    plt.plot(xs, ys, label="$L_{ineq2} = -\\log(L_{sti} - 2.35) - \\log(2.75 - L_{sti})$", color="orange")
    plt.axvspan(x_min, 2.35, color="orange", alpha=0.3, hatch="/")
    plt.axvspan(2.75, x_max, color="orange", alpha=0.3, hatch="/")

    plt.xlim(x_min, x_max)
    plt.xlabel("$L_{sti}$ ($\\mu m$)")
    plt.ylabel("Loss")
    plt.title("Two-sided inequality: $L_{sti} \\in (2.35, 2.75)$")
    plt.legend()

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.25)
    plt.show()


if __name__ == "__main__":
    plot_constr_losses()
