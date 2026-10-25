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
    plt.title("$V_{th} = 1$")
    plt.legend()

    # Minimization, linear.
    x_min = 0
    x_max = 50
    xs = np.linspace(x_min, x_max, 50)
    ys = xs

    plt.subplot(222)
    plt.plot(xs, ys, label="$L_{min} = R_{sp}$", color="green")
    plt.xlabel("$R_{sp}$ ($m\\Omega$)")
    #plt.ylabel("Loss")
    plt.title("minimize $R_{sp}$")
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
    plt.title("$BV > 15$")
    plt.legend()

    # Two sided barrier.
    x_min = 2
    x_max = 3
    xs = np.linspace(2.35 + 1e-3, 2.75 - 1e-3, 100)
    ys = -np.log(xs - 2.35) - np.log(2.75 - xs)

    plt.subplot(224)
    plt.plot(xs, ys, label="$L_{ineq2} = -\\log(L_{sti} - 2.35)$\n            $- \\log(2.75 - L_{sti})$", color="orange")
    plt.axvspan(x_min, 2.35, color="orange", alpha=0.3, hatch="/")
    plt.axvspan(2.75, x_max, color="orange", alpha=0.3, hatch="/")

    plt.xlim(x_min, x_max)
    plt.xlabel("$L_{sti}$ ($\\mu m$)")
    #plt.ylabel("Loss")
    plt.title("$2.35 < L_{sti} < 2.75$")
    plt.legend()

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.35)
    plt.show()


if __name__ == "__main__":
    plt.rc("axes", titlesize=16)
    plt.rc("axes", labelsize=14)
    plt.rc("legend", fontsize=15)

    plot_constr_losses()
