
import matplotlib.pyplot as plt
import numpy as np

from .models import exponential_rise


def save_fit_plot(data, result, output_path):
    """Save fluorescence data, fitted curve, and residual diagnostics."""

    time_s = data["time_s"].to_numpy(dtype=float)
    fluorescence = data["fluorescence"].to_numpy(dtype=float)

    dense_time = np.linspace(time_s[0], time_s[-1], 600)

    params = [
        result["baseline"],
        result["amplitude"],
        result["onset_s"],
        result["tau_s"],
    ]

    dense_fit = exponential_rise(dense_time, *params)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # Top panel: measured signal and fitted model
    axes[0].scatter(
        time_s,
        fluorescence,
        s=16,
        alpha=0.75,
        label="Measured signal",
    )

    axes[0].plot(
        dense_time,
        dense_fit,
        color="red",
        linewidth=2,
        label="Exponential-rise fit",
    )

    axes[0].axvline(
        result["onset_s"],
        linestyle="--",
        alpha=0.7,
        label="Fitted onset",
    )

    axes[0].axvline(
        result["onset_s"] + result["tau_s"],
        linestyle=":",
        alpha=0.8,
        label="Onset + tau",
    )

    axes[0].set_ylabel("Fluorescence intensity")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    # Bottom panel: residuals within the fitted interval
    axes[1].axhline(0, linestyle="--", linewidth=1)

    axes[1].scatter(
        result["fit_time_s"],
        result["residuals"],
        s=14,
    )

    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Residual")
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)



def save_raw_plot(data, output_path):
    """Save raw fluorescence intensity versus time."""

    time_s = data["time_s"].to_numpy(dtype=float)
    fluorescence = data["fluorescence"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(time_s, fluorescence, linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fluorescence intensity")
    ax.set_title("GUV Fluorescence vs Time")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
