
import numpy as np
from scipy.optimize import curve_fit

from .models import exponential_rise


def fit_exponential_rise(
    time_s,
    fluorescence,
    fit_start=None,
    fit_end=None,
    min_points=30,
    min_duration_s=10.0,
):
    """Fit an exponential-rise model to fluorescence data."""

    time_s = np.asarray(time_s, dtype=float)
    fluorescence = np.asarray(fluorescence, dtype=float)

    if time_s.ndim != 1 or fluorescence.ndim != 1:
        raise ValueError("Inputs must be one-dimensional arrays.")

    if len(time_s) != len(fluorescence):
        raise ValueError("Input arrays must have equal length.")

    if len(time_s) < 5:
        raise ValueError("At least 5 data points are required.")

    if not np.isfinite(time_s).all():
        raise ValueError("Time values must be finite.")

    if not np.isfinite(fluorescence).all():
        raise ValueError("Fluorescence values must be finite.")

    if not np.all(np.diff(time_s) > 0):
        raise ValueError("Time values must be strictly increasing.")

    start = time_s[0] if fit_start is None else float(fit_start)
    end = time_s[-1] if fit_end is None else float(fit_end)

    if start >= end:
        raise ValueError("fit_start must be smaller than fit_end.")

    mask = (time_s >= start) & (time_s <= end)
    t = time_s[mask]
    y = fluorescence[mask]

    if len(t) < min_points:
        raise ValueError(
            f"Fit interval has {len(t)} points; "
            f"at least {min_points} are required."
        )

    duration = t[-1] - t[0]

    if duration < min_duration_s:
        raise ValueError(
            f"Fit duration is {duration:.2f} s; "
            f"at least {min_duration_s:.2f} s are required."
        )

    baseline_guess = float(np.median(y[:max(1, len(y) // 5)]))
    amplitude_guess = float(y[-1] - baseline_guess)

    if abs(amplitude_guess) < 1e-12:
        amplitude_guess = float(np.ptp(y))

    if abs(amplitude_guess) < 1e-12:
        raise ValueError("Signal variation is too small to fit.")

    onset_guess = float(t[0])
    tau_guess = duration / 5.0

    lower_bounds = [
        -np.inf,
        -np.inf,
        t[0] - 0.25 * duration,
        max(duration * 1e-5, 1e-9),
    ]

    upper_bounds = [
        np.inf,
        np.inf,
        t[-1],
        duration * 10.0,
    ]

    params, covariance = curve_fit(
        exponential_rise,
        t,
        y,
        p0=[
            baseline_guess,
            amplitude_guess,
            onset_guess,
            tau_guess,
        ],
        bounds=(lower_bounds, upper_bounds),
        maxfev=20000,
    )

    baseline, amplitude, onset_s, tau_s = params

    fitted = exponential_rise(t, *params)
    residuals = y - fitted

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y))**2))

    r_squared = (
        float(1.0 - ss_res / ss_tot)
        if ss_tot > 0
        else float("nan")
    )

    residual_rmse = float(np.sqrt(np.mean(residuals**2)))
    residual_std = float(np.std(residuals, ddof=1))
    fit_duration_s = float(t[-1] - t[0])
    points_used = int(len(t))
    tau_to_duration = float(tau_s / fit_duration_s)

    return {
        "baseline": float(baseline),
        "amplitude": float(amplitude),
        "onset_s": float(onset_s),
        "tau_s": float(tau_s),
        "r_squared": r_squared,
        "residual_rmse": residual_rmse,
        "residual_std": residual_std,
        "fit_duration_s": fit_duration_s,
        "points_used": points_used,
        "tau_to_duration": tau_to_duration,
        "fit_time_s": t,
        "observed": y,
        "fitted": fitted,
        "residuals": residuals,
        "covariance": covariance,
    }