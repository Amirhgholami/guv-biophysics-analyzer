
import numpy as np

from .fitting import fit_exponential_rise


def find_best_fit(
    time_s,
    fluorescence,
    min_points=100,
    min_duration_s=20.0,
    max_candidates=300,
):
    """
    Search candidate time windows and return the best exponential fit.

    This is an initial screening algorithm, not a guarantee of
    physical validity.
    """
    time_s = np.asarray(time_s, dtype=float)
    fluorescence = np.asarray(fluorescence, dtype=float)

    if time_s.ndim != 1 or fluorescence.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")

    if len(time_s) != len(fluorescence):
        raise ValueError("Input arrays must have equal length.")

    if len(time_s) < min_points:
        raise ValueError("Not enough data points for automatic fitting.")

    if not np.isfinite(time_s).all() or not np.isfinite(fluorescence).all():
        raise ValueError("Inputs contain non-finite values.")

    if not np.all(np.diff(time_s) > 0):
        raise ValueError("Time values must be strictly increasing.")

    n = len(time_s)

    # Candidate start/end indices distributed across the recording.
    starts = np.unique(
        np.linspace(0, n - min_points, min(20, n - min_points + 1), dtype=int)
    )
    ends = np.unique(
        np.linspace(min_points, n, min(20, n - min_points + 1), dtype=int)
    )

    candidates = []

    for start_idx in starts:
        for end_idx in ends:
            if end_idx - start_idx < min_points:
                continue

            t0 = time_s[start_idx]
            t1 = time_s[end_idx - 1]

            if t1 - t0 < min_duration_s:
                continue

            candidates.append((start_idx, end_idx))

    if len(candidates) > max_candidates:
        selected = np.linspace(
            0, len(candidates) - 1, max_candidates, dtype=int
        )
        candidates = [candidates[i] for i in selected]

    best_result = None
    best_score = -np.inf
    successful_fits = 0

    for start_idx, end_idx in candidates:
        t = time_s[start_idx:end_idx]
        y = fluorescence[start_idx:end_idx]

        try:
            result = fit_exponential_rise(
                t,
                y,
                min_points=min_points,
                min_duration_s=min_duration_s,
            )
        except (ValueError, RuntimeError, FloatingPointError):
            continue

        successful_fits += 1

        tau = result["tau_s"]
        duration = result["fit_duration_s"]
        r_squared = result["r_squared"]

        if not np.isfinite(r_squared) or not np.isfinite(tau):
            continue

        # Reject fits whose tau is too large relative to the fit window.
        if tau <= 0 or tau > duration / 2:
            continue

        # Penalize fits with weak explanatory power.
        score = r_squared

        if score > best_score:
            best_score = score
            best_result = dict(result)
            best_result["fit_start_s"] = float(t[0])
            best_result["fit_end_s"] = float(t[-1])
            best_result["candidate_points"] = int(len(t))

    if best_result is None:
        return {
            "success": False,
            "message": (
                "No candidate passed the initial screening criteria. "
                "This does not necessarily mean the experiment is invalid."
            ),
            "candidates_tested": len(candidates),
            "successful_fits": successful_fits,
        }

    best_result["success"] = True
    best_result["message"] = "Candidate fit found; scientific validation required."
    best_result["candidates_tested"] = len(candidates)
    best_result["successful_fits"] = successful_fits

    return best_result
