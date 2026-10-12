import numpy as np

def exponential_rise(time_s, baseline, amplitude, onset_s, tau_s):
    """
    Model an exponential rise in fluorescence intensity.

    Parameters
    ----------
    time_s : array-like
        Time in seconds.
    baseline : float
        Baseline fluorescence intensity.
    amplitude : float
        Change in fluorescence intensity.
    onset_s : float
        Response onset time in seconds.
    tau_s : float
        Response time constant in seconds.

    Returns
    -------
    numpy.ndarray
        Predicted fluorescence intensity.
    """
    time_s = np.asarray(time_s, dtype=float)

    if tau_s <= 0:
        raise ValueError("tau_s must be positive.")

    elapsed_s = np.maximum(time_s - onset_s, 0.0)

    response = baseline + amplitude * (
        1.0 - np.exp(-elapsed_s / tau_s)
    )

    return np.where(time_s < onset_s, baseline, response)
