import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def preprocess_series(x, demean=True, detrend_linear=False):
    """
    Basic preprocessing for period estimation.
    """
    x = np.asarray(x, dtype=float)
    if demean:
        x = x - np.mean(x)

    if detrend_linear:
        n = np.arange(x.size)
        # least-squares linear fit
        A = np.vstack([n, np.ones_like(n)]).T
        slope, intercept = np.linalg.lstsq(A, x, rcond=None)[0]
        x = x - (slope * n + intercept)

    return x

def dominant_period_cepstrum(x, min_period=2, max_period=None, demean=True, detrend_linear=False):
    """
    Estimate dominant period using real cepstrum:
    cepstrum = IFFT(log(|FFT(x)|))

    Returns
    -------
    period : int or None
        Estimated period in samples.
    quefrency : ndarray
    cep : ndarray
    """
    x = preprocess_series(x, demean=demean, detrend_linear=detrend_linear)
    N = x.size

    X = np.fft.rfft(x)
    mag = np.abs(X)
    mag[mag == 0] = 1e-12

    logmag = np.log(mag)
    cep = np.fft.irfft(logmag, n=N)  # real cepstrum length N
    quefrency = np.arange(N)

    if max_period is None:
        max_period = N - 1

    lo = int(max(min_period, 1))
    hi = int(min(max_period, N - 1))
    if hi <= lo:
        return None, quefrency, cep

    period = lo + int(np.argmax(cep[lo:hi+1]))
    return period, quefrency, cep

def split_into_periods(x, period, discard_remainder=True):
    """
    Split a time series into contiguous periods.

    Parameters
    ----------
    x : array-like, shape (N,)
        Time series.
    period : int
        Period length in samples.
    discard_remainder : bool
        If True, drop incomplete final period.

    Returns
    -------
    blocks : ndarray, shape (n_periods, period)
        Period-wise blocks.
    """
    x = np.asarray(x, dtype=float)
    N = x.size
    n_blocks = N // period

    if n_blocks == 0:
        return np.empty((0, period))

    trimmed = x[:n_blocks * period]
    blocks = trimmed.reshape(n_blocks, period)
    return blocks

def average_per_period(x, period):
    """
    Compute the average value per period, then average over periods.

    Parameters
    ----------
    x : array-like, shape (N,)
        Time series.
    period : int
        Period length in samples.

    Returns
    -------
    avg_T : float
        Average per period.
    per_period_avgs : ndarray
        Average value in each period.
    """
    x = np.asarray(x, dtype=float)
    N = x.size

    n_periods = N // period
    if n_periods == 0:
        return np.nan, np.array([])

    trimmed = x[:n_periods * period]
    blocks = trimmed.reshape(n_periods, period)

    per_period_avgs = np.mean(blocks, axis=1)
    avg_T = float(np.mean(per_period_avgs))

    return avg_T, per_period_avgs


def period_averaged_variance(x, period):
    """
    Average variance computed within each period.

    Returns
    -------
    var_T : float
        Mean variance per period.
    vars_per_period : ndarray
        Variance in each period.
    """
    blocks = split_into_periods(x, period)
    if blocks.size == 0:
        return np.nan, np.array([])

    vars_per_period = np.var(blocks, axis=1, ddof=0)
    var_T = np.mean(vars_per_period)
    return var_T, vars_per_period

def directional_increment_rates_per_period(x, period):
    """
    Compute average increase and decrease rates per period.

    Returns
    -------
    r_plus : float
        Mean positive increment per period.
    r_minus : float
        Mean absolute negative increment per period.
    r_net : float
        Mean signed increment per period.
    """
    x = np.asarray(x, dtype=float)
    blocks = split_into_periods(x, period)

    if blocks.size == 0:
        return np.nan, np.nan, np.nan

    r_plus_list = []
    r_minus_list = []
    r_net_list = []

    for block in blocks:
        dx = np.diff(block)

        pos = dx[dx > 0]
        neg = dx[dx < 0]

        r_plus_list.append(np.mean(pos) if pos.size else 0.0)
        r_minus_list.append(np.mean(np.abs(neg)) if neg.size else 0.0)
        r_net_list.append(np.mean(dx))

    return (
        float(np.mean(r_plus_list)),
        float(np.mean(r_minus_list)),
        float(np.mean(r_net_list))
    )

def directional_asymmetry(x, period):
    r_plus, r_minus, _ = directional_increment_rates_per_period(x, period)
    denom = r_plus + r_minus
    if denom == 0:
        return 0.0
    return (r_plus - r_minus) / denom

def period_statistics_summary(
    x,
    min_period=10,
    max_period=100
):
    """
    Compute period-based statistics and return a pandas DataFrame summary.

    Parameters
    ----------
    x : array-like
        Time series (e.g. distance vs Floquet kicks).
    min_period, max_period : int
        Search bounds for dominant period estimation (cepstrum).

    Returns
    -------
    df : pandas.DataFrame
        Summary table with quantities, descriptions, and values.
    diagnostics : dict
        Dictionary containing detailed arrays for further analysis.
    """
    x = np.asarray(x, dtype=float)

    # --- Dominant period (cepstrum) ---
    period, quefrency, cep = dominant_period_cepstrum(
        x,
        min_period=min_period,
        max_period=max_period
    )

    if period is None or np.isnan(period):
        raise ValueError("No dominant period detected.")

    # --- Period-based quantities ---
    avg_T, per_period_avgs = average_per_period(x, period)
    var_T, vars_per_period = period_averaged_variance(x, period)
    asymmetry = directional_asymmetry(x, period)

    # --- Assemble DataFrame ---
    data = [
        {
            "quantity": "dominant period",
            "value": period
        },
        {
            "quantity": "average per period",
            "value": avg_T
        },
        {
            "quantity": "variance_per_period",
            "value": var_T
        },
        {
            "quantity": "directional_asymmetry",
            "value": asymmetry
        }
    ]

    df = pd.DataFrame(data)

    # --- Diagnostics (returned separately) ---
    diagnostics = {
        "period": period,
        "quefrency": quefrency,
        "cepstrum": cep,
        "per_period_averages": per_period_avgs,
        "per_period_variances": vars_per_period
    }

    return df, diagnostics
