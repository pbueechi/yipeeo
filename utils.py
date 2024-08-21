""" Collects general-purpose functions used across different modules. """

import numpy as np
import pandas as pd
from scipy.stats import norm
from numba import njit
from numba import prange
from datetime import timedelta


def mdim_linregress(ts1, ts2, nan_out=np.nan):
    """
    Applies linear regression on given ts1 and ts2 time series pair
    ts1 and ts2 could be multi-dimensional data arrays but
    the first dimension should be the iteration over time.

    ts1 and ts2 can be also masked numpy arrays, but both arrays should have
    the same mask defined.

    It calculate parameters needed for regression model according to
    http://mathworld.wolfram.com/LeastSquaresFitting.html

    Note
    ----
    Please use data type np.float64 for ts1 and ts2 in order to get
    full precision for the computed slope and intercept parameters. If
    ts1 and ts2 are np.float32, results can not be as accurate.

    Parameters
    ----------
    ts1 : numpy.ndarray (float64)
        1d, 2d or 3d time series.
    ts2 : numpy.ndarray (float64)
        1d, 2d, or 3d time series
    nan_out : numerical, optional
        NaN value of output (default: np.nan)

    Returns
    -------
    slope : numpy.ndarray
        Slope parameter.
    intercept : numpy.ndarray
        Intercept parameter.
    corr : numpy.ndarray
        Correlation coefficient.
    mean_ts1 : numpy.ndarray
        Mean of time series 1.
    mean_ts2 : numpy.ndarray
        Mean of time series 2.

    """
    sum_ts1 = np.sum(ts1, axis=0)
    sum_ts2 = np.sum(ts2, axis=0)
    sum_ts1_p2 = np.sum(ts1 ** 2, axis=0)
    sum_ts2_p2 = np.sum(ts2 ** 2, axis=0)
    sum_ts1ts2 = np.sum(ts1 * ts2, axis=0)

    n = np.sum(~np.isnan(ts1), axis=0)
    denominator = (n * sum_ts1_p2 - sum_ts1 ** 2)
    intercept = (sum_ts2 * sum_ts1_p2 - sum_ts1 * sum_ts1ts2) / denominator
    slope = (n * sum_ts1ts2 - sum_ts1 * sum_ts2) / denominator

    mean_ts1 = sum_ts1 / n
    mean_ts2 = sum_ts2 / n
    ss_ts1 = sum_ts1_p2 - n * mean_ts1 ** 2
    ss_ts2 = sum_ts2_p2 - n * mean_ts2 ** 2
    ss_ts1ts2 = sum_ts1ts2 - n * mean_ts1 * mean_ts2
    corr = ss_ts1ts2 / np.sqrt(ss_ts1 * ss_ts2)

    if np.isscalar(slope):
        if ~np.isfinite(slope):
            slope = nan_out
        if ~np.isfinite(intercept):
            intercept = nan_out
        if ~np.isfinite(corr):
            corr = nan_out
        if ~np.isfinite(mean_ts1):
            mean_ts1 = nan_out
        if ~np.isfinite(mean_ts2):
            mean_ts2 = nan_out
    else:
        slope[~np.isfinite(slope)] = nan_out
        intercept[~np.isfinite(intercept)] = nan_out
        corr[~np.isfinite(corr)] = nan_out
        mean_ts1[~np.isfinite(mean_ts1)] = nan_out
        mean_ts2[~np.isfinite(mean_ts2)] = nan_out

    return slope, intercept, corr, mean_ts1, mean_ts2


@njit(parallel=True)
def find_upper_outliers(ar, outliers, k_sigma=1.):
    """
    Classifies every value above `mean + k_sigma * sigma` as an outlier.

    Parameters
    ----------
    ar : np.ndarray
        2D array to search outliers from.
    outliers : np.ndarray
        Empty outliers mask matching the shape of `ar`.
    k_sigma : float, optional
        Multiples of the standard deviation used for excluding values outside of `ar`'s lower range (defaults to 2).

    Returns
    -------
    outliers : np.ndarray
        Outliers mask, where 1 marks an outlier and 0 not.

    """
    mean_val = np.nanmean(ar)
    std_val = np.nanstd(ar)
    upper_bound = mean_val + k_sigma * std_val
    n_rows, n_cols = ar.shape
    for i in prange(n_rows):
        for j in prange(n_cols):
            val = ar[i, j]
            if np.isnan(val) or (val > upper_bound):
                outliers[i, j] = True

    return outliers


def select_tasks(tasks, task_id=1, num_tasks=1):
    """
    Selects `num_tasks` tasks at the given `task_id` from all `tasks`.

    Parameters
    ----------
    tasks : list
        Task list.
    task_id : int, optional
        Index on the `len(tasks)/num_tasks` axis (defaults to 1).
    num_tasks : int, optional
        Number tasks to be selected from `tasks` (defaults to 1).

    Returns
    -------
    list
        Subset of `tasks` containing the elements at the index defined by `task_id` and `num_tasks`.

    """
    n_tasks = len(tasks)
    task_step = n_tasks / float(num_tasks)
    task_idxs = [int(round(i * task_step)) for i in range(num_tasks)] + [n_tasks]

    return tasks[task_idxs[task_id - 1]:task_idxs[task_id]]


def polygon2pixels(polygon):
    """
    Transforms the coordinates of the given polygon to pixel coordinates.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        Polygon to extract the coordinates from.

    Returns
    -------
    float, float
        Pixel columns and row coordinates.

    Notes
    -----
    Pixel coordinates are assumed to refer to the upper-left corner.

    """
    xs, ys = zip(*polygon.exterior.coords)
    min_x, max_y = min(xs), max(ys)
    cols = np.array(xs) - min_x
    rows = np.abs(np.array(ys) - max_y)

    return cols/10., rows/10.

def compute_fig_size(diag, ratio):
    """
    Computes size of a matplotlib figure for dynamic creation.

    Parameters
    ----------
    diag : float
        Length of the diagonal of the figure.
    ratio : float
        Ratio of height to width of the figure.

    Returns
    -------
    w, h : float, float
        Width and height of the figure.

    """
    w = np.sqrt(diag**2/(1 + ratio**2))
    h = np.sqrt(diag**2 - w**2)

    return w, h


def db2lin(x):
    """
    Converts value from dB to linear units.

    Parameters
    ----------
    x : number
        Value in dB.

    Returns
    -------
    float
        Value in linear units.

    """
    return 10**(x/10.)


def lin2db(x):
    """
    Converts value from linear to dB units.

    Parameters
    ----------
    x : number
        Value in linear units.

    Returns
    -------
    float
        Value in dB.

    """
    return 10 * np.log10(x)


def rolling_mean(ar, n=5, in_db=False):
    """
    Computes a rolling mean with a window length of `2*n+1` on a time series.

    Parameters
    ----------
    ar : np.ndarray
        Time series to apply a rolling mean on.
    n : int, optional
        Number of neighbouring values to be included in the mean calculation (defaults to 5).

    Returns
    -------
    np.ndarray
        Smoothed time series.

    """
    smoothed_vals = []
    nan_mask = np.isnan(ar)
    lin_ar = db2lin(ar) if in_db else ar
    lin_ar[nan_mask] = np.nan
    for i in range(n, len(ar)-n):
        smoothed_val = np.nanmean(lin_ar[i-n:i+n])
        smoothed_vals.append(smoothed_val)
    smoothed_ar = lin2db(np.array(smoothed_vals)) if in_db else np.array(smoothed_vals)
    return np.array([np.nan]*n + smoothed_ar.tolist() + [np.nan]*n)


def rolling_mean_ts(ar_fine, ts_fine, ts_coarse, d=6, db=True, gauss=False):
    """
    Computes a rolling mean over the values `ar_fine` with the corresponding `ts_fine` timestamps at the given coarser
    temporal sampling `ts_coarse`.

    Parameters
    ----------
    ar_fine : np.ndarray
        Time series to apply a rolling mean on.
    ts_fine : np.ndarray
        Timestamps corresponding to `ar_fine`.
    ts_coarse : np.ndarray
        Timestamps corresponding to the target/smoothed time series.
    d : int, optional
        Number of neighbouring days to be included in the mean calculation (defaults to 6).
    db : bool, optional
        True if the time series contains decibel values (default).
    gauss : bool, optional
        True if Gaussian weighted averaging should be applied (defaults to false).

    Returns
    -------
    np.ndarray
        Smoothed time series.

    """
    gauss_fun = norm(loc = 0., scale = d/3.)
    nan_mask = np.isnan(ar_fine)
    ar_fine = db2lin(ar_fine) if db else ar_fine
    ar_fine[nan_mask] = np.nan
    ar_coarse = np.ones(len(ts_coarse)) * np.nan
    for i, t in enumerate(ts_coarse):
        idxs = (ts_fine >= (t - timedelta(days=d))) & (ts_fine <= (t + timedelta(days=d)))
        if not any(idxs):
            continue
        if gauss:
            w = gauss_fun.pdf([abs(pd.Timedelta(t_diff).days)
                               for t_diff in ts_fine[idxs] - np.array(t, dtype='datetime64')])
            data = ar_fine[idxs].data.compute()
            nan_mask = np.isnan(data)
            ar_coarse[i] = np.sum(w[~nan_mask] * data[~nan_mask])/np.sum(w[~nan_mask])
        else:
            ar_coarse[i] = np.nanmean(ar_fine[idxs])
    ar_coarse = lin2db(ar_coarse) if db else ar_coarse
    return ar_coarse



def rolling_op_ts(ar_fine, ts_fine, ts_coarse, op, d=6):
    """
    Computes a rolling reduction/operation over the values `ar_fine` with the corresponding `ts_fine` timestamps at
    the given coarser temporal sampling `ts_coarse`.

    Parameters
    ----------
    ar_fine : np.ndarray
        Time series to apply a rolling mean on.
    ts_fine : np.ndarray
        Timestamps corresponding to `ar_fine`.
    ts_coarse : np.ndarray
        Timestamps corresponding to the target/smoothed time series.
    op : callable
        Reducer.
    d : int, optional
        Number of neighbouring days to be included in the mean calculation (defaults to 6).

    Returns
    -------
    np.ndarray
        Smoothed time series.

    """
    ar_coarse = np.ones(len(ts_coarse)) * np.nan
    for i, t in enumerate(ts_coarse):
        idxs = (ts_fine >= (t - timedelta(days=d))) & (ts_fine <= (t + timedelta(days=d)))
        if not any(idxs):
            continue
        ar_coarse[i] = op(ar_fine[idxs])

    return ar_coarse

