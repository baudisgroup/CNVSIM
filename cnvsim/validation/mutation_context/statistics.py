"""Conditional inference shared by GENIE and TCGA analyses."""

import warnings
import numpy as np
from scipy.stats import chi2, hypergeom
from statsmodels.stats.contingency_tables import StratifiedTable


def exact_2x2_from_margins(strata_margins):
    """Exact two-sided conditional hypergeometric score-tail P from [(N, M, n1, a), ...]."""
    if not strata_margins:
        return np.nan
    mass = np.array([1.0], dtype=float)
    offset = 0
    observed = 0
    expected = 0.0
    for N, M, n1, a in strata_margins:
        lo = max(0, n1 - (N - M))
        hi = min(n1, M)
        x = np.arange(lo, hi + 1)
        mass = np.convolve(mass, hypergeom.pmf(x, N, M, n1))
        offset += lo
        observed += a
        expected += n1 * (M / N)
    values = np.arange(len(mass)) + offset
    tail = np.abs(values - expected) >= abs(observed - expected) - 1e-10
    return float(min(1.0, max(0.0, mass[tail].sum())))


def fast_two_by_two_exact_and_cmh(left_mask, pos_arr, st_idx, n_strata):
    """Vectorized 2x2 stratified exact convolution + asymptotic CMH across integer strata st_idx."""
    lm = left_mask.astype(int)
    pm = pos_arr.astype(int)
    N_vec = np.bincount(st_idx, minlength=n_strata)
    M_vec = np.bincount(st_idx, weights=pm, minlength=n_strata).astype(int)
    n1_vec = np.bincount(st_idx, weights=lm, minlength=n_strata).astype(int)
    a_vec = np.bincount(st_idx, weights=lm * pm, minlength=n_strata).astype(int)

    valid = (
        (N_vec > 1) & (n1_vec > 0) & (n1_vec < N_vec) & (M_vec > 0) & (M_vec < N_vec)
    )
    if not np.any(valid):
        return np.nan, np.nan, np.nan, np.nan, np.nan, 0, 0

    N_v = N_vec[valid]
    M_v = M_vec[valid]
    n1_v = n1_vec[valid]
    a_v = a_vec[valid]
    n0_v = N_v - n1_v
    b_v = n1_v - a_v
    c_v = M_v - a_v
    d_v = n0_v - c_v

    kept_n = int(N_v.sum())
    n_inf = int(valid.sum())
    margins = list(zip(N_v.tolist(), M_v.tolist(), n1_v.tolist(), a_v.tolist()))
    p_exact = exact_2x2_from_margins(margins)

    tables = np.stack(
        [np.stack([a_v, b_v], axis=0), np.stack([c_v, d_v], axis=0)], axis=0
    ).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z = StratifiedTable(tables, shift_zeros=False)
        stat = z.test_null_odds(correction=True).statistic
        p_asymp = float(chi2.sf(stat, 1))
        or_val = float(z.oddsratio_pooled)
        cilo, cihi = z.oddsratio_pooled_confint()
    return p_asymp, p_exact, or_val, float(cilo), float(cihi), n_inf, kept_n
