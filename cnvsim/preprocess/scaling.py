from __future__ import annotations

"""Canonical feature scaling shared by preprocessing transformations."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def make_feature_scaler(feature_scaling: str = "standard") -> StandardScaler:
    """Create the canonical variance-standardizing PCA input scaler."""

    mode = str(feature_scaling).strip().lower()
    if mode == "standard":
        return StandardScaler(with_mean=True, with_std=True)
    raise ValueError(f"feature_scaling must be 'standard'; got {feature_scaling!r}")


def l2_normalize_rows(features: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
    """Scale every sample vector to unit Euclidean norm."""

    values = features.to_numpy(dtype=np.float32)
    norms = np.sqrt((values * values).sum(axis=1, keepdims=True))
    values = values / np.maximum(norms, float(eps))
    return pd.DataFrame(values, index=features.index, columns=features.columns)
