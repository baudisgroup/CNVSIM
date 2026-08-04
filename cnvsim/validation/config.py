from __future__ import annotations

"""Configuration for validation workflows, separate from atlas discovery."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class HoldoutValidationParams:
    """Settings for leakage-controlled fixed-reference projection."""

    n_folds: int = 5
    test_fold: int = 0
    random_seed: int = 2701
    min_train_per_context: int = 160


@dataclass(frozen=True)
class SignatureRecoveryParams:
    """Settings for complementary-fold signature recovery."""

    n_folds: int = 5
    random_seed: int = 7301
    min_effect_features: int = 2


@dataclass(frozen=True)
class ValidationConfig:
    holdout: HoldoutValidationParams
    signature_recovery: SignatureRecoveryParams = SignatureRecoveryParams()


def _validate(config: ValidationConfig) -> None:
    params = config.holdout
    errors: list[str] = []
    if params.n_folds < 2:
        errors.append("holdout.n_folds must be at least 2")
    if not 0 <= params.test_fold < params.n_folds:
        errors.append("holdout.test_fold must be within holdout.n_folds")
    if params.min_train_per_context < 2:
        errors.append("holdout.min_train_per_context must be at least 2")
    recovery = config.signature_recovery
    if recovery.n_folds < 2:
        errors.append("signature_recovery.n_folds must be at least 2")
    if recovery.min_effect_features < 1:
        errors.append(
            "signature_recovery.min_effect_features must be at least 1"
        )
    if errors:
        raise ValueError("Invalid validation configuration:\n- " + "\n- ".join(errors))


def load_validation_config(path: str | Path) -> ValidationConfig:
    """Load projection and fixed-parameter recovery validation settings."""

    data = yaml.safe_load(Path(path).read_text()) or {}
    holdout = data.get("holdout") or {}
    recovery = data.get("signature_recovery") or {}
    config = ValidationConfig(
        holdout=HoldoutValidationParams(
            n_folds=int(holdout.get("n_folds", 5)),
            test_fold=int(holdout.get("test_fold", 0)),
            random_seed=int(holdout.get("random_seed", 2701)),
            min_train_per_context=int(holdout.get("min_train_per_context", 160)),
        ),
        signature_recovery=SignatureRecoveryParams(
            n_folds=int(recovery.get("n_folds", 5)),
            random_seed=int(recovery.get("random_seed", 7301)),
            min_effect_features=int(
                recovery.get("min_effect_features", 2)
            ),
        ),
    )
    _validate(config)
    return config
