from __future__ import annotations

"""Assign ICD-O topography codes to the anatomical groups used by CNVSIM."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RootMappingResult:
    """Raw topography code, three-character code, and CNVSIM grouped root."""

    icdo_topography_id: str
    core: str | None
    root: str | None


def icdot_root_grouped(icdot_id: str) -> str | None:
    """Map one ICD-O-T code to the anatomical grouping used in the paper.

    The grouping combines related three-character topography codes before
    subtype discovery so that very small site labels do not define separate
    comparison spaces. It is a CNVSIM analysis policy rather than an official
    additional level of the ICD-O hierarchy.

    Examples
    --------
    - C00..C14  -> C00_C14
    - C18..C20  -> C18_C20
    - etc.

    Returns
    -------
    Grouped root label (e.g., "C50", "C60_C63"), or None if input is invalid.
    """
    if not isinstance(icdot_id, str):
        return None

    code = icdot_id.split("icdot-")[-1] if "icdot-" in icdot_id else icdot_id
    if not code or not code.startswith("C"):
        return None

    core = code[:3]
    try:
        num = int(core[1:3])
    except ValueError:
        return None

    if 0 <= num <= 14:
        return "C00_C14"
    if num in (15, 17, 21, 26):
        return "C15_C17_C21_C26"
    if 18 <= num <= 20:
        return "C18_C20"
    if 22 <= num <= 24:
        return "C22_C24"
    if num == 25:
        return "C25"
    if 30 <= num <= 39:
        return "C30_C39"
    if num in (40, 41, 48):
        return "C40_C41_C48"
    if num == 42:
        return "C42"
    if num == 44:
        return "C44"
    if num == 47:
        return "C47"
    if num == 49:
        return "C49"
    if num == 50:
        return "C50"
    if 51 <= num <= 58:
        return "C51_C58"
    if 60 <= num <= 63:
        return "C60_C63"
    if 64 <= num <= 68:
        return "C64_C68"
    if num in (69, 70, 72):
        return "C69_C70_C72"
    if num == 71:
        return "C71"
    if 73 <= num <= 75:
        return "C73_C75"
    if num == 76:
        return "C76"
    if num == 77:
        return "C77"
    if num == 80:
        return "C80"

    return core


def assign_root(icdo_topography_id: str) -> RootMappingResult:
    """Return both the parsed three-character code and its grouped root."""

    if not isinstance(icdo_topography_id, str):
        return RootMappingResult(str(icdo_topography_id), None, None)

    code = icdo_topography_id.split("icdot-")[-1] if "icdot-" in icdo_topography_id else icdo_topography_id
    core = code[:3] if code and code.startswith("C") and len(code) >= 3 else None
    root = icdot_root_grouped(icdo_topography_id)
    return RootMappingResult(icdo_topography_id=icdo_topography_id, core=core, root=root)
