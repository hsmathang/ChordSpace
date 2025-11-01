"""Utilities for comparing normalisation proposals."""

from .service import ProposalsComparisonService
from .registry import (
    PROPOSAL_INFO,
    METRIC_INFO,
    AVAILABLE_REDUCTIONS,
    PREPROCESSORS,
)

__all__ = [
    "ProposalsComparisonService",
    "PROPOSAL_INFO",
    "METRIC_INFO",
    "AVAILABLE_REDUCTIONS",
    "PREPROCESSORS",
]
