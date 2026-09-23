"""Exact paper-style replica-interference simulation package."""

from .circuit import ReplicaPaperExperiment
from .analysis import log_linear_fit, rzne_two_point, richardson_two_point

__all__ = [
    "ReplicaPaperExperiment",
    "log_linear_fit",
    "rzne_two_point",
    "richardson_two_point",
]
