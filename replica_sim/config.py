"""Default simulation configuration."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SimulationConfig:
    p_main: float = 0.02
    max_c: int = 4
    strengths: tuple[float, ...] = (
        0.005, 0.010, 0.015, 0.020, 0.030,
        0.040, 0.050, 0.060, 0.080,
    )
    shots: int | None = None

    @property
    def scales(self) -> np.ndarray:
        return np.array([2 * c + 1 for c in range(self.max_c + 1)], dtype=float)
