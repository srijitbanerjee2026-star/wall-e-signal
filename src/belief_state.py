"""Bayesian grid filter for noisy directional signal measurements."""

from __future__ import annotations

import math
from typing import Any


Coordinate = tuple[int, int]


class BeliefState:
    """Maintain and update a normalized posterior over every traversable cell."""

    def __init__(
        self,
        width: int,
        height: int,
        obstacles: list[Coordinate] | set[Coordinate],
        ambient_sigma: float = 0.13,
        scan_sigma: float = 0.055,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Belief grid dimensions must be positive.")
        self.cells = [
            (x, y)
            for y in range(height)
            for x in range(width)
            if (x, y) not in set(obstacles)
        ]
        if not self.cells:
            raise ValueError("Belief grid must contain at least one traversable cell.")
        if ambient_sigma <= 0 or scan_sigma <= 0:
            raise ValueError("Sensor noise values must be positive.")
        self.ambient_sigma = ambient_sigma
        self.scan_sigma = scan_sigma
        self.probabilities = {cell: 1.0 / len(self.cells) for cell in self.cells}
        self.evidence_count = 0
        self.discovered_clues: set[str] = set()

    def update(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Apply signal-vector and newly discovered clue likelihoods."""
        position = tuple(observation["position"])
        vector = observation["signal_vector"]
        observed = (float(vector["x"]), float(vector["y"]))
        sigma = self.scan_sigma if observation.get("signal_quality") == "scan" else self.ambient_sigma

        log_weights: dict[Coordinate, float] = {}
        for cell, prior in self.probabilities.items():
            if prior <= 0:
                continue
            predicted = self._expected_vector(position, cell)
            log_likelihood = -sum((actual - expected) ** 2 for actual, expected in zip(observed, predicted)) / (
                2 * sigma**2
            )
            for clue in observation.get("discovered_clues", []):
                bearing = clue["bearing"]
                clue_origin = tuple(clue.get("origin", position))
                predicted_bearing = self._unit_vector(clue_origin, cell)
                clue_sigma = max(0.06, (1.0 - float(clue["confidence"])) * 0.8)
                log_likelihood -= sum(
                    (float(bearing[axis]) - predicted_bearing[index]) ** 2
                    for index, axis in enumerate(("x", "y"))
                ) / (2 * clue_sigma**2)
            log_weights[cell] = math.log(prior) + log_likelihood

        if not log_weights:
            raise RuntimeError("Belief update has no finite prior probability mass.")
        peak = max(log_weights.values())
        weights = {cell: math.exp(value - peak) for cell, value in log_weights.items()}
        total = sum(weights.values())
        if not math.isfinite(total) or total <= 0:
            raise RuntimeError("Belief update could not normalize its likelihoods.")
        self.probabilities = {cell: weights.get(cell, 0.0) / total for cell in self.cells}

        for clue in observation.get("discovered_clues", []):
            self.discovered_clues.add(str(clue["id"]))
        self.evidence_count += 1
        return self.summary()

    @staticmethod
    def _unit_vector(origin: Coordinate, target: Coordinate) -> tuple[float, float]:
        dx, dy = target[0] - origin[0], target[1] - origin[1]
        distance = math.hypot(dx, dy)
        return (dx / distance, dy / distance) if distance else (0.0, 0.0)

    @classmethod
    def _expected_vector(cls, origin: Coordinate, target: Coordinate) -> tuple[float, float]:
        ux, uy = cls._unit_vector(origin, target)
        strength = 1.0 / (1.0 + math.dist(origin, target))
        return ux * strength, uy * strength

    def summary(self, top_n: int = 5) -> dict[str, Any]:
        ranked = sorted(self.probabilities.items(), key=lambda item: item[1], reverse=True)
        entropy = -sum(probability * math.log(probability) for probability in self.probabilities.values() if probability)
        return {
            "estimate": list(ranked[0][0]),
            "confidence": ranked[0][1],
            "entropy": entropy,
            "evidence_count": self.evidence_count,
            "discovered_clues": sorted(self.discovered_clues),
            "top_candidates": [
                {"position": list(cell), "probability": probability}
                for cell, probability in ranked[:top_n]
            ],
        }

    def probability_at(self, position: Coordinate) -> float:
        return self.probabilities.get(position, 0.0)
