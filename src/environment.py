"""Hidden-state grid simulation and noisy signal observations."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import yaml


Coordinate = tuple[int, int]


class EnvironmentError(RuntimeError):
    """Raised when a mission configuration or action is invalid."""


class Environment:
    """A deterministic (seeded) simulator that keeps mission secrets private."""

    def __init__(self, config: dict[str, Any]) -> None:
        mission = config.get("mission")
        if not isinstance(mission, dict):
            raise EnvironmentError("Configuration must contain a 'mission' mapping.")

        self.name = str(mission.get("name", "Unnamed mission"))
        self.max_actions = self._positive_int(mission.get("max_actions"), "max_actions")
        grid = mission.get("grid", {})
        self.width = self._positive_int(grid.get("width"), "grid.width")
        self.height = self._positive_int(grid.get("height"), "grid.height")
        self.obstacles = {self._coordinate(value, "grid.obstacles") for value in grid.get("obstacles", [])}

        robot = mission.get("robot", {})
        self.start = self._coordinate(robot.get("start"), "robot.start")
        self.capacity = self._positive_int(robot.get("battery_capacity"), "robot.battery_capacity")
        self.initial_battery = self._positive_int(robot.get("initial_battery"), "robot.initial_battery")
        if self.initial_battery > self.capacity:
            raise EnvironmentError("robot.initial_battery cannot exceed battery_capacity.")

        target = mission.get("hidden_target", {})
        self._target = self._coordinate(target.get("position"), "hidden_target.position")
        self._clues = []
        for clue in mission.get("hidden_clues", []):
            self._clues.append(
                {
                    "id": str(clue["id"]),
                    "position": self._coordinate(clue["position"], "hidden_clues.position"),
                    "confidence": float(clue.get("confidence", 0.9)),
                }
            )

        self.stations = {self._coordinate(value, "recharge_stations") for value in mission.get("recharge_stations", [])}
        costs = mission.get("costs", {})
        self.costs = {
            "move": self._positive_int(costs.get("move"), "costs.move"),
            "scan": self._positive_int(costs.get("scan"), "costs.scan"),
            "recharge": self._positive_int(costs.get("recharge"), "costs.recharge"),
            "recharge_amount": self._positive_int(costs.get("recharge_amount"), "costs.recharge_amount"),
            "action_time": self._positive_int(costs.get("action_time"), "costs.action_time"),
        }
        sensors = mission.get("sensors", {})
        self.signal_noise_sigma = self._positive_float(
            sensors.get("signal_noise_sigma", 0.13), "sensors.signal_noise_sigma"
        )
        self.scan_noise_sigma = self._positive_float(
            sensors.get("scan_noise_sigma", 0.055), "sensors.scan_noise_sigma"
        )
        self.clue_radius = self._positive_float(
            sensors.get("clue_discovery_radius", 1.5), "sensors.clue_discovery_radius"
        )
        self.target_detection_radius = self._positive_float(
            sensors.get("target_detection_radius", 0.75), "sensors.target_detection_radius"
        )
        self.detection_radius = self._positive_float(
            sensors.get("detection_radius", 1.0), "sensors.detection_radius"
        )

        for point, label in [(self.start, "start"), (self._target, "hidden target")]:
            self._validate_open_cell(point, label)
        for point in self.stations:
            self._validate_open_cell(point, "recharge station")
        for clue in self._clues:
            self._validate_open_cell(clue["position"], "hidden clue")
            if not 0 < clue["confidence"] <= 1:
                raise EnvironmentError("Clue confidence must be in the interval (0, 1].")

        self._rng = random.Random(int(mission.get("seed", 0)))
        self.position = self.start
        self.battery = self.initial_battery
        self.elapsed_time = 0
        self.action_count = 0
        self._discovered_clues: set[str] = set()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Environment":
        config_path = Path(path)
        try:
            with config_path.open("r", encoding="utf-8") as stream:
                config = yaml.safe_load(stream)
        except (OSError, yaml.YAMLError) as exc:
            raise EnvironmentError(f"Unable to load mission configuration '{config_path}': {exc}") from exc
        if not isinstance(config, dict):
            raise EnvironmentError("Mission configuration must be a YAML mapping.")
        return cls(config)

    @staticmethod
    def _coordinate(value: Any, label: str) -> Coordinate:
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 2
            or any(not isinstance(part, int) or isinstance(part, bool) for part in value)
        ):
            raise EnvironmentError(f"{label} must be a pair of integer coordinates.")
        return int(value[0]), int(value[1])

    @staticmethod
    def _positive_int(value: Any, label: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise EnvironmentError(f"{label} must be a positive integer.")
        return value

    @staticmethod
    def _positive_float(value: Any, label: str) -> float:
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise EnvironmentError(f"{label} must be a positive number.")
        return float(value)

    def _validate_open_cell(self, point: Coordinate, label: str) -> None:
        x, y = point
        if not (0 <= x < self.width and 0 <= y < self.height) or point in self.obstacles:
            raise EnvironmentError(f"{label} {point} is outside the grid or blocked.")

    def public_info(self) -> dict[str, Any]:
        """Return planning metadata without exposing target or clue placements."""
        return {
            "mission_name": self.name,
            "width": self.width,
            "height": self.height,
            "obstacles": sorted(self.obstacles),
            "start": self.start,
            "battery_capacity": self.capacity,
            "initial_battery": self.initial_battery,
            "recharge_stations": sorted(self.stations),
            "costs": dict(self.costs),
            "max_actions": self.max_actions,
            "detection_radius": self.detection_radius,
        }

    def reset(self) -> dict[str, Any]:
        """Reset transient simulator state and return an initial sensor reading."""
        self.position = self.start
        self.battery = self.initial_battery
        self.elapsed_time = 0
        self.action_count = 0
        self._discovered_clues.clear()
        return self._observe("initialize", "initial", scan_quality=False)

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        """Execute one action, charging resources and returning a structured observation."""
        if self.action_count >= self.max_actions:
            raise EnvironmentError("Mission action limit has been reached.")
        if not isinstance(action, dict) or action.get("type") not in {"move", "scan", "recharge"}:
            raise EnvironmentError("Action type must be 'move', 'scan', or 'recharge'.")
        kind = action["type"]
        cost = self.costs[kind]
        if kind == "recharge":
            if self.position not in self.stations:
                raise EnvironmentError("Cannot recharge away from a recharge station.")
            if self.battery >= self.capacity:
                raise EnvironmentError("Battery is already full.")
            self.battery = min(self.capacity, self.battery + self.costs["recharge_amount"])
        else:
            if self.battery < cost:
                raise EnvironmentError(f"Insufficient battery for {kind}: need {cost}, have {self.battery}.")
            self.battery -= cost
            if kind == "move":
                destination = self._coordinate(action.get("to"), "action.to")
                if abs(destination[0] - self.position[0]) + abs(destination[1] - self.position[1]) != 1:
                    raise EnvironmentError("Moves must go exactly one grid cell in a cardinal direction.")
                self._validate_open_cell(destination, "move destination")
                self.position = destination

        self.elapsed_time += self.costs["action_time"]
        self.action_count += 1
        return self._observe(kind, str(action.get("reason", "")), scan_quality=kind == "scan")

    def _observe(self, action_type: str, reason: str, scan_quality: bool) -> dict[str, Any]:
        dx = self._target[0] - self.position[0]
        dy = self._target[1] - self.position[1]
        distance = math.hypot(dx, dy)
        true_strength = 1.0 / (1.0 + distance)
        sigma = self.scan_noise_sigma if scan_quality else self.signal_noise_sigma
        if distance:
            ux, uy = dx / distance, dy / distance
        else:
            ux, uy = 0.0, 0.0
        vector = {
            "x": ux * true_strength + self._rng.gauss(0.0, sigma),
            "y": uy * true_strength + self._rng.gauss(0.0, sigma),
        }
        strength = max(0.0, true_strength + self._rng.gauss(0.0, sigma / 2))

        discovered = []
        for clue in self._clues:
            if clue["id"] in self._discovered_clues:
                continue
            if math.dist(self.position, clue["position"]) <= self.clue_radius:
                self._discovered_clues.add(clue["id"])
                clue_dx = self._target[0] - clue["position"][0]
                clue_dy = self._target[1] - clue["position"][1]
                clue_distance = math.hypot(clue_dx, clue_dy)
                discovered.append(
                    {
                        "id": clue["id"],
                        "origin": list(clue["position"]),
                        "bearing": {
                            "x": clue_dx / clue_distance if clue_distance else 0.0,
                            "y": clue_dy / clue_distance if clue_distance else 0.0,
                        },
                        "confidence": clue["confidence"],
                    }
                )

        return {
            "action": action_type,
            "reason": reason,
            "position": list(self.position),
            "signal_vector": vector,
            "signal_strength": strength,
            "signal_quality": "scan" if scan_quality else "ambient",
            "discovered_clues": discovered,
            "battery_remaining": self.battery,
            "timestamp": self.elapsed_time,
            "action_count": self.action_count,
            "target_detected": distance <= self.target_detection_radius,
        }
