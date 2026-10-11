"""Resource-aware autonomous planning and mission execution."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

from .belief_state import BeliefState
from .environment import Environment
from .logger import MissionLogger


Coordinate = tuple[int, int]


@dataclass
class MissionResult:
    mission: str
    status: str
    inferred_location: list[int]
    confidence: float
    evidence_count: int
    discovered_clues: list[str]
    actions_taken: int
    elapsed_time: int
    battery_remaining: int
    next_steps: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Agent:
    """Operate WALL-E without access to the simulator's hidden target or clues."""

    def __init__(self, environment: Environment, logger: MissionLogger | None = None) -> None:
        self.environment = environment
        self.info = environment.public_info()
        self.logger = logger or MissionLogger(self.info["mission_name"])
        self.belief = BeliefState(
            self.info["width"],
            self.info["height"],
            self.info["obstacles"],
            environment.signal_noise_sigma,
            environment.scan_noise_sigma,
        )
        self.scanned: set[Coordinate] = set()
        self.explored: set[Coordinate] = set()
        self._last_observation: dict[str, Any] | None = None
        self._aborted_reason: str | None = None

    def run(self) -> MissionResult:
        observation = self.environment.reset()
        self._last_observation = observation
        self.explored.add(tuple(observation["position"]))
        initial_belief = self.belief.update(observation)
        self.logger.record_initial(observation, initial_belief)

        while self.environment.action_count < self.info["max_actions"]:
            if observation["target_detected"] and observation["action"] == "scan":
                break
            decision = self._decide(observation)
            if decision is None:
                break
            action, reasoning = decision
            observation = self.environment.step(action)
            self._last_observation = observation
            position = tuple(observation["position"])
            self.explored.add(position)
            if action["type"] == "scan":
                self.scanned.add(position)
            belief_summary = self.belief.update(observation)
            self.logger.record_action(action, observation, belief_summary, reasoning)
            if observation["target_detected"] and action["type"] == "scan":
                break

        return self._result(observation)

    def _decide(self, observation: dict[str, Any]) -> tuple[dict[str, Any], str] | None:
        position = tuple(observation["position"])
        battery = int(observation["battery_remaining"])
        costs = self.info["costs"]
        stations = self.info["recharge_stations"]
        reachable_stations = [
            (station, self._path(position, station))
            for station in stations
            if self._path(position, station) is not None
        ]
        if not reachable_stations:
            self._aborted_reason = "No recharge station is reachable from WALL-E's current position."
            return None
        station, path_to_station = min(reachable_stations, key=lambda item: len(item[1] or []))
        return_reserve = (len(path_to_station or []) + 1) * costs["move"]
        maintenance_threshold = return_reserve + costs["scan"]

        if position in stations and battery < self.info["battery_capacity"] and battery <= maintenance_threshold:
            action = {"type": "recharge", "reason": "Restore energy at a known recharge station."}
            return action, f"Battery is {battery}; recharge now to preserve a safe exploration reserve."

        if battery <= maintenance_threshold:
            if position == station:
                if battery < self.info["battery_capacity"]:
                    return (
                        {"type": "recharge", "reason": "Restore energy before continuing the search."},
                        "Recharge before the remaining energy falls below the safe return reserve.",
                    )
            elif len(path_to_station or []) * costs["move"] <= battery:
                next_position = path_to_station[1]
                return (
                    {"type": "move", "to": list(next_position), "reason": "Return to recharge station"},
                    f"Battery is {battery}; move toward the nearest station at {list(station)} before resuming search.",
                )
            else:
                self._aborted_reason = "Battery reserve is insufficient to reach a recharge station safely."
                return None

        if position not in self.scanned and battery >= costs["scan"] + return_reserve:
            return (
                {"type": "scan", "reason": "Collect a higher-quality local signal measurement."},
                "This location has not been scanned; spend a scan action while retaining the return reserve.",
            )

        target = self._select_search_goal(position)
        path = self._path(position, target) if target is not None else None
        if path and len(path) > 1:
            destination = path[1]
            return (
                {"type": "move", "to": list(destination), "reason": "Explore a promising signal region"},
                (
                    f"Move toward posterior-favored cell {list(target)} using the shortest safe path; "
                    f"current peak belief is {self.belief.summary()['confidence']:.3f}."
                ),
            )

        if position in stations and battery < self.info["battery_capacity"]:
            return (
                {"type": "recharge", "reason": "Recharge before further exploration."},
                "Recharge at the current station before continuing the search.",
            )
        self._aborted_reason = "No safe unscanned search goal is available."
        return None

    def _select_search_goal(self, current: Coordinate) -> Coordinate | None:
        candidates: list[tuple[float, Coordinate]] = []
        for cell in self.belief.cells:
            path = self._path(current, cell)
            if path is None:
                continue
            distance = len(path) - 1
            if distance == 0 and cell in self.scanned:
                continue
            probability = self.belief.probability_at(cell)
            novelty = 1.0 if cell not in self.explored else 0.0
            score = probability + 0.12 * novelty / (1.0 + 0.15 * distance)
            candidates.append((score, cell))
        return max(candidates, default=(0.0, None), key=lambda item: (item[0], -item[1][1], -item[1][0]))[1]

    def _path(self, start: Coordinate, goal: Coordinate) -> list[Coordinate] | None:
        blocked = set(self.info["obstacles"])
        if start in blocked or goal in blocked:
            return None
        queue = deque([start])
        previous: dict[Coordinate, Coordinate | None] = {start: None}
        while queue:
            current = queue.popleft()
            if current == goal:
                path = []
                cursor: Coordinate | None = current
                while cursor is not None:
                    path.append(cursor)
                    cursor = previous[cursor]
                return list(reversed(path))
            x, y = current
            for neighbor in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if (
                    0 <= neighbor[0] < self.info["width"]
                    and 0 <= neighbor[1] < self.info["height"]
                    and neighbor not in blocked
                    and neighbor not in previous
                ):
                    previous[neighbor] = current
                    queue.append(neighbor)
        return None

    def _result(self, observation: dict[str, Any]) -> MissionResult:
        summary = self.belief.summary()
        detected = bool(observation.get("target_detected")) and observation.get("action") == "scan"
        if detected:
            status = "target_detected"
            next_steps = ["Transmit EVE's confirmed coordinates and request mission-control acknowledgement."]
        elif self._aborted_reason:
            status = "aborted"
            next_steps = [self._aborted_reason, "Dispatch a recovery unit with additional battery capacity."]
        elif self.environment.action_count >= self.info["max_actions"]:
            status = "action_limit_reached"
            next_steps = ["Provide more action budget and continue searching from the highest-probability region."]
        else:
            status = "signal_localized"
            next_steps = ["Verify the inferred location with a close-range scan."]
        return MissionResult(
            mission=self.info["mission_name"],
            status=status,
            inferred_location=summary["estimate"],
            confidence=summary["confidence"],
            evidence_count=summary["evidence_count"],
            discovered_clues=summary["discovered_clues"],
            actions_taken=self.environment.action_count,
            elapsed_time=self.environment.elapsed_time,
            battery_remaining=self.environment.battery,
            next_steps=next_steps,
        )
