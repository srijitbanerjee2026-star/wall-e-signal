"""Structured action-trace collection and JSON serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MissionLogger:
    """Collect an ordered, JSON-serializable record of the mission."""

    def __init__(self, mission_name: str) -> None:
        self.mission_name = mission_name
        self.events: list[dict[str, Any]] = []

    def record_initial(self, observation: dict[str, Any], belief: dict[str, Any]) -> None:
        self.events.append(
            {
                "sequence": 0,
                "event": "initial_observation",
                "observation": observation,
                "belief": belief,
                "reasoning": "Initialize the belief from WALL-E's first noisy signal reading.",
            }
        )

    def record_action(
        self,
        action: dict[str, Any],
        observation: dict[str, Any],
        belief: dict[str, Any],
        reasoning: str,
    ) -> None:
        self.events.append(
            {
                "sequence": len(self.events),
                "event": "action",
                "action": action,
                "observation": observation,
                "belief": belief,
                "reasoning": reasoning,
            }
        )

    def to_dict(self, result: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"mission": self.mission_name, "events": self.events, "result": result}

    def write_json(self, path: str | Path, result: dict[str, Any] | None = None) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(result), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
