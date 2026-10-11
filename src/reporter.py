"""Mission report generation in Markdown and JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .agent import MissionResult


def write_reports(result: MissionResult, output_dir: str | Path) -> tuple[Path, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "mission_report.json"
    markdown_path = directory / "mission_report.md"
    data = result.to_dict()

    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    steps = "\n".join(f"- {step}" for step in result.next_steps)
    clues = ", ".join(result.discovered_clues) if result.discovered_clues else "None discovered"
    markdown = f"""# {result.mission} — Mission Report

## Outcome

- **Status:** `{result.status}`
- **Inferred EVE location:** `{tuple(result.inferred_location)}`
- **Final confidence:** {result.confidence:.1%}
- **Evidence updates:** {result.evidence_count}
- **Clues discovered:** {clues}

## Resource Summary

- **Actions taken:** {result.actions_taken}
- **Elapsed simulation time:** {result.elapsed_time}
- **Battery remaining:** {result.battery_remaining}

## Recommended Next Steps

{steps}
"""
    markdown_path.write_text(markdown, encoding="utf-8")
    return markdown_path, json_path


def report_summary(result: MissionResult) -> dict[str, Any]:
    """Return a compact summary suitable for embedding in the action trace."""
    return result.to_dict()
