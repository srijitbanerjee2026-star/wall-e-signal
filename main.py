"""Run the default WALL-E signal-search mission."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.agent import Agent
from src.environment import Environment
from src.logger import MissionLogger
from src.reporter import report_summary, write_reports


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Run WALL-E and EVE's Signal simulation.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "default_mission.yaml",
        help="Path to a YAML mission configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Directory for the action trace and mission reports.",
    )
    args = parser.parse_args()

    environment = Environment.from_yaml(args.config)
    logger = MissionLogger(environment.name)
    result = Agent(environment, logger).run()
    logger.write_json(args.output_dir / "sample_trace.json", report_summary(result))
    markdown_path, json_path = write_reports(result, args.output_dir)
    print(f"Mission status: {result.status}")
    print(f"Inferred location: {tuple(result.inferred_location)} ({result.confidence:.1%} confidence)")
    print(f"Trace: {args.output_dir / 'sample_trace.json'}")
    print(f"Report: {markdown_path}")
    print(f"JSON report: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
