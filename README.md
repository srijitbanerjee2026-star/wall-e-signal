# WALL-E and EVE's Signal

A deterministic, configurable grid-search simulation. WALL-E receives noisy
directional sensor readings and discovers signal fragments through exploration;
the simulator keeps EVE's location and undiscovered clue placements out of the
agent's public mission data. A Bayesian grid filter estimates the target while
a heuristic planner balances scans, exploration, and safe battery returns.

The planner is deliberately heuristic/POMDP-style and has no external LLM
dependency. The environment's random seed makes a mission reproducible.

## Requirements

- Python 3.10 or newer
- Dependencies in `requirements.txt`

## Run a mission

From this directory:

```powershell
python -m pip install -r requirements.txt
python main.py
```

Use a different mission file or output directory with:

```powershell
python main.py --config config/default_mission.yaml --output-dir outputs
```

## Mission configuration

`config/default_mission.yaml` describes the grid, obstacles, initial battery,
action costs, recharge stations, sensor noise, and simulator-private target and
clue placements. `Environment.public_info()` intentionally omits all hidden
coordinates before passing mission metadata to the agent. Coordinates are
`[x, y]`, with `(0, 0)` at the top-left.

Every action returns an observation containing WALL-E's coordinates, a noisy
signal vector and strength, any newly discovered directional clues, remaining
battery, and elapsed simulation time. Actions are rejected with explicit
errors when invalid or unaffordable. Signal and clue measurements update the
Bayesian posterior over traversable cells.

## Outputs

- `outputs/sample_trace.json` — ordered actions, observations, belief updates,
  planner reasoning, and final result.
- `outputs/mission_report.md` — human-readable mission outcome and next steps.
- `outputs/mission_report.json` — machine-readable final mission report.
