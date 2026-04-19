#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python - <<'PY'
import json
import secrets
import subprocess

# Sample three independent seed-starts at runtime — no hardcoded seeds.
# This mirrors the public leaderboard's "fresh random seed" rescoring.
seeds = [secrets.randbelow(2**31) for _ in range(3)]
print(f"# seed-starts: {seeds}")

all_sims = []
for seed in seeds:
    out_path = f"/tmp/obpm_out_{seed}.json"
    with open(out_path, "w") as f:
        subprocess.run(
            [
                "python", "-m", "orderbook_pm_challenge", "run", "strategy.py",
                "--simulations", "200",
                "--seed-start", str(seed),
                "--workers", "4",
                "--json",
            ],
            stdout=f,
            check=True,
        )
    with open(out_path) as f:
        all_sims.extend(json.load(f)["simulation_results"])

successes = [s for s in all_sims if not s["failed"]]
mean_edge = sum(s["total_edge"] for s in successes) / len(successes) if successes else 0.0

print("---")
print(f"mean_edge:        {mean_edge:.6f}")
print(f"correct:          {len(successes)}")
print(f"total:            {len(all_sims)}")
PY
