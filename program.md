# Orderbook Market Making

Evolve `strategy.py` — a subclass of `BaseStrategy` — to maximize **mean participant edge** on a FIFO binary-contract prediction-market simulator.

## Task Overview

The simulator runs 200 independent simulations, each 2000 steps long. In every simulation your strategy manages passive limit orders on a single shared FIFO order book for a binary `YES` contract. The contract settles to `1.0` or `0.0` depending on whether a hidden latent Gaussian score process ends above zero.

Your score is the mean `total_edge` across all successful simulations. Edge is earned each time one of your resting limit orders is filled:

- you buy `q` at price `x` ticks: `edge = q * (p_t - x/100)`
- you sell `q` at price `x` ticks: `edge = q * (x/100 - p_t)`

where `p_t` is the true probability at the moment of the fill.

**Higher mean edge is better.**

## Market Mechanics

### Instrument

A single `YES` share. Prices are integer percentage ticks `{1, ..., 99}` (economic price = ticks / 100). Quantities are rounded to the nearest `0.01`.

### Latent Process

```
Z_{t+1} = Z_t + sigma * epsilon_t + sum_{k=1}^{N_t} J_{t,k}
epsilon_t ~ N(0, 1)
N_t ~ Poisson(lambda_jump)
J_{t,k} ~ N(mu_jump, sigma_jump^2)
```

Defaults: `n_steps=2000`, `sigma=0.02`, `lambda_jump=0.001`, `mu_jump=0.0`, `sigma_jump=0.75`.

The informed fair value at step `t` is `p_t = Pr(Z_T > 0 | Z_t)`.

### Book Mechanics

Standard price-time (FIFO) priority. Your orders are passive (limit only). The book also contains a static hidden competitor ladder of resting orders centred near the starting probability with a small spread; the competitor's orders never recenter but replenish on the opposite side when fully filled.

### Event Loop (per step)

1. Refresh competitor replenishments from the previous step.
2. Call `strategy.on_step(state)` with current private state and previous-step fill totals.
3. Apply your cancels and new passive limit orders.
4. Advance the latent process; compute `p_t`.
5. The informed arbitrageur sweeps stale quotes: buys every ask below `p_t`, sells into every bid above `p_t`.
6. Uninformed retail market orders arrive and execute against the book.
7. Record participant fills, edge, and competitor replenishments.

### Retail Flow

Arrivals per step: `Poisson(0.8)`. Side: `Bernoulli(0.5)`. Notional: `LogNormal`. These are exogenous to the strategy.

### Collateral Rules

- Starts with `$1000`, zero YES inventory, zero NO inventory.
- Resting bids reserve `price * quantity` cash.
- Uncovered resting asks reserve `(1 - price) * quantity` cash.
- Minting: 1 YES + 1 NO costs `$1` (available implicitly via the engine).
- Invalid actions (over-collateral, bad ticks) raise an exception and **fail** the simulation. Failures are excluded from scoring.

## Setup

```bash
bash prepare.sh
```

This installs the vendored `orderbook_pm_challenge` package in editable mode so the `python -m orderbook_pm_challenge` CLI is available.

Run the baseline eval:

```bash
bash eval/eval.sh
```

## Evaluation

```bash
bash eval/eval.sh
```

At launch, the eval samples **three independent seed-starts** uniformly from `[0, 2**31)` via `secrets.randbelow`. For each seed-start, the simulator runs **200 simulations × 2000 steps** with `--workers 4`, using the built-in per-simulation hyperparameter sampling. Results are aggregated across all 600 simulations into a single `mean_edge`. The sampled seed-starts are echoed to stdout before the scoring block for traceability.

Output:

```
# seed-starts: [<s1>, <s2>, <s3>]
---
mean_edge:        <float with 6 decimals>
correct:          <success_count>
total:            <simulation_count>
```

Parse the score with: `grep "^mean_edge:" run.log`

**Reproducibility note.** Because the seed-starts are re-sampled on every invocation, `mean_edge` will fluctuate run-to-run even when `strategy.py` is unchanged. Treat score differences smaller than the run-to-run noise floor (empirically a few tenths of an edge unit for the starter) as statistically insignificant. Optimise for the distribution, not for a lucky seed.

## Alignment with the public leaderboard

The public leaderboard at `optimizationarena.com/prediction-market-challenge` scores strategies on 200 simulations at 2,000 steps with per-simulation hyperparameter sampling, and the hackathon was "rescored with a fresh random seed". This task's local eval follows the same simulator and the same scoring formula, with three deliberate choices:

1. **Three random seed-starts per eval, sampled at runtime.** We run 3 × 200 = 600 simulations per eval. Each seed-start is drawn fresh from `secrets.randbelow(2**31)` each time `eval/eval.sh` runs, mirroring the leaderboard's "fresh random seed" rescoring and discouraging local overfitting to a specific seed regime.
2. **Parameter variance.** The website summarises four varying hyperparameters (jump intensity, jump variance, retail arrival rate, competitor spread). Our vendored `ParameterVariance` samples eight — those four plus `initial_score`, `jump_mean`, `retail_mean_notional`, `competitor_quote_notional`. This follows the canonical simulator code as shipped.
3. **Failed simulations are excluded from the mean.** `mean_edge` averages `total_edge` across successful simulations only. A strategy that raises uncaught exceptions has its failed sims dropped from the denominator rather than penalised as zero. Keep your strategy exception-safe.

## Strategy API

The simulator calls `strategy.on_step(state)` once per step. Return a list of zero or more actions.

### `StepState` fields

| Field | Type | Description |
|---|---|---|
| `step` | `int` | Current step index (0-based) |
| `steps_remaining` | `int` | Steps left in the simulation |
| `yes_inventory` | `float` | Current YES share inventory |
| `no_inventory` | `float` | Current NO share inventory |
| `cash` | `float` | Total cash (including reserved) |
| `reserved_cash` | `float` | Cash locked by resting bids |
| `free_cash` | `float` | Cash available for new orders (`cash - reserved_cash`) |
| `competitor_best_bid_ticks` | `int \| None` | Best resting competitor bid in ticks (None if absent) |
| `competitor_best_ask_ticks` | `int \| None` | Best resting competitor ask in ticks (None if absent) |
| `buy_filled_quantity` | `float` | Aggregate filled quantity on your resting bids in the previous step |
| `sell_filled_quantity` | `float` | Aggregate filled quantity on your resting asks in the previous step |
| `own_orders` | `tuple[OwnOrderView, ...]` | Your currently resting orders |

Each `OwnOrderView` has: `order_id`, `side`, `price_ticks`, `remaining_quantity`, `submitted_step`.

### Available actions

```python
from orderbook_pm_challenge.types import PlaceOrder, CancelOrder, CancelAll, Side

PlaceOrder(side=Side.BUY, price_ticks=48, quantity=5.0)
PlaceOrder(side=Side.SELL, price_ticks=52, quantity=5.0, client_order_id="optional")
CancelOrder(order_id="<uuid>")
CancelAll()
```

### BaseStrategy

```python
from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import StepState

class Strategy(BaseStrategy):
    def on_step(self, state: StepState) -> list:
        ...
```

## Modifiable Files

- `strategy.py` — the only file the agent may edit.

## Off-Limits

- `eval/` — evaluation harness, must not be modified.
- `prepare.sh` — install script, must not be modified.
- `orderbook_pm_challenge/` — vendored simulator package, must not be modified.
- `pyproject.toml` — package metadata, must not be modified.

## Output Format

The eval script always ends with:

```
---
mean_edge:        <float, 6 decimal places>
correct:          <int, successful simulations>
total:            <int, total simulations>
```

## Simplicity Criterion

The simplest strategy that reliably improves `mean_edge` above the starter baseline is preferred over one that achieves a marginally higher score through complexity that is brittle across seeds.
