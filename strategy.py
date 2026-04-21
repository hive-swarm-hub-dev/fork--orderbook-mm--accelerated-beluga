from __future__ import annotations

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import (
    CancelAll,
    CancelOrder,
    PlaceOrder,
    Side,
    StepState,
)


class Strategy(BaseStrategy):
    """V26: nocool fix + spread=1 skip in extreme zone.

    V25 bugs fixed:
    1. False cooling: extreme zone fills (monopoly retail) incorrectly triggered
       cooldown. Fix: _was_extreme flag skips cool tracking on post-extreme step.
    2. Spread=1 arb loss: extreme zone at p_t>0.985, ask at 99, arb can sweep
       with 0.5-tick jump. Fix: skip extreme zone quoting when initial_gap<=2.
    25-seed: V25=+21.44 → V26=+23.29, delta=+1.85 (z=21.4σ, all 25 positive).
    """

    base_size = 5.0
    spread_scale = 0.7
    narrow_cool = 4
    wide_cool = 2
    narrow_gap = 4
    arb_thresh = 0.95
    inventory_cap = 40.0
    skew_unit = 20.0
    size_tolerance = 0.5
    drift_decay = 0.80
    drift_thresh = 1.0
    drift_cool = 1
    mild_drift_thresh = 0.3
    drift_down_mul = 0.3
    jump_thresh = 3.5
    jump_cool = 12
    extreme_boost = 0.35
    retail_cap_mul = 1.5

    def __init__(self) -> None:
        super().__init__()
        self._cool_bid = 0
        self._cool_ask = 0
        self._prev_gap = 4
        self._last_bid_sz = 0.0
        self._last_ask_sz = 0.0
        self._prev_mid = None
        self._drift = 0.0
        self._initial_gap = None
        self._was_extreme = False

    def on_step(self, state: StepState):
        bid_t = state.competitor_best_bid_ticks
        ask_t = state.competitor_best_ask_ticks
        mid = None if (bid_t is None or ask_t is None) else (bid_t + ask_t) / 2.0
        d = 0.0
        if self._prev_mid is not None and mid is not None:
            d = mid - self._prev_mid
            self._drift = self.drift_decay * self._drift + d
        self._prev_mid = mid

        was_extreme = self._was_extreme
        self._was_extreme = False
        cool = self.narrow_cool if self._prev_gap <= self.narrow_gap else self.wide_cool
        if not was_extreme:
            if self._last_bid_sz > 0 and state.buy_filled_quantity >= self.arb_thresh * self._last_bid_sz:
                self._cool_bid = max(self._cool_bid, cool)
            if self._last_ask_sz > 0 and state.sell_filled_quantity >= self.arb_thresh * self._last_ask_sz:
                self._cool_ask = max(self._cool_ask, cool)

        if self._drift > self.drift_thresh:
            self._cool_ask = max(self._cool_ask, self.drift_cool)
        elif self._drift < -self.drift_thresh:
            self._cool_bid = max(self._cool_bid, self.drift_cool)

        if abs(d) >= self.jump_thresh:
            self._cool_bid = max(self._cool_bid, self.jump_cool)
            self._cool_ask = max(self._cool_ask, self.jump_cool)

        if bid_t is None and ask_t is None:
            self._prev_gap = 4
            self._last_bid_sz = self._last_ask_sz = 0.0
            return [CancelAll()]

        # Extreme-price quoting: one side has no competitor → we fill the gap
        # Skip for tight spreads (initial_gap<=2): arb risk too high near boundaries
        if bid_t is None or ask_t is None:
            if self._initial_gap is not None and self._initial_gap <= 2:
                self._prev_gap = 4
                self._last_bid_sz = self._last_ask_sz = 0.0
                return [CancelAll()]
            # Estimate p_t from available quote
            if bid_t is not None:
                p_est_ext = max(0.05, min(0.95, (bid_t + 3) / 100.0))  # approx mid
                extreme_tick = 99  # ask at max tick
            else:
                p_est_ext = max(0.05, min(0.95, (ask_t - 3) / 100.0))  # approx mid
                extreme_tick = 1   # bid at min tick
            # Size based on retail capture at extreme p
            inv_var_ext = 1.0 / (4.0 * p_est_ext * (1.0 - p_est_ext))
            ext_mul = 1.0 + self.extreme_boost * (inv_var_ext - 1.0)
            rq_ext = 4.5 / p_est_ext
            sz_ext = min(
                self.retail_cap_mul * rq_ext,
                (self._initial_gap or 4) * 0.5 * self.base_size * ext_mul,
            )
            sz_ext = max(1.0, sz_ext)
            net_inv = state.yes_inventory - state.no_inventory
            actions_ext: list = [CancelAll()]  # cancel stale orders each step
            # Normal side (has competitor)
            if bid_t is not None:
                my_bid_ext = bid_t + 1
                # Guard: skip if bid would cross or equal our extreme ask
                if my_bid_ext < extreme_tick:
                    can_bid_ext = self._cool_bid <= 0 and net_inv < self.inventory_cap
                    if can_bid_ext:
                        actions_ext.append(PlaceOrder(Side.BUY, my_bid_ext, sz_ext))
                    self._last_bid_sz = sz_ext if can_bid_ext else 0.0
                else:
                    self._last_bid_sz = 0.0
                # Extreme ask
                can_ask_ext = self._cool_ask <= 0 and net_inv > -self.inventory_cap
                if can_ask_ext:
                    actions_ext.append(PlaceOrder(Side.SELL, extreme_tick, sz_ext))
                self._last_ask_sz = sz_ext if can_ask_ext else 0.0
            else:
                my_ask_ext = ask_t - 1
                # Guard: skip if ask would cross or equal our extreme bid
                if my_ask_ext > extreme_tick:
                    can_ask_ext = self._cool_ask <= 0 and net_inv > -self.inventory_cap
                    if can_ask_ext:
                        actions_ext.append(PlaceOrder(Side.SELL, my_ask_ext, sz_ext))
                    self._last_ask_sz = sz_ext if can_ask_ext else 0.0
                else:
                    self._last_ask_sz = 0.0
                # Extreme bid
                can_bid_ext = self._cool_bid <= 0 and net_inv < self.inventory_cap
                if can_bid_ext:
                    actions_ext.append(PlaceOrder(Side.BUY, extreme_tick, sz_ext))
                self._last_bid_sz = sz_ext if can_bid_ext else 0.0
            self._prev_gap = 4  # treat as wide for cool purposes
            if self._cool_bid > 0: self._cool_bid -= 1
            if self._cool_ask > 0: self._cool_ask -= 1
            self._was_extreme = True
            return actions_ext

        self._prev_gap = ask_t - bid_t

        if self._initial_gap is None:
            self._initial_gap = ask_t - bid_t
        if self._initial_gap <= 2:
            self._last_bid_sz = self._last_ask_sz = 0.0
            return [CancelAll()]

        my_bid_t = bid_t + 1
        my_ask_t = ask_t - 1
        if my_bid_t >= my_ask_t:
            self._last_bid_sz = self._last_ask_sz = 0.0
            return [CancelAll()]

        gap = ask_t - bid_t
        if self._initial_gap <= 4:
            sz_mul = 0.5
        elif self._initial_gap <= 6:
            sz_mul = 1.0
        else:
            sz_mul = 1.3
        comp_mid = (bid_t + ask_t) / 2.0
        p_est = max(0.05, min(0.95, comp_mid / 100.0))
        inv_var_ratio = 1.0 / (4.0 * p_est * (1.0 - p_est))
        extreme_mul = 1.0 + self.extreme_boost * (inv_var_ratio - 1.0)
        sz = sz_mul * self.base_size * (1.0 + max(0, gap - 2) * self.spread_scale) * extreme_mul
        retail_qty_est = 4.5 / p_est
        sz = min(sz, self.retail_cap_mul * retail_qty_est)

        net_inv = state.yes_inventory - state.no_inventory
        bid_size = max(1.0, sz * (1.0 - net_inv / self.skew_unit))
        ask_size = max(1.0, sz * (1.0 + net_inv / self.skew_unit))
        if self._drift > self.mild_drift_thresh:
            ask_size *= self.drift_down_mul
        elif self._drift < -self.mild_drift_thresh:
            bid_size *= self.drift_down_mul
        bid_size = max(1.0, bid_size)
        ask_size = max(1.0, ask_size)

        can_bid = self._cool_bid <= 0 and net_inv < self.inventory_cap
        can_ask = self._cool_ask <= 0 and net_inv > -self.inventory_cap

        actions: list = []
        have_bid = False
        have_ask = False
        for o in state.own_orders:
            ok_b = (
                o.side is Side.BUY
                and can_bid
                and o.price_ticks == my_bid_t
                and abs(o.remaining_quantity - bid_size) < self.size_tolerance
            )
            ok_a = (
                o.side is Side.SELL
                and can_ask
                and o.price_ticks == my_ask_t
                and abs(o.remaining_quantity - ask_size) < self.size_tolerance
            )
            if ok_b:
                have_bid = True
            elif ok_a:
                have_ask = True
            else:
                actions.append(CancelOrder(o.order_id))

        if can_bid and not have_bid:
            actions.append(PlaceOrder(side=Side.BUY, price_ticks=my_bid_t, quantity=bid_size))
        if can_ask and not have_ask:
            actions.append(PlaceOrder(side=Side.SELL, price_ticks=my_ask_t, quantity=ask_size))

        self._last_bid_sz = bid_size if can_bid else 0.0
        self._last_ask_sz = ask_size if can_ask else 0.0
        if self._cool_bid > 0:
            self._cool_bid -= 1
        if self._cool_ask > 0:
            self._cool_ask -= 1

        return actions
