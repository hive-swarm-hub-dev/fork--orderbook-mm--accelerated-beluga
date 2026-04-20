from __future__ import annotations
from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, CancelOrder, PlaceOrder, Side, StepState

class Strategy(BaseStrategy):
    base_size = 5.0
    spread_scale = 0.7
    narrow_cool = 7
    wide_cool = 2
    narrow_gap = 4
    arb_thresh = 0.95
    inventory_cap = 40
    skew_unit = 20
    drift_decay = 0.80
    drift_thresh = 1.0
    drift_cool = 3
    mild_drift_thresh = 0.3
    drift_down_mul = 0.3
    fast_decay = 0.0
    fast_thresh = 0.5
    fast_cool = 6
    jump_thresh = 3.5
    jump_cool = 12

    def __init__(self):
        super().__init__()
        self._cool_bid = 0; self._cool_ask = 0
        self._prev_gap = 4
        self._last_bid_sz = 0.0; self._last_ask_sz = 0.0
        self._prev_mid = None
        self._drift = 0.0
        self._fast_drift = 0.0
        self._initial_gap = None

    def on_step(self, state):
        bid_t = state.competitor_best_bid_ticks
        ask_t = state.competitor_best_ask_ticks
        mid = None if (bid_t is None or ask_t is None) else (bid_t + ask_t) / 2.0
        if self._prev_mid is not None and mid is not None:
            d = mid - self._prev_mid
            self._drift = self.drift_decay * self._drift + d
            self._fast_drift = self.fast_decay * self._fast_drift + d
        self._prev_mid = mid
        cool = self.narrow_cool if self._prev_gap <= self.narrow_gap else self.wide_cool
        if self._last_bid_sz > 0 and state.buy_filled_quantity >= self.arb_thresh * self._last_bid_sz:
            self._cool_bid = max(self._cool_bid, cool)
        if self._last_ask_sz > 0 and state.sell_filled_quantity >= self.arb_thresh * self._last_ask_sz:
            self._cool_ask = max(self._cool_ask, cool)
        if self._drift > self.drift_thresh:
            self._cool_ask = max(self._cool_ask, self.drift_cool)
        elif self._drift < -self.drift_thresh:
            self._cool_bid = max(self._cool_bid, self.drift_cool)
        # Fast drift: catches sudden jumps that haven't yet built up the slow EWMA.
        if self._fast_drift > self.fast_thresh:
            self._cool_ask = max(self._cool_ask, self.fast_cool)
        elif self._fast_drift < -self.fast_thresh:
            self._cool_bid = max(self._cool_bid, self.fast_cool)
        if abs(self._fast_drift) >= self.jump_thresh:
            self._cool_bid = max(self._cool_bid, self.jump_cool)
            self._cool_ask = max(self._cool_ask, self.jump_cool)

        if bid_t is None or ask_t is None:
            self._prev_gap = 4
            self._last_bid_sz = self._last_ask_sz = 0.0
            return [CancelAll()]
        self._prev_gap = ask_t - bid_t
        if self._initial_gap is None:
            self._initial_gap = ask_t - bid_t
        if self._initial_gap <= 2:
            self._last_bid_sz = self._last_ask_sz = 0.0
            return [CancelAll()]
        my_bid_t = bid_t + 1; my_ask_t = ask_t - 1
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
        sz = sz_mul * self.base_size * (1.0 + max(0, gap - 2) * self.spread_scale)
        net = state.yes_inventory - state.no_inventory
        bid_sz = max(1.0, sz * (1 - net/self.skew_unit))
        ask_sz = max(1.0, sz * (1 + net/self.skew_unit))
        if self._drift > self.mild_drift_thresh:
            ask_sz *= self.drift_down_mul
        elif self._drift < -self.mild_drift_thresh:
            bid_sz *= self.drift_down_mul
        bid_sz = max(1.0, bid_sz); ask_sz = max(1.0, ask_sz)
        can_bid = self._cool_bid <= 0 and net < self.inventory_cap
        can_ask = self._cool_ask <= 0 and net > -self.inventory_cap
        acts = []; hb = False; ha = False
        for o in state.own_orders:
            ob = o.side is Side.BUY and can_bid and o.price_ticks == my_bid_t and abs(o.remaining_quantity - bid_sz) < 0.5
            oa = o.side is Side.SELL and can_ask and o.price_ticks == my_ask_t and abs(o.remaining_quantity - ask_sz) < 0.5
            if ob: hb = True
            elif oa: ha = True
            else: acts.append(CancelOrder(o.order_id))
        if can_bid and not hb: acts.append(PlaceOrder(Side.BUY, my_bid_t, bid_sz))
        if can_ask and not ha: acts.append(PlaceOrder(Side.SELL, my_ask_t, ask_sz))
        self._last_bid_sz = bid_sz if can_bid else 0.0
        self._last_ask_sz = ask_sz if can_ask else 0.0
        if self._cool_bid > 0: self._cool_bid -= 1
        if self._cool_ask > 0: self._cool_ask -= 1
        return acts
