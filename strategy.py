from __future__ import annotations
from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, CancelOrder, PlaceOrder, Side, StepState

class Strategy(BaseStrategy):
    base_size = 4.0
    spread_scale = 0.5
    narrow_cool = 7
    wide_cool = 2
    narrow_gap = 4
    arb_thresh = 0.95
    inventory_cap = 30.0
    skew_unit = 30.0

    def __init__(self):
        super().__init__()
        self._cool_bid = 0
        self._cool_ask = 0
        self._prev_gap = 4
        self._last_bid_sz = 0.0
        self._last_ask_sz = 0.0

    def on_step(self, state):
        cool = self.narrow_cool if self._prev_gap <= self.narrow_gap else self.wide_cool
        if self._last_bid_sz > 0 and state.buy_filled_quantity >= self.arb_thresh * self._last_bid_sz:
            self._cool_bid = max(self._cool_bid, cool)
        if self._last_ask_sz > 0 and state.sell_filled_quantity >= self.arb_thresh * self._last_ask_sz:
            self._cool_ask = max(self._cool_ask, cool)
        bid_t = state.competitor_best_bid_ticks
        ask_t = state.competitor_best_ask_ticks
        if bid_t is None or ask_t is None:
            self._prev_gap = 4
            self._last_bid_sz = self._last_ask_sz = 0.0
            return [CancelAll()]
        self._prev_gap = ask_t - bid_t
        my_bid_t = bid_t + 1
        my_ask_t = ask_t - 1
        if my_bid_t >= my_ask_t:
            self._last_bid_sz = self._last_ask_sz = 0.0
            return [CancelAll()]
        gap = ask_t - bid_t
        sz = self.base_size * (1.0 + max(0, gap - 2) * self.spread_scale)
        net = state.yes_inventory - state.no_inventory
        bid_sz = max(1.0, sz * (1 - net/self.skew_unit))
        ask_sz = max(1.0, sz * (1 + net/self.skew_unit))
        can_bid = self._cool_bid <= 0 and net < self.inventory_cap
        can_ask = self._cool_ask <= 0 and net > -self.inventory_cap
        acts = []
        have_bid = False; have_ask = False
        for o in state.own_orders:
            ok_b = o.side is Side.BUY and can_bid and o.price_ticks == my_bid_t and abs(o.remaining_quantity - bid_sz) < 0.5
            ok_a = o.side is Side.SELL and can_ask and o.price_ticks == my_ask_t and abs(o.remaining_quantity - ask_sz) < 0.5
            if ok_b: have_bid = True
            elif ok_a: have_ask = True
            else: acts.append(CancelOrder(o.order_id))
        if can_bid and not have_bid: acts.append(PlaceOrder(Side.BUY, my_bid_t, bid_sz))
        if can_ask and not have_ask: acts.append(PlaceOrder(Side.SELL, my_ask_t, ask_sz))
        self._last_bid_sz = bid_sz if (can_bid and (have_bid or not have_bid)) else 0.0
        self._last_ask_sz = ask_sz if (can_ask and (have_ask or not have_ask)) else 0.0
        if not can_bid: self._last_bid_sz = 0.0
        if not can_ask: self._last_ask_sz = 0.0
        if self._cool_bid > 0: self._cool_bid -= 1
        if self._cool_ask > 0: self._cool_ask -= 1
        return acts
