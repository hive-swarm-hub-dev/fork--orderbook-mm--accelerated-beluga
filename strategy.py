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
    """V8: V7 with spread_scale 0.3 -> 0.5.

    Ramp up size more aggressively when competitor spread is wide. Sweep
    showed 0.5 beats 0.3 by ~0.2 on four-seed mean; 0.7 starts to plateau
    and 1.0 begins to hurt (inventory-cap-bound at wide spreads).
    """

    base_size = 4.0
    spread_scale = 0.5
    cooldown_steps = 5
    inventory_cap = 30.0
    skew_unit = 30.0
    size_tolerance = 0.5

    def __init__(self) -> None:
        super().__init__()
        self._cool_bid = 0
        self._cool_ask = 0

    def on_step(self, state: StepState):
        if state.buy_filled_quantity > 0:
            self._cool_bid = self.cooldown_steps
        if state.sell_filled_quantity > 0:
            self._cool_ask = self.cooldown_steps

        bid_t = state.competitor_best_bid_ticks
        ask_t = state.competitor_best_ask_ticks
        if bid_t is None or ask_t is None:
            return [CancelAll()]

        my_bid_t = bid_t + 1
        my_ask_t = ask_t - 1
        if my_bid_t >= my_ask_t:
            return [CancelAll()]

        gap = ask_t - bid_t
        sz = self.base_size * (1.0 + max(0, gap - 2) * self.spread_scale)

        net_inv = state.yes_inventory - state.no_inventory
        bid_size = max(1.0, sz * (1.0 - net_inv / self.skew_unit))
        ask_size = max(1.0, sz * (1.0 + net_inv / self.skew_unit))

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

        if self._cool_bid > 0:
            self._cool_bid -= 1
        if self._cool_ask > 0:
            self._cool_ask -= 1

        return actions
