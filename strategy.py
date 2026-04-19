from __future__ import annotations
from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState

class Strategy(BaseStrategy):
    """Larger size on the side we want inventory to move away from."""
    base_size = 4.0
    cooldown_steps = 5
    inventory_cap = 30.0

    def __init__(self):
        super().__init__()
        self._cool_bid = 0
        self._cool_ask = 0

    def on_step(self, state):
        if state.buy_filled_quantity > 0: self._cool_bid = self.cooldown_steps
        if state.sell_filled_quantity > 0: self._cool_ask = self.cooldown_steps
        bid_t = state.competitor_best_bid_ticks
        ask_t = state.competitor_best_ask_ticks
        if bid_t is None or ask_t is None: return [CancelAll()]
        my_bid_t = bid_t + 1
        my_ask_t = ask_t - 1
        if my_bid_t >= my_ask_t: return [CancelAll()]
        net = state.yes_inventory - state.no_inventory
        # Reduce bid size if long, increase ask size if long
        bid_size = max(1.0, self.base_size * (1 - net/50.0))
        ask_size = max(1.0, self.base_size * (1 + net/50.0))
        acts = [CancelAll()]
        if self._cool_bid <= 0 and net < self.inventory_cap:
            acts.append(PlaceOrder(Side.BUY, my_bid_t, bid_size))
        elif self._cool_bid > 0: self._cool_bid -= 1
        if self._cool_ask <= 0 and net > -self.inventory_cap:
            acts.append(PlaceOrder(Side.SELL, my_ask_t, ask_size))
        elif self._cool_ask > 0: self._cool_ask -= 1
        return acts
