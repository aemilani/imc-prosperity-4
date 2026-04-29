import jsonpickle
import math
import numpy as np
from dataclasses import dataclass
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Tuple


@dataclass
class Spread:
    name: str
    product_names: Tuple[int]
    product_weights: Tuple[int]
    fair_value: float
    price_mean: float
    price_std: float
    window_size: int
    z_score_thr: float

    def __post_init__(self):
        self.limit = 10 // np.abs(self.product_weights).max()


@dataclass
class SpreadSleep(Spread):
    name: str = 'SPREAD_SLEEP'
    product_names: Tuple[str] = (
        'SLEEP_POD_SUEDE', 'SLEEP_POD_LAMB_WOOL', 'SLEEP_POD_POLYESTER', 'SLEEP_POD_NYLON', 'SLEEP_POD_COTTON'
    )
    product_weights: Tuple[int] = (1, -1, -2, -1, -1)
    price_mean: float = 0
    price_std: float = 0


def calc_ema_stats(previous_state: Dict, spr: Spread) -> tuple[float, float]:
    current_price = spr.fair_value
    ema_mean = spr.price_mean
    ema_std = spr.price_std

    if current_price is None:
        return ema_mean, ema_std

    ema_mean = previous_state.get(f'{spr.name}_ema_mean', ema_mean)
    ema_std = previous_state.get(f'{spr.name}_ema_std', ema_std)

    alpha = 2 / (spr.window_size + 1)

    diff = current_price - ema_mean

    ema_var = ema_std ** 2
    ema_var = (1 - alpha) * ema_var + alpha * (diff ** 2)
    ema_mean = ema_mean + (alpha * diff)

    current_std = math.sqrt(ema_var)

    return ema_mean, current_std


def trade_mean_reversion(state: TradingState, spr: Spread) -> List[Order]:
    order_depth: OrderDepth = state.order_depths[spr.name]
    orders: List[Order] = []

    if not spr.fair_value:
        return orders

    safe_std = spr.price_std if spr.price_std > 0 else 1e-6
    z_score = (spr.fair_value - spr.price_mean) / safe_std

    if z_score < -spr.z_score_thr:
        target_position = spr.limit
    elif z_score > spr.z_score_thr:
        target_position = -spr.limit
    elif z_score < 0 and spr.position < 0:
        target_position = 0
    elif z_score > 0 and spr.position > 0:
        target_position = 0
    else:
        target_position = spr.position

    position_diff = round(spr.position - target_position)

    if position_diff > 0 and len(order_depth.buy_orders) != 0:  # SELL
        best_bid = max(order_depth.buy_orders.keys())
        best_bid_amount = order_depth.buy_orders[best_bid]

        size = min(position_diff, best_bid_amount)
        orders.append(Order(spr.name, best_bid, -size))
        spr.posted_sell_volume += size
    elif position_diff < 0 and len(order_depth.sell_orders) != 0:  # BUY
        best_ask = min(order_depth.sell_orders.keys())
        best_ask_amount = -1 * order_depth.sell_orders[best_ask]

        size = min(-position_diff, best_ask_amount)
        orders.append(Order(spr.name, best_ask, size))
        spr.posted_buy_volume += size

    return orders


class Trader:
    def run(self, state: TradingState):
        conversions = 0

        previous_state = {}
        if state.traderData:
            try:
                previous_state = jsonpickle.decode(state.traderData)
            except Exception:
                pass

        result = {}
        for product_name in state.order_depths:
            orders: List[Order] = []
            prod = make_product(product_name)

            position = state.position.get(product_name, 0)
            prod.position = position

            fair_value = calc_fair_value(state, previous_state, prod)
            prod.fair_value = fair_value
            previous_state[f'{prod.name}_last_price'] = fair_value

            orders.extend(trade_mean_reversion(state, prod))

            result[product_name] = orders

        trader_data = jsonpickle.encode(previous_state)
        return result, conversions, trader_data
