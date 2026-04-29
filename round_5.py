import jsonpickle
import math
import numpy as np
from dataclasses import dataclass
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Tuple


@dataclass
class Product:
    name: str
    limit: int
    fair_value: float = None
    position: int = 0
    posted_buy_volume: int = 0
    posted_sell_volume: int = 0
    best_bid: float = None
    best_ask: float = None
    best_bid_size: int = None
    best_ask_size: int = None
    take_thr: int = None
    default_thr: int = None
    disregard_thr: int = None
    join_thr: int = None
    volume_thr: int = None
    price_mean: float = None
    price_std: float = None
    window_size:  int = None
    z_score_thr: float = None


@dataclass
class Spread(Product):
    name: str = 'SPREAD'
    limit: int = 60
    product_names: Tuple[str] = ('PICNIC_BASKET1', 'PICNIC_BASKET2', 'CROISSANTS', 'JAMS', 'DJEMBES')
    product_weights: Tuple[int] = (1, -1, -2, -1, -1)  # Basket1, Basket2, Croissants, Jams, Djembes
    mean: float = -202.3
    std: float = 83.9


CONFIGS = {
    "GALAXY": dict(limit=10, price_mean=0, price_std=0, z_score_thr=1.9, window_size=10000),

    "SLEEP": dict(limit=10, price_mean=0, price_std=0, z_score_thr=1.9, window_size=10000),

    "MICROCHIP": dict(limit=10, price_mean=0, price_std=0, z_score_thr=1.9, window_size=10000),

    "PEBBLES": dict(limit=10, price_mean=0, price_std=0, z_score_thr=1.9, window_size=10000),

    "ROBOT": dict(limit=10, price_mean=0, price_std=0, z_score_thr=1.9, window_size=10000),

    "UV": dict(limit=10, price_mean=0, price_std=0, z_score_thr=1.9, window_size=10000),

    "TRANSLATOR": dict(limit=10, price_mean=0, price_std=0, z_score_thr=1.9, window_size=10000),

    "PANEL": dict(limit=10, price_mean=0, price_std=0, z_score_thr=1.9, window_size=10000),

    "OXYGEN": dict(limit=10, price_mean=0, price_std=0, z_score_thr=1.9, window_size=10000),

    "SNACKPACK": dict(limit=10, price_mean=0, price_std=0, z_score_thr=1.9, window_size=10000),
}


def make_product(name: str) -> Product:
    return Product(name=name, **CONFIGS[name.split("_")[0]])


def calc_fair_value(state: TradingState, previous_state: Dict, prod: Product) -> float:
    previous_price: float | None = previous_state.get(f'{prod.name}_last_price')
    order_depth: OrderDepth = state.order_depths[prod.name]

    if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())

        fair_value = (best_ask + best_bid) / 2

        if not previous_price:
            return fair_value
        else:
            curr_logr = np.log(fair_value / previous_price)

            if prod.name == "ROBOT_IRONING" or prod.name == "OXYGEN_SHAKE_EVENING_BREATH":
                mr_param = -0.16
            elif prod.name == "OXYGEN_SHAKE_CHOCOLATE":
                mr_param = -0.12
            else:
                mr_param = 0

            next_logr = curr_logr * mr_param
            return fair_value * np.exp(next_logr)
    else:
        return previous_price


def calc_ema_stats(previous_state: Dict, prod: Product) -> tuple[float, float]:
    current_price = prod.fair_value
    ema_mean = prod.price_mean
    ema_std = prod.price_std

    if current_price is None:
        return ema_mean, ema_std

    ema_mean = previous_state.get(f'{prod.name}_ema_mean', ema_mean)
    ema_std = previous_state.get(f'{prod.name}_ema_std', ema_std)

    alpha = 2 / (prod.window_size + 1)

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
