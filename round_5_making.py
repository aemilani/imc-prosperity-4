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
    "GALAXY": dict(limit=10, take_thr=1, default_thr=6, volume_thr=10, disregard_thr=1, join_thr=3,
                   price_mean=0, price_std=0, z_score_thr=1.9, window_size=10000),

    "SLEEP": dict(limit=10, take_thr=1, default_thr=4, volume_thr=10, disregard_thr=1, join_thr=2,
                  price_mean=0, price_std=0, z_score_thr=1.9, window_size=10000),

    "MICROCHIP": dict(limit=10, take_thr=1, default_thr=4, volume_thr=10, disregard_thr=1, join_thr=2,
                      price_mean=0, price_std=0, z_score_thr=1.9, window_size=10000),

    "PEBBLES": dict(limit=10, take_thr=1, default_thr=6, volume_thr=10, disregard_thr=1, join_thr=3,
                    price_mean=0, price_std=0, z_score_thr=1.9, window_size=10000),

    "ROBOT": dict(limit=10, take_thr=1, default_thr=3, volume_thr=10, disregard_thr=1, join_thr=2,
                  price_mean=0, price_std=0, z_score_thr=1.9, window_size=10000),

    "UV": dict(limit=10, take_thr=1, default_thr=6, volume_thr=10, disregard_thr=1, join_thr=3,
               price_mean=0, price_std=0, z_score_thr=1.9, window_size=10000),

    "TRANSLATOR": dict(limit=10, take_thr=1, default_thr=4, volume_thr=10, disregard_thr=1, join_thr=2,
                       price_mean=0, price_std=0, z_score_thr=1.9, window_size=10000),

    "PANEL": dict(limit=10, take_thr=1, default_thr=4, volume_thr=10, disregard_thr=1, join_thr=2,
                  price_mean=0, price_std=0, z_score_thr=1.9, window_size=10000),

    "OXYGEN": dict(limit=10, take_thr=1, default_thr=6, volume_thr=10, disregard_thr=1, join_thr=3,
                   price_mean=0, price_std=0, z_score_thr=1.9, window_size=10000),

    "SNACKPACK": dict(limit=10, take_thr=1, default_thr=8, volume_thr=10, disregard_thr=1, join_thr=3,
                      price_mean=0, price_std=0, z_score_thr=1.9, window_size=10000),
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


def trade_mean_reversal(state: TradingState, spr: Spread) -> List[Order]:
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


def trade_making(state: TradingState, prod: Product) -> List[Order]:
    order_depth: OrderDepth = state.order_depths[prod.name]
    orders: List[Order] = []

    if not prod.fair_value:
        return orders

    # Market taking
    if len(order_depth.sell_orders) != 0:
        best_ask = min(order_depth.sell_orders.keys())
        best_ask_amount = -1 * order_depth.sell_orders[best_ask]

        if abs(best_ask_amount) <= prod.volume_thr:
            if best_ask <= prod.fair_value - prod.take_thr:
                quantity = min(
                    best_ask_amount, prod.limit - prod.position
                )  # max amt to buy
                if quantity > 0:
                    orders.append(Order(prod.name, best_ask, quantity))
                    prod.posted_buy_volume += quantity
                    order_depth.sell_orders[best_ask] += quantity
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]

    if len(order_depth.buy_orders) != 0:
        best_bid = max(order_depth.buy_orders.keys())
        best_bid_amount = order_depth.buy_orders[best_bid]

        if abs(best_bid_amount) <= prod.volume_thr:
            if best_bid >= prod.fair_value + prod.take_thr :
                quantity = min(
                    best_bid_amount, prod.limit + prod.position
                )  # should be the max we can sell
                if quantity > 0:
                    orders.append(Order(prod.name, best_bid, -1 * quantity))
                    prod.posted_sell_volume += quantity
                    order_depth.buy_orders[best_bid] -= quantity
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]

    # Skewing
    position_after_take = prod.position + prod.posted_buy_volume - prod.posted_sell_volume
    max_skew_ticks = 2 * prod.default_thr  # Tunable: maximum ticks to shift when at the position limit
    inventory_ratio = position_after_take / prod.limit
    skewed_fair_value = round(prod.fair_value - (inventory_ratio * max_skew_ticks))

    # Position clearance
    fair_for_bid = skewed_fair_value
    fair_for_ask = skewed_fair_value
    buy_quantity = prod.limit - (prod.position + prod.posted_buy_volume)
    sell_quantity = prod.limit + (prod.position - prod.posted_sell_volume)

    if position_after_take > 0:
        # Aggregate volume from all buy orders with price greater than fair_for_ask
        clear_quantity = sum(
            volume
            for price, volume in order_depth.buy_orders.items()
            if price >= fair_for_ask
        )
        clear_quantity = min(clear_quantity, position_after_take)
        sent_quantity = min(sell_quantity, clear_quantity)
        if sent_quantity > 0:
            orders.append(Order(prod.name, fair_for_ask, -abs(sent_quantity)))
            prod.posted_sell_volume += abs(sent_quantity)

    if position_after_take < 0:
        # Aggregate volume from all sell orders with price lower than fair_for_bid
        clear_quantity = sum(
            abs(volume)
            for price, volume in order_depth.sell_orders.items()
            if price <= fair_for_bid
        )
        clear_quantity = min(clear_quantity, abs(position_after_take))
        sent_quantity = min(buy_quantity, clear_quantity)
        if sent_quantity > 0:
            orders.append(Order(prod.name, fair_for_bid, abs(sent_quantity)))
            prod.posted_buy_volume += abs(sent_quantity)

    # Market making
    asks_above_fair = [
        price
        for price in order_depth.sell_orders.keys()
        if price > skewed_fair_value + prod.disregard_thr
    ]
    bids_below_fair = [
        price
        for price in order_depth.buy_orders.keys()
        if price < skewed_fair_value - prod.disregard_thr
    ]
    best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
    best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

    target_ask = round(skewed_fair_value + prod.default_thr)
    ask = target_ask
    if best_ask_above_fair is not None:
        if abs(best_ask_above_fair - skewed_fair_value) <= prod.join_thr:
            ask = min(best_ask_above_fair, target_ask)  # join # Enforce ceiling
        else:
            ask = min(best_ask_above_fair - 1, target_ask)  # penny # Enforce ceiling

    target_bid = round(skewed_fair_value - prod.default_thr)
    bid = target_bid
    if best_bid_below_fair is not None:
        if abs(skewed_fair_value - best_bid_below_fair) <= prod.join_thr:
            bid = max(best_bid_below_fair, target_bid)  # join # Enforce floor
        else:
            bid = max(best_bid_below_fair + 1, target_bid)  # penny # Enforce floor

    buy_quantity = prod.limit - (prod.position + prod.posted_buy_volume)
    if buy_quantity > 0:
        orders.append(Order(prod.name, round(bid), buy_quantity))

    sell_quantity = prod.limit + (prod.position - prod.posted_sell_volume)
    if sell_quantity > 0:
        orders.append(Order(prod.name, round(ask), -sell_quantity))

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

            orders.extend(trade_making(state, prod))

            result[product_name] = orders

        trader_data = jsonpickle.encode(previous_state)
        return result, conversions, trader_data
