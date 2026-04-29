import jsonpickle
import math
import numpy as np
from dataclasses import dataclass
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict


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


# def trade_product(state: TradingState, prod: Product) -> List[Order]:
#     order_depth: OrderDepth = state.order_depths[prod.name]
#     orders: List[Order] = []
#
#     if not prod.fair_value:
#         return orders
#
#     safe_std = prod.price_std if prod.price_std > 0 else 1e-6
#     z_score = (prod.fair_value - prod.price_mean) / safe_std
#
#     if z_score < -prod.z_score_thr:
#         target_position = prod.limit
#     elif z_score > prod.z_score_thr:
#         target_position = -prod.limit
#     elif z_score < 0 and prod.position < 0:
#         target_position = 0
#     elif z_score > 0 and prod.position > 0:
#         target_position = 0
#     else:
#         target_position = prod.position
#
#     position_diff = round(prod.position - target_position)
#
#     if position_diff > 0 and len(order_depth.buy_orders) != 0:  # SELL
#         best_bid = max(order_depth.buy_orders.keys())
#         best_bid_amount = order_depth.buy_orders[best_bid]
#
#         size = min(position_diff, best_bid_amount)
#         orders.append(Order(prod.name, best_bid, -size))
#         prod.posted_sell_volume += size
#     elif position_diff < 0 and len(order_depth.sell_orders) != 0:  # BUY
#         best_ask = min(order_depth.sell_orders.keys())
#         best_ask_amount = -1 * order_depth.sell_orders[best_ask]
#
#         size = min(-position_diff, best_ask_amount)
#         orders.append(Order(prod.name, best_ask, size))
#         prod.posted_buy_volume += size
#
#     if position_diff == 0 and -0.2 * prod.limit <= prod.position <= 0.2 * prod.limit:
#
#         # Market taking
#         if len(order_depth.sell_orders) != 0:
#             best_ask = min(order_depth.sell_orders.keys())
#             best_ask_amount = -1 * order_depth.sell_orders[best_ask]
#
#             if best_ask <= prod.fair_value - prod.take_thr:
#                 quantity = min(
#                     best_ask_amount, prod.limit - prod.position
#                 )
#                 if quantity > 0:
#                     orders.append(Order(prod.name, best_ask, quantity))
#                     prod.posted_buy_volume += quantity
#                     order_depth.sell_orders[best_ask] += quantity
#                     if order_depth.sell_orders[best_ask] == 0:
#                         del order_depth.sell_orders[best_ask]
#         if len(order_depth.buy_orders) != 0:
#             best_bid = max(order_depth.buy_orders.keys())
#             best_bid_amount = order_depth.buy_orders[best_bid]
#
#             if best_bid >= prod.fair_value + prod.take_thr:
#                 quantity = min(
#                     best_bid_amount, prod.limit + prod.position
#                 )
#                 if quantity > 0:
#                     orders.append(Order(prod.name, best_bid, -1 * quantity))
#                     prod.posted_sell_volume += quantity
#                     order_depth.buy_orders[best_bid] -= quantity
#                     if order_depth.buy_orders[best_bid] == 0:
#                         del order_depth.buy_orders[best_bid]
#
#         # Market making
#         best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
#         best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
#
#         whale_bid_price = None
#         if order_depth.buy_orders:
#             for price, volume in order_depth.buy_orders.items():
#                 if 10 <= volume <= 15:
#                     whale_bid_price = price
#                     break  # Found him!
#
#         whale_ask_price = None
#         if order_depth.sell_orders:
#             for price, volume in order_depth.sell_orders.items():
#                 if -15 <= volume <= -10:
#                     whale_ask_price = price
#                     break  # Found him!
#
#         if best_bid and best_ask and (best_ask - best_bid > 1):
#             if whale_bid_price:
#                 my_mm_bid = whale_bid_price + 1  # penny
#             else:
#                 my_mm_bid = best_bid + 1  # Fallback if Whale isn't detected
#
#             if whale_ask_price:
#                 my_mm_ask = whale_ask_price - 1  # penny
#             else:
#                 my_mm_ask = best_ask - 1  # Fallback if Whale isn't detected
#
#             if my_mm_bid >= best_ask:
#                 my_mm_bid = best_ask - 1
#             if my_mm_ask <= best_bid:
#                 my_mm_ask = best_bid + 1
#
#         else:
#             my_mm_bid = prod.fair_value - prod.default_thr
#             my_mm_ask = prod.fair_value + prod.default_thr
#
#         buy_quantity = prod.limit - (prod.position + prod.posted_buy_volume)
#         if buy_quantity > 0:
#             orders.append(Order(prod.name, round(my_mm_bid), buy_quantity))  # Buy order
#
#         sell_quantity = prod.limit + (prod.position - prod.posted_sell_volume)
#         if sell_quantity > 0:
#             orders.append(Order(prod.name, round(my_mm_ask), -sell_quantity))  # Sell order
#
#     return orders


def trade_product(state: TradingState, prod: Product) -> List[Order]:
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

    # Position clearance
    position_after_take = prod.position + prod.posted_buy_volume - prod.posted_sell_volume
    fair_for_bid = round(prod.fair_value)
    fair_for_ask = round(prod.fair_value)
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
        if price > prod.fair_value + prod.disregard_thr
    ]
    bids_below_fair = [
        price
        for price in order_depth.buy_orders.keys()
        if price < prod.fair_value - prod.disregard_thr
    ]
    best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
    best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

    ask = round(prod.fair_value + prod.default_thr)
    if best_ask_above_fair is not None:
        if abs(best_ask_above_fair - prod.fair_value) <= prod.join_thr:
            ask = best_ask_above_fair  # join
        else:
            ask = best_ask_above_fair - 1  # penny

    bid = round(prod.fair_value - prod.default_thr)
    if best_bid_below_fair is not None:
        if abs(prod.fair_value - best_bid_below_fair) <= prod.join_thr:
            bid = best_bid_below_fair
        else:
            bid = best_bid_below_fair + 1

    buy_quantity = prod.limit - (prod.position + prod.posted_buy_volume)
    if buy_quantity > 0:
        orders.append(Order(prod.name, round(bid), buy_quantity))  # Buy order

    sell_quantity = prod.limit + (prod.position - prod.posted_sell_volume)
    if sell_quantity > 0:
        orders.append(Order(prod.name, round(ask), -sell_quantity))  # Sell order

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

            orders.extend(trade_product(state, prod))

            result[product_name] = orders

        trader_data = jsonpickle.encode(previous_state)
        return result, conversions, trader_data
