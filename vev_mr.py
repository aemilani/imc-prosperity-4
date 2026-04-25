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


@dataclass
class Vev5000(Product):
    name: str = 'VEV_5000'
    limit: int = 300
    price_mean: float = 253
    price_std: float = 12
    z_score_take_thr: float = 1.5


@dataclass
class Vev5100(Product):
    name: str = 'VEV_5100'
    limit: int = 300
    price_mean: float = 168
    price_std: float = 11
    z_score_take_thr: float = 1.5


@dataclass
class Vev5200(Product):
    name: str = 'VEV_5200'
    limit: int = 300
    price_mean: float = 97
    price_std: float = 8
    z_score_take_thr: float = 1.5


@dataclass
class Vev5300(Product):
    name: str = 'VEV_5300'
    limit: int = 300
    price_mean: float = 49
    price_std: float = 5
    z_score_take_thr: float = 1.5


def calc_vev5000_fair_value(state: TradingState, previous_state: Dict) -> float:
    previous_price: float | None = previous_state.get('vev5000_last_price')
    order_depth: OrderDepth = state.order_depths['VEV_5000']

    if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())

        filtered_asks = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 15]
        filtered_bids = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 15]
        best_filtered_ask = min(filtered_asks) if len(filtered_asks) > 0 else None
        best_filtered_bid = max(filtered_bids) if len(filtered_bids) > 0 else None

        if best_filtered_ask and best_filtered_bid:
            fair_value = (best_filtered_ask + best_filtered_bid) / 2
        else:
            fair_value = (best_ask + best_bid) / 2

        if not previous_price:
            return fair_value
        else:
            curr_logr = np.log(fair_value / previous_price)
            next_logr = curr_logr * -0.05  # mean-reversion param
            return fair_value * np.exp(next_logr)
    else:
        return previous_price


def calc_vev5000_ema_stats(previous_state: Dict, vev:Vev5000) -> tuple[float, float]:
    current_price = vev.fair_value
    ema_mean = vev.price_mean
    ema_std = vev.price_std

    if current_price is None:
        return ema_mean, ema_std

    ema_mean = previous_state.get('vev5000_ema_mean', ema_mean)
    ema_std = previous_state.get('vev5000_ema_std', ema_std)

    window_size = 10000
    alpha = 2 / (window_size + 1)

    diff = current_price - ema_mean

    # Update Variance FIRST, then Mean (order is mathematically important)
    ema_var = ema_std ** 2
    ema_var = (1 - alpha) * ema_var + alpha * (diff ** 2)
    ema_mean = ema_mean + (alpha * diff)

    current_std = math.sqrt(ema_var)

    return ema_mean, current_std


def trade_vev5000(state: TradingState, vev:Vev5000) -> List[Order]:
    order_depth: OrderDepth = state.order_depths['VEV_5000']
    orders: List[Order] = []

    if not vev.fair_value:
        return orders

    safe_std = vev.price_std if vev.price_std > 0 else 1e-6
    z_score = (vev.fair_value - vev.price_mean) / safe_std

    if z_score < -vev.z_score_take_thr:
        target_position = vev.limit
    elif z_score > vev.z_score_take_thr:
        target_position = -vev.limit
    elif z_score < 0 and vev.position < 0:
        target_position = 0
    elif z_score > 0 and vev.position > 0:
        target_position = 0
    else:
        target_position = vev.position

    position_diff = round(vev.position - target_position)

    if position_diff > 0 and len(order_depth.buy_orders) != 0:  # SELL
        best_bid = max(order_depth.buy_orders.keys())
        best_bid_amount = order_depth.buy_orders[best_bid]

        size = min(position_diff, best_bid_amount)
        orders.append(Order(vev.name, best_bid, -size))
    elif position_diff < 0 and len(order_depth.sell_orders) != 0:  # BUY
        best_ask = min(order_depth.sell_orders.keys())
        best_ask_amount = -1 * order_depth.sell_orders[best_ask]

        size = min(-position_diff, best_ask_amount)
        orders.append(Order(vev.name, best_ask, size))

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
            position = state.position.get(product_name, 0)
            print(f'{product_name} position: {position}')
            orders: List[Order] = []
            if product_name == 'VEV_5000':
                vev5000_fair_value = calc_vev5000_fair_value(state, previous_state)
                vev5000 = Vev5000(position=position, fair_value=vev5000_fair_value)
                ema_mean, ema_std = calc_vev5000_ema_stats(previous_state, vev5000)
                vev5000.price_mean = ema_mean
                vev5000.price_std = ema_std
                orders.extend(trade_vev5000(state, vev5000))
                previous_state['vev5000_last_price'] = vev5000_fair_value
                previous_state['vev5000_ema_mean'] = ema_mean
                previous_state['vev5000_ema_std'] = ema_std

            result[product_name] = orders
            print('---')

        trader_data = jsonpickle.encode(previous_state)
        return result, conversions, trader_data