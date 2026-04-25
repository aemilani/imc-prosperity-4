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
class CallOption(Product):
    strike_price: int = None
    time_to_expiry: float = None
    implied_vol: float = None
    delta: float = None
    vega: float = None
    moneyness: float = None
    price_mean: float = None
    price_std: float = None
    z_score_take_thr: float = None
    vol_thr: float = None
    mr_param: float = None


@dataclass
class Velvet(Product):
    name: str = 'VELVETFRUIT_EXTRACT'
    limit: int = 200
    price_mean: float = 5250
    price_std: float = 15
    z_score_take_thr: float = 1.5


CALL_CONFIGS = {
    5000: dict(limit=300, price_mean=253, price_std=12, z_score_take_thr=1.5, vol_thr=15, mr_param=-0.05),
    5100: dict(limit=300, price_mean=168, price_std=11, z_score_take_thr=1.5, vol_thr=15, mr_param=-0.08),
    5200: dict(limit=300, price_mean=97,  price_std=8, z_score_take_thr=1.5, vol_thr=15, mr_param=-0.08),
    5300: dict(limit=300, price_mean=49,  price_std=5, z_score_take_thr=1.5, vol_thr=15, mr_param=-0.14),
}


def make_call(strike: int) -> CallOption:
    return CallOption(name=f"VEV_{strike}", strike_price=strike, **CALL_CONFIGS[strike])


def calc_velvet_fair_value(state: TradingState, previous_state: Dict) -> float:
    previous_price: float | None = previous_state.get('velvet_last_price')
    order_depth: OrderDepth = state.order_depths['VELVETFRUIT_EXTRACT']

    if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())

        filtered_asks = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 30]
        filtered_bids = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 30]
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
            next_logr = curr_logr * -0.04  # mean-reversion param
            return fair_value * np.exp(next_logr)
    else:
        return previous_price


def calc_vev_fair_value(state: TradingState, previous_state: Dict, vev: CallOption) -> float:
    previous_price: float | None = previous_state.get(f'vev{vev.strike_price}_last_price')
    order_depth: OrderDepth = state.order_depths[f'VEV_{vev.strike_price}']

    if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())

        filtered_asks = [price for price in order_depth.sell_orders.keys() if \
                         abs(order_depth.sell_orders[price]) >= vev.vol_thr]
        filtered_bids = [price for price in order_depth.buy_orders.keys() if \
                         abs(order_depth.buy_orders[price]) >= vev.vol_thr]
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
            next_logr = curr_logr * vev.mr_param
            return fair_value * np.exp(next_logr)
    else:
        return previous_price


def calc_velvet_ema_stats(previous_state: Dict, velvet:Velvet) -> tuple[float, float]:
    current_price = velvet.fair_value
    ema_mean = velvet.price_mean
    ema_std = velvet.price_std

    if current_price is None:
        return ema_mean, ema_std

    ema_mean = previous_state.get('velvet_ema_mean', ema_mean)
    ema_std = previous_state.get('velvet_ema_std', ema_std)

    window_size = 10000
    alpha = 2 / (window_size + 1)

    diff = current_price - ema_mean

    # Update Variance FIRST, then Mean (order is mathematically important)
    ema_var = ema_std ** 2
    ema_var = (1 - alpha) * ema_var + alpha * (diff ** 2)
    ema_mean = ema_mean + (alpha * diff)

    current_std = math.sqrt(ema_var)

    return ema_mean, current_std


def calc_vev_ema_stats(previous_state: Dict, vev:CallOption) -> tuple[float, float]:
    current_price = vev.fair_value
    ema_mean = vev.price_mean
    ema_std = vev.price_std

    if current_price is None:
        return ema_mean, ema_std

    ema_mean = previous_state.get(f'vev{vev.strike_price}_ema_mean', ema_mean)
    ema_std = previous_state.get(f'vev{vev.strike_price}_ema_std', ema_std)

    window_size = 10000
    alpha = 2 / (window_size + 1)

    diff = current_price - ema_mean

    # Update Variance FIRST, then Mean (order is mathematically important)
    ema_var = ema_std ** 2
    ema_var = (1 - alpha) * ema_var + alpha * (diff ** 2)
    ema_mean = ema_mean + (alpha * diff)

    current_std = math.sqrt(ema_var)

    return ema_mean, current_std


def trade_velvet(state: TradingState, velvet:Velvet) -> List[Order]:
    order_depth: OrderDepth = state.order_depths['VELVETFRUIT_EXTRACT']
    orders: List[Order] = []

    if not velvet.fair_value:
        return orders

    safe_std = velvet.price_std if velvet.price_std > 0 else 1e-6
    z_score = (velvet.fair_value - velvet.price_mean) / safe_std

    if z_score < -velvet.z_score_take_thr:
        target_position = velvet.limit
    elif z_score > velvet.z_score_take_thr:
        target_position = -velvet.limit
    elif z_score < 0 and velvet.position < 0:
        target_position = 0
    elif z_score > 0 and velvet.position > 0:
        target_position = 0
    else:
        target_position = velvet.position

    position_diff = round(velvet.position - target_position)

    if position_diff > 0 and len(order_depth.buy_orders) != 0:  # SELL
        best_bid = max(order_depth.buy_orders.keys())
        best_bid_amount = order_depth.buy_orders[best_bid]

        size = min(position_diff, best_bid_amount)
        orders.append(Order(velvet.name, best_bid, -size))
    elif position_diff < 0 and len(order_depth.sell_orders) != 0:  # BUY
        best_ask = min(order_depth.sell_orders.keys())
        best_ask_amount = -1 * order_depth.sell_orders[best_ask]

        size = min(-position_diff, best_ask_amount)
        orders.append(Order(velvet.name, best_ask, size))

    return orders


def trade_vev(state: TradingState, vev:CallOption) -> List[Order]:
    order_depth: OrderDepth = state.order_depths[f'VEV_{vev.strike_price}']
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
        if 'VELVETFRUIT_EXTRACT' in state.order_depths:
            orders: List[Order] = []

            velvet_position = state.position.get('VELVETFRUIT_EXTRACT', 0)
            velvet_fair_value = calc_velvet_fair_value(state, previous_state)

            velvet = Velvet(position=velvet_position, fair_value=velvet_fair_value)

            ema_mean, ema_std = calc_velvet_ema_stats(previous_state, velvet)
            velvet.price_mean = ema_mean
            velvet.price_std = ema_std

            orders.extend(trade_velvet(state, velvet))

            previous_state['velvet_last_price'] = velvet_fair_value
            previous_state['velvet_ema_mean'] = ema_mean
            previous_state['velvet_ema_std'] = ema_std

            result['VELVETFRUIT_EXTRACT'] = orders

            strikes = CALL_CONFIGS.keys()
            calls = []
            for strike in strikes:
                calls.append(make_call(strike))

            for call in calls:
                orders: List[Order] = []

                call.position = state.position.get(call.name, 0)
                call_fair_value = calc_vev_fair_value(state, previous_state, call)
                call.fair_value = call_fair_value

                ema_mean, ema_std = calc_vev_ema_stats(previous_state, call)
                call.price_mean = ema_mean
                call.price_std = ema_std

                orders.extend(trade_vev(state, call))

                previous_state[f'vev{call.strike_price}_last_price'] = call_fair_value
                previous_state[f'vev{call.strike_price}_ema_mean'] = ema_mean
                previous_state[f'vev{call.strike_price}_ema_std'] = ema_std

                result[call.name] = orders

        trader_data = jsonpickle.encode(previous_state)
        return result, conversions, trader_data
