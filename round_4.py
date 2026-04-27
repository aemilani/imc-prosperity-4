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
class Hydrogel(Product):
    name: str = 'HYDROGEL_PACK'
    limit: int = 200
    default_thr: int = 8
    price_mean: float = 9995
    price_std: float = 35
    z_score_take_thr: float = 1.5


@dataclass
class Velvet(Product):
    name: str = 'VELVETFRUIT_EXTRACT'
    limit: int = 200
    price_mean: float = 5248
    price_std: float = 18
    z_score_take_thr: float = 1.5


CALL_CONFIGS = {
    4000: dict(limit=300, price_mean=1248, price_std=18, z_score_take_thr=1.9, vol_thr=7, mr_param=-0.07),
    4500: dict(limit=300, price_mean=748, price_std=18, z_score_take_thr=1.7, vol_thr=15, mr_param=-0.05),
    5000: dict(limit=300, price_mean=251, price_std=17, z_score_take_thr=1.2, vol_thr=15, mr_param=-0.07),
    5100: dict(limit=300, price_mean=161, price_std=16, z_score_take_thr=1.2, vol_thr=15, mr_param=-0.07),
    5200: dict(limit=300, price_mean=89,  price_std=13, z_score_take_thr=1.1, vol_thr=15, mr_param=-0.1),
    5300: dict(limit=300, price_mean=41,  price_std=9, z_score_take_thr=1.0, vol_thr=13, mr_param=-0.16),
    5400: dict(limit=300, price_mean=13,  price_std=4, z_score_take_thr=1.0, vol_thr=11, mr_param=-0.29),
    5500: dict(limit=300, price_mean=5,  price_std=2, z_score_take_thr=1.0, vol_thr=15, mr_param=-0.24),
}


def make_call(strike: int) -> CallOption:
    return CallOption(name=f"VEV_{strike}", strike_price=strike, **CALL_CONFIGS[strike])


def calc_hydrogel_fair_value(state: TradingState, previous_state: Dict) -> float:
    previous_price: float | None = previous_state.get('hydrogel_last_price')
    order_depth: OrderDepth = state.order_depths['HYDROGEL_PACK']

    if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())

        filtered_asks = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 10]
        filtered_bids = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 10]
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


def calc_velvet_fair_value(state: TradingState, previous_state: Dict) -> float:
    previous_price: float | None = previous_state.get('velvet_last_price')
    order_depth: OrderDepth = state.order_depths['VELVETFRUIT_EXTRACT']

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
            next_logr = curr_logr * -0.07  # mean-reversion param
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


def calc_hydrogel_ema_stats(previous_state: Dict, hydrogel:Hydrogel) -> tuple[float, float]:
    current_price = hydrogel.fair_value
    ema_mean = hydrogel.price_mean
    ema_std = hydrogel.price_std

    if current_price is None:
        return ema_mean, ema_std

    ema_mean = previous_state.get('hydrogel_ema_mean', ema_mean)
    ema_std = previous_state.get('hydrogel_ema_std', ema_std)

    window_size = 10000
    alpha = 2 / (window_size + 1)

    diff = current_price - ema_mean

    # Update Variance FIRST, then Mean (order is mathematically important)
    ema_var = ema_std ** 2
    ema_var = (1 - alpha) * ema_var + alpha * (diff ** 2)
    ema_mean = ema_mean + (alpha * diff)

    current_std = math.sqrt(ema_var)

    return ema_mean, current_std


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


def trade_hydrogel(state: TradingState, hydrogel:Hydrogel) -> List[Order]:
    order_depth: OrderDepth = state.order_depths['HYDROGEL_PACK']
    orders: List[Order] = []

    if not hydrogel.fair_value:
        return orders

    safe_std = hydrogel.price_std if hydrogel.price_std > 0 else 1e-6
    z_score = (hydrogel.fair_value - hydrogel.price_mean) / safe_std

    if z_score < -hydrogel.z_score_take_thr:
        target_position = hydrogel.limit
    elif z_score > hydrogel.z_score_take_thr:
        target_position = -hydrogel.limit
    elif z_score < 0 and hydrogel.position < 0:
        target_position = 0
    elif z_score > 0 and hydrogel.position > 0:
        target_position = 0
    else:
        target_position = hydrogel.position

    position_diff = round(hydrogel.position - target_position)

    if position_diff > 0 and len(order_depth.buy_orders) != 0:  # SELL
        best_bid = max(order_depth.buy_orders.keys())
        best_bid_amount = order_depth.buy_orders[best_bid]

        size = min(position_diff, best_bid_amount)
        orders.append(Order(hydrogel.name, best_bid, -size))
        hydrogel.posted_sell_volume += size
    elif position_diff < 0 and len(order_depth.sell_orders) != 0:  # BUY
        best_ask = min(order_depth.sell_orders.keys())
        best_ask_amount = -1 * order_depth.sell_orders[best_ask]

        size = min(-position_diff, best_ask_amount)
        orders.append(Order(hydrogel.name, best_ask, size))
        hydrogel.posted_buy_volume += size

    if position_diff == 0 and hydrogel.position == 0:

        # Market making
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        whale_bid_price = None
        if order_depth.buy_orders:
            for price, volume in order_depth.buy_orders.items():
                if 10 <= volume <= 15:
                    whale_bid_price = price
                    break  # Found him!

        whale_ask_price = None
        if order_depth.sell_orders:
            for price, volume in order_depth.sell_orders.items():
                if -15 <= volume <= -10:
                    whale_ask_price = price
                    break  # Found him!

        if best_bid and best_ask and (best_ask - best_bid > 1):
            if whale_bid_price:
                my_mm_bid = whale_bid_price + 1  # penny
            else:
                my_mm_bid = best_bid + 1  # Fallback if Whale isn't detected

            if whale_ask_price:
                my_mm_ask = whale_ask_price - 1  # penny
            else:
                my_mm_ask = best_ask - 1  # Fallback if Whale isn't detected

            if my_mm_bid >= best_ask:
                my_mm_bid = best_ask - 1
            if my_mm_ask <= best_bid:
                my_mm_ask = best_bid + 1

        else:
            my_mm_bid = hydrogel.fair_value - hydrogel.default_thr
            my_mm_ask = hydrogel.fair_value + hydrogel.default_thr

        buy_quantity = hydrogel.limit - (hydrogel.position + hydrogel.posted_buy_volume)
        if buy_quantity > 0:
            orders.append(Order(hydrogel.name, round(my_mm_bid), buy_quantity))  # Buy order

        sell_quantity = hydrogel.limit + (hydrogel.position - hydrogel.posted_sell_volume)
        if sell_quantity > 0:
            orders.append(Order(hydrogel.name, round(my_mm_ask), -sell_quantity))  # Sell order

    return orders


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
        if 'HYDROGEL_PACK' in state.order_depths:
            orders: List[Order] = []

            hydrogel_position = state.position.get('HYDROGEL_PACK', 0)
            hydrogel_fair_value = calc_hydrogel_fair_value(state, previous_state)

            hydrogel = Hydrogel(position=hydrogel_position, fair_value=hydrogel_fair_value)

            ema_mean, ema_std = calc_hydrogel_ema_stats(previous_state, hydrogel)
            hydrogel.price_mean = ema_mean
            hydrogel.price_std = ema_std

            orders.extend(trade_hydrogel(state, hydrogel))

            previous_state['hydrogel_last_price'] = hydrogel_fair_value
            previous_state['hydrogel_ema_mean'] = ema_mean
            previous_state['hydrogel_ema_std'] = ema_std

            result['HYDROGEL_PACK'] = orders

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
