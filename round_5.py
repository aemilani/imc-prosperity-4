import jsonpickle
import math
import numpy as np
from dataclasses import dataclass
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Tuple


@dataclass
class Spread:
    name: str
    product_names: Tuple[str, ...]
    product_weights: Tuple[int, ...]
    price_mean: float = None
    price_std: float = None
    z_score_thr: float = 1.5
    window_size: int = 10000
    fair_value: float = None
    position: int = 0

    def __post_init__(self):
        self.limit = int(10 // np.abs(self.product_weights).max())


@dataclass
class SpreadSleep(Spread):
    name: str = 'SPREAD_SLEEP'
    product_names: Tuple[str, ...] = (
        'SLEEP_POD_SUEDE', 'SLEEP_POD_LAMB_WOOL', 'SLEEP_POD_POLYESTER', 'SLEEP_POD_NYLON', 'SLEEP_POD_COTTON'
    )
    product_weights: Tuple[int, ...] = (1, -1, -1, 1, 0)
    price_mean: float = -1115
    price_std: float = 528


@dataclass
class SpreadMicrochip(Spread):
    name: str = 'SPREAD_MICROCHIP'
    product_names: Tuple[str, ...] = (
        'MICROCHIP_CIRCLE','MICROCHIP_OVAL','MICROCHIP_SQUARE','MICROCHIP_RECTANGLE','MICROCHIP_TRIANGLE'
    )
    product_weights: Tuple[int, ...] = (1, 1, 0, -1, -1)
    price_mean: float = -907
    price_std: float = 630


@dataclass
class SpreadPebbles(Spread):
    name: str = 'SPREAD_PEBBLES'
    product_names: Tuple[str, ...] = (
        'PEBBLES_XS', 'PEBBLES_S', 'PEBBLES_M', 'PEBBLES_L', 'PEBBLES_XL'
    )
    product_weights: Tuple[int, ...] = (1, 1, 1, 1, 1)


@dataclass
class SpreadRobot(Spread):
    name: str = 'SPREAD_ROBOT'
    product_names: Tuple[str, ...] = (
        'ROBOT_VACUUMING', 'ROBOT_MOPPING', 'ROBOT_DISHES', 'ROBOT_LAUNDRY', 'ROBOT_IRONING'
    )
    product_weights: Tuple[int, ...] = (1, 1, 1, 0, 1)
    price_mean: float = 38801
    price_std: float = 386


@dataclass
class SpreadTranslator(Spread):
    name: str = 'SPREAD_TRANSLATOR'
    product_names: Tuple[str, ...] = (
        'TRANSLATOR_SPACE_GRAY', 'TRANSLATOR_ASTRO_BLACK', 'TRANSLATOR_ECLIPSE_CHARCOAL',
        'TRANSLATOR_GRAPHITE_MIST', 'TRANSLATOR_VOID_BLUE'
    )
    product_weights: Tuple[int, ...] = (0, 1, -1, 0, 1)
    price_mean: float = 10630
    price_std: float = 423


@dataclass
class SpreadOxygen(Spread):
    name: str = 'SPREAD_OXYGEN'
    product_names: Tuple[str, ...] = (
        'OXYGEN_SHAKE_MORNING_BREATH', 'OXYGEN_SHAKE_EVENING_BREATH', 'OXYGEN_SHAKE_MINT',
        'OXYGEN_SHAKE_CHOCOLATE', 'OXYGEN_SHAKE_GARLIC'
    )
    product_weights: Tuple[int, ...] = (1, 1, 0, -1, 1)
    price_mean: float = 21705
    price_std: float = 559


@dataclass
class SpreadSnackpack(Spread):
    name: str = 'SPREAD_SNACKPACK'
    product_names: Tuple[str, ...] = (
        'SNACKPACK_CHOCOLATE', 'SNACKPACK_VANILLA', 'SNACKPACK_PISTACHIO', 'SNACKPACK_STRAWBERRY', 'SNACKPACK_RASPBERRY'
    )
    product_weights: Tuple[int, ...] = (0, 0, 0, 0, 1)
    price_mean: float = 10117
    price_std: float = 186


def get_spread_position(state: TradingState, spr: Spread) -> int:
    rs = []
    anchor_sign = 0

    for name, w in zip(spr.product_names, spr.product_weights):
        if w == 0:
            continue

        pos = state.position.get(name, 0)
        rs.append(abs(pos) // abs(w))

        if anchor_sign == 0 and pos != 0:
            # If position and weight have the same sign, we are Long (+1)
            # If position and weight have opposite signs, we are Short (-1)
            anchor_sign = 1 if (pos * w > 0) else -1

    if not rs:
        return 0

    r = min(rs)

    return r * anchor_sign


def get_spread_products_orders(state: TradingState, spr: Spread) -> Tuple[List[int], List[int], List[int], List[int]]:
    order_depths: Dict[str, OrderDepth] = state.order_depths
    best_bids, best_asks, best_bid_volumes, best_ask_volumes = [], [], [], []

    for product, w in zip(spr.product_names, spr.product_weights):
        if w == 0:
            best_bids.append(0)
            best_asks.append(0)
            best_bid_volumes.append(0)
            best_ask_volumes.append(0)
            continue

        best_bid = max(order_depths[product].buy_orders.keys()) if order_depths[product].buy_orders else None
        best_ask = min(order_depths[product].sell_orders.keys()) if order_depths[product].sell_orders else None

        if not best_bid or not best_ask:
            return [], [], [], []

        best_bids.append(best_bid)
        best_asks.append(best_ask)
        best_bid_volumes.append(order_depths[product].buy_orders[best_bid])
        best_ask_volumes.append(-order_depths[product].sell_orders[best_ask])

    return best_bids, best_asks, best_bid_volumes, best_ask_volumes


def get_spread_order_depth(state: TradingState, spr: Spread) -> OrderDepth:
    best_bids, best_asks, best_bid_volumes, best_ask_volumes = get_spread_products_orders(state, spr)

    spread_order_depth = OrderDepth()
    if not best_bids or not best_asks:
        return spread_order_depth

    spread_bid, spread_ask = 0, 0
    spread_bid_volumes, spread_ask_volumes = [], []
    for bid, ask, bid_vol, ask_vol, w in zip(best_bids, best_asks, best_bid_volumes, best_ask_volumes,
                                             spr.product_weights):
        if w > 0:
            spread_bid += bid * w
            spread_ask += ask * w
            spread_bid_volumes.append(bid_vol // abs(w))
            spread_ask_volumes.append(ask_vol // abs(w))
        if w < 0:
            spread_bid += ask * w
            spread_ask += bid * w
            spread_bid_volumes.append(ask_vol // abs(w))
            spread_ask_volumes.append(bid_vol // abs(w))

    spread_bid_volume = min(spread_bid_volumes)
    spread_ask_volume = min(spread_ask_volumes)
    spread_order_depth.buy_orders[spread_bid] = spread_bid_volume
    spread_order_depth.sell_orders[spread_ask] = -spread_ask_volume

    return spread_order_depth


def get_spread_mid_price(state: TradingState, spr: Spread) -> float | None:
    spread_order_depth: OrderDepth = get_spread_order_depth(state, spr)
    if not spread_order_depth.buy_orders or not spread_order_depth.sell_orders:
        return None
    return (max(spread_order_depth.buy_orders.keys()) + min(spread_order_depth.sell_orders.keys())) / 2


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


def trade_mean_reversion(state: TradingState, spr: Spread) -> Dict[str, List[Order]]:
    orders: Dict[str, List[Order]] = {key: [] for key in spr.product_names}

    if not spr.fair_value:
        return orders

    order_depth = get_spread_order_depth(state, spr)
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return orders

    best_bid = max(order_depth.buy_orders.keys())
    best_ask = min(order_depth.sell_orders.keys())
    best_bid_size = abs(order_depth.buy_orders[best_bid])
    best_ask_size = abs(order_depth.sell_orders[best_ask])

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

    product_names = spr.product_names
    best_bids, best_asks, _, _ = get_spread_products_orders(state, spr)
    if not best_bids or not best_asks:
        return orders

    new_target_spread_position = spr.position
    if position_diff > 0:  # We want to sell spreads
        new_target_spread_position -= min(position_diff, best_bid_size)
    elif position_diff < 0:  # We want to buy spreads
        new_target_spread_position += min(-position_diff, best_ask_size)

    for product, w, bid, ask in zip(product_names, spr.product_weights, best_bids, best_asks):
        if w == 0:
            continue

        target_leg_pos = new_target_spread_position * w
        current_leg_pos = state.position.get(product, 0)
        leg_order_qty = target_leg_pos - current_leg_pos

        if leg_order_qty < 0:
            orders[product].append(Order(product, round(bid), int(leg_order_qty)))  # Sell
        elif leg_order_qty > 0:
            orders[product].append(Order(product, round(ask), int(leg_order_qty)))  # Buy

    return orders


def trade_pebbles(state: TradingState, pbl: SpreadPebbles) -> Dict[str, List[Order]]:
    orders: Dict[str, List[Order]] = {key: [] for key in pbl.product_names}

    if not pbl.fair_value:
        return orders

    order_depth = get_spread_order_depth(state, pbl)
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return orders

    best_bid = max(order_depth.buy_orders.keys())
    best_ask = min(order_depth.sell_orders.keys())
    best_bid_size = abs(order_depth.buy_orders[best_bid])
    best_ask_size = abs(order_depth.sell_orders[best_ask])

    if pbl.fair_value >= 50013:
        target_position = -pbl.limit
    elif pbl.fair_value <= 49984:
        target_position = pbl.limit
    else:
        target_position = pbl.position

    position_diff = round(pbl.position - target_position)

    product_names = pbl.product_names
    best_bids, best_asks, _, _ = get_spread_products_orders(state, pbl)
    if not best_bids or not best_asks:
        return orders

    new_target_spread_position = pbl.position
    if position_diff > 0:  # We want to sell spreads
        new_target_spread_position -= min(position_diff, best_bid_size)
    elif position_diff < 0:  # We want to buy spreads
        new_target_spread_position += min(-position_diff, best_ask_size)

    for product, w, bid, ask in zip(product_names, pbl.product_weights, best_bids, best_asks):
        if w == 0:
            continue

        target_leg_pos = new_target_spread_position * w
        current_leg_pos = state.position.get(product, 0)
        leg_order_qty = target_leg_pos - current_leg_pos

        if leg_order_qty < 0:
            orders[product].append(Order(product, round(bid), int(leg_order_qty)))  # Sell
        elif leg_order_qty > 0:
            orders[product].append(Order(product, round(ask), int(leg_order_qty)))  # Buy

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

        spreads = [SpreadSleep(), SpreadMicrochip(), SpreadRobot(),
                   SpreadTranslator(), SpreadOxygen(), SpreadSnackpack()]

        result = {}
        for spr in spreads:
            if all(name in state.order_depths for name in spr.product_names):
                position = get_spread_position(state, spr)
                mid_price = get_spread_mid_price(state, spr)
                spr.position = position
                spr.fair_value = mid_price

                ema_mean, ema_std = calc_ema_stats(previous_state, spr)
                spr.price_mean = ema_mean
                spr.price_std = ema_std
                previous_state[f'{spr.name}_ema_mean'] = ema_mean
                previous_state[f'{spr.name}_ema_std'] = ema_std

                spr_orders: Dict[str, List[Order]] = trade_mean_reversion(state, spr)
                for product_name, orders in spr_orders.items():
                    result[product_name] = orders

        spr = SpreadPebbles()
        if all(name in state.order_depths for name in spr.product_names):
            position = get_spread_position(state, spr)
            mid_price = get_spread_mid_price(state, spr)
            spr.position = position
            spr.fair_value = mid_price

            spr_orders: Dict[str, List[Order]] = trade_pebbles(state, spr)
            for product_name, orders in spr_orders.items():
                result[product_name] = orders

        trader_data = jsonpickle.encode(previous_state)
        return result, conversions, trader_data
