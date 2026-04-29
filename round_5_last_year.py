import jsonpickle
import numpy as np
from dataclasses import dataclass
from datamodel import OrderDepth, TradingState, Order, ConversionObservation
from typing import List, Dict, Tuple
from math import log, sqrt
from statistics import NormalDist


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
class Croissants(Product):
    name: str = 'CROISSANTS'
    limit: int = 250


@dataclass
class Jams(Product):
    name: str = 'JAMS'
    limit: int = 350


@dataclass
class Djembes(Product):
    name: str = 'DJEMBES'
    limit: int = 60


@dataclass
class Basket1(Product):
    name: str = 'PICNIC_BASKET1'
    limit: int = 60


@dataclass
class Basket2(Product):
    name: str = 'PICNIC_BASKET2'
    limit: int = 100


@dataclass
class Spread(Product):
    name: str = 'SPREAD'
    limit: int = 60
    product_names: Tuple[str] = ('PICNIC_BASKET1', 'PICNIC_BASKET2', 'CROISSANTS', 'JAMS', 'DJEMBES')
    product_weights: Tuple[int] = (1, -1, -2, -1, -1)  # Basket1, Basket2, Croissants, Jams, Djembes
    mean: float = -202.3
    std: float = 83.9


def get_spread_position(state: TradingState) -> int:
    return state.position.get('PICNIC_BASKET1', 0)


def get_target_spread_position_size(spread: Spread) -> int:
    zscore = (spread.fair_value - spread.mean) / spread.std

    thr = 0.8
    if zscore < -thr:
        target_position = spread.limit
    elif zscore > thr:
        target_position = -spread.limit
    else:
        target_position = spread.position

    return target_position


def get_spread_products_orders(state: TradingState) -> Tuple[List[int], List[int], List[int], List[int]]:
    order_depths: Dict[str, OrderDepth] = state.order_depths
    products = [Basket1(), Basket2(), Croissants(), Jams(), Djembes()]

    best_bids, best_asks, best_bid_volumes, best_ask_volumes = [], [], [], []
    for product in products:
        best_bid = (max(order_depths[product.name].buy_orders.keys())
                    if order_depths[product.name].buy_orders else 0)
        best_ask = (min(order_depths[product.name].sell_orders.keys())
                    if order_depths[product.name].sell_orders else float("inf"))
        best_bid_volume = order_depths[product.name].buy_orders[best_bid]
        best_ask_volume = -order_depths[product.name].sell_orders[best_ask]
        best_bids.append(best_bid)
        best_asks.append(best_ask)
        best_bid_volumes.append(best_bid_volume)
        best_ask_volumes.append(best_ask_volume)

    return best_bids, best_asks, best_bid_volumes, best_ask_volumes


def get_spread_order_depth(state: TradingState) -> OrderDepth:
    best_bids, best_asks, best_bid_volumes, best_ask_volumes = get_spread_products_orders(state)

    spread = Spread()
    spread_order_depth = OrderDepth()
    product_weights = spread.product_weights
    spread_bid, spread_ask = 0, 0
    spread_bid_volumes, spread_ask_volumes = [], []
    for bid, ask, bid_vol, ask_vol, w in zip(best_bids, best_asks, best_bid_volumes, best_ask_volumes, product_weights):
        if w > 0:
            spread_bid += bid * w
            spread_ask += ask * w
            spread_bid_volumes.append(abs(bid_vol // w))
            spread_ask_volumes.append(abs(ask_vol // w))
        if w < 0:
            spread_bid += ask * w
            spread_ask += bid * w
            spread_bid_volumes.append(abs(ask_vol // w))
            spread_ask_volumes.append(abs(bid_vol // w))
    spread_bid_volume = min(spread_bid_volumes)
    spread_ask_volume = min(spread_ask_volumes)
    spread_order_depth.buy_orders[spread_bid] = spread_bid_volume
    spread_order_depth.sell_orders[spread_ask] = -spread_ask_volume

    return spread_order_depth


def get_spread_mid_price(state: TradingState) -> float:
    spread_order_depth: OrderDepth = get_spread_order_depth(state)
    return (max(spread_order_depth.buy_orders.keys()) + min(spread_order_depth.sell_orders.keys())) / 2


def trade_spread(state: TradingState, spread: Spread) -> Dict[str, List[Order]]:
    order_depth = get_spread_order_depth(state)
    best_bid = max(order_depth.buy_orders.keys())
    best_ask = min(order_depth.sell_orders.keys())
    best_bid_size = abs(order_depth.buy_orders[best_bid])
    best_ask_size = abs(order_depth.sell_orders[best_ask])
    current_position = spread.position
    target_position = get_target_spread_position_size(spread)
    position_diff = round(current_position - target_position)

    products = [Basket1(), Basket2(), Croissants(), Jams(), Djembes()]
    best_bids, best_asks, _, _ = get_spread_products_orders(state)

    orders: Dict[str, List[Order]] = {key: [] for key in spread.product_names}
    if position_diff > 0:  # sell spread
        size = min(position_diff, best_bid_size)
        for product, w, bid, ask in zip(products, spread.product_weights, best_bids, best_asks):
            if w > 0:
                orders[product.name].append(Order(product.name, round(bid), -abs(size * w)))  # sell product
            else:
                orders[product.name].append(Order(product.name, round(ask), abs(size * w)))  # buy product
    elif position_diff < 0:  # buy spread
        size = min(-position_diff, best_ask_size)
        for product, w, bid, ask in zip(products, spread.product_weights, best_bids, best_asks):
            if w > 0:
                orders[product.name].append(Order(product.name, round(ask), abs(size * w)))  # buy product
            else:
                orders[product.name].append(Order(product.name, round(bid), -abs(size * w)))  # sell product

    return orders


class Trader:
    def run(self, state: TradingState):
        conversions = 0
        trader_data = ""

        result = {}
        if 'PICNIC_BASKET1' in state.order_depths:
            spread_position = get_spread_position(state)
            spread_mid_price = get_spread_mid_price(state)

            spread = Spread(position=spread_position, fair_value=spread_mid_price)
            print(f'{spread.name} position: {spread.position}')

            spread_orders: Dict[str, List[Order]] = trade_spread(state, spread)
            for product_name, orders in spread_orders.items():
                result[product_name] = orders

        return result, conversions, trader_data