import jsonpickle
import numpy as np
from dataclasses import dataclass
from datamodel import OrderDepth, TradingState, Order
from typing import List


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
class Hydrogel(Product):
    name: str = 'HYDROGEL_PACK'
    limit: int = 200
    take_thr: int = 1
    clear_thr: int = 0
    disregard_thr: int = 1
    default_thr: int = 8
    volume_thr: int = 10
    price_mean: float = 9995
    price_std: float = 35
    z_score_take_thr: float = 1.5
    z_score_clear_thr: float = 0


def calc_hydrogel_fair_value(state: TradingState) -> float:
    previous_price = None
    if state.traderData:
        previous_state = jsonpickle.decode(state.traderData)
        previous_price = previous_state.get('hydrogel_last_price', None)

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


def trade_hydrogel(state: TradingState, hydrogel:Hydrogel) -> List[Order]:
    order_depth: OrderDepth = state.order_depths['HYDROGEL_PACK']
    orders: List[Order] = []

    if not hydrogel.fair_value:
        return orders

    # TODO: Do the rest only when target_position=0 or position=0 or (position_diff=0 and position=0)

    # Position clearance
    position_after_take = hydrogel.position + hydrogel.posted_buy_volume - hydrogel.posted_sell_volume
    fair_for_bid = round(hydrogel.fair_value - hydrogel.clear_thr)
    fair_for_ask = round(hydrogel.fair_value + hydrogel.clear_thr)
    buy_quantity = hydrogel.limit - (hydrogel.position + hydrogel.posted_buy_volume)
    sell_quantity = hydrogel.limit + (hydrogel.position - hydrogel.posted_sell_volume)

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
            orders.append(Order(hydrogel.name, fair_for_ask, -abs(sent_quantity)))
            hydrogel.posted_sell_volume += abs(sent_quantity)

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
            orders.append(Order(hydrogel.name, fair_for_bid, abs(sent_quantity)))
            hydrogel.posted_buy_volume += abs(sent_quantity)

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


class Trader:
    def run(self, state: TradingState):
        conversions = 0
        hydrogel_fair_value = None

        result = {}
        for product_name in state.order_depths:
            position = state.position.get(product_name, 0)
            print(f'{product_name} position: {position}')
            orders: List[Order] = []
            if product_name == 'HYDROGEL_PACK':
                hydrogel_fair_value = calc_hydrogel_fair_value(state)
                hydrogel = Hydrogel(position=position, fair_value=hydrogel_fair_value)
                orders.extend(trade_hydrogel(state, hydrogel))

            result[product_name] = orders
            print('---')

        trader_data = jsonpickle.encode({
            'hydrogel_last_price': hydrogel_fair_value,
        })
        return result, conversions, trader_data