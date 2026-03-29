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
class Emeralds(Product):
    name: str = 'EMERALDS'
    limit: int = 80
    fair_value: float = 10000
    take_thr: int = 1
    clear_thr: int = 0
    disregard_thr: int = 2
    join_thr: int = 4
    default_thr: int = 7
    soft_pos_limit: int = 60


@dataclass
class Tomatoes(Product):
    name: str = 'TOMATOES'
    limit: int = 80
    take_thr: int = 1
    clear_thr: int = 0
    disregard_thr: int = 1
    join_thr: int = 3
    default_thr: int = 6
    volume_thr: int = 15


def calc_tomatoes_fair_value(state: TradingState) -> float:
    previous_price = None
    if state.traderData:
        previous_state = jsonpickle.decode(state.traderData)
        previous_price = previous_state.get('tomatoes_last_price', None)

    order_depth: OrderDepth = state.order_depths['TOMATOES']

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
            next_logr = curr_logr * -0.21  # mean-reversion param
            return fair_value * np.exp(next_logr)
    else:
        return previous_price


def trade_emeralds(state: TradingState, emeralds: Emeralds) -> List[Order]:
    order_depth: OrderDepth = state.order_depths['EMERALDS']
    orders: List[Order] = []

    # Market taking
    if len(order_depth.sell_orders) != 0:
        best_ask = min(order_depth.sell_orders.keys())
        best_ask_amount = -1 * order_depth.sell_orders[best_ask]

        if best_ask <= emeralds.fair_value - emeralds.take_thr:
            quantity = min(
                best_ask_amount, emeralds.limit - emeralds.position
            )  # max amt to buy
            if quantity > 0:
                orders.append(Order(emeralds.name, best_ask, quantity))
                emeralds.posted_buy_volume += quantity
                order_depth.sell_orders[best_ask] += quantity
                if order_depth.sell_orders[best_ask] == 0:
                    del order_depth.sell_orders[best_ask]
    if len(order_depth.buy_orders) != 0:
        best_bid = max(order_depth.buy_orders.keys())
        best_bid_amount = order_depth.buy_orders[best_bid]

        if best_bid >= emeralds.fair_value + emeralds.take_thr:
            quantity = min(
                best_bid_amount, emeralds.limit + emeralds.position
            )  # should be the max we can sell
            if quantity > 0:
                orders.append(Order(emeralds.name, best_bid, -1 * quantity))
                emeralds.posted_sell_volume += quantity
                order_depth.buy_orders[best_bid] -= quantity
                if order_depth.buy_orders[best_bid] == 0:
                    del order_depth.buy_orders[best_bid]

    # Position clearance
    position_after_take = emeralds.position + emeralds.posted_buy_volume - emeralds.posted_sell_volume
    fair_for_bid = round(emeralds.fair_value - emeralds.clear_thr)
    fair_for_ask = round(emeralds.fair_value + emeralds.clear_thr)
    buy_quantity = emeralds.limit - (emeralds.position + emeralds.posted_buy_volume)
    sell_quantity = emeralds.limit + (emeralds.position - emeralds.posted_sell_volume)

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
            orders.append(Order(emeralds.name, fair_for_ask, -abs(sent_quantity)))
            emeralds.posted_sell_volume += abs(sent_quantity)

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
            orders.append(Order(emeralds.name, fair_for_bid, abs(sent_quantity)))
            emeralds.posted_buy_volume += abs(sent_quantity)

    # Market making
    asks_above_fair = [
        price
        for price in order_depth.sell_orders.keys()
        if price > emeralds.fair_value + emeralds.disregard_thr
    ]
    bids_below_fair = [
        price
        for price in order_depth.buy_orders.keys()
        if price < emeralds.fair_value - emeralds.disregard_thr
    ]
    best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
    best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

    ask = round(emeralds.fair_value + emeralds.default_thr)
    if best_ask_above_fair is not None:
        if abs(best_ask_above_fair - emeralds.fair_value) <= emeralds.join_thr:
            ask = best_ask_above_fair  # join
        else:
            ask = best_ask_above_fair - 1  # penny

    bid = round(emeralds.fair_value - emeralds.default_thr)
    if best_bid_below_fair is not None:
        if abs(emeralds.fair_value - best_bid_below_fair) <= emeralds.join_thr:
            bid = best_bid_below_fair
        else:
            bid = best_bid_below_fair + 1

    if emeralds.position > emeralds.soft_pos_limit:
        ask -= 1
    elif emeralds.position < -1 * emeralds.soft_pos_limit:
        bid += 1

    buy_quantity = emeralds.limit - (emeralds.position + emeralds.posted_buy_volume)
    if buy_quantity > 0:
        orders.append(Order(emeralds.name, round(bid), buy_quantity))  # Buy order

    sell_quantity = emeralds.limit + (emeralds.position - emeralds.posted_sell_volume)
    if sell_quantity > 0:
        orders.append(Order(emeralds.name, round(ask), -sell_quantity))  # Sell order

    return orders


def trade_tomatoes(state: TradingState, tomatoes:Tomatoes) -> List[Order]:
    order_depth: OrderDepth = state.order_depths['TOMATOES']
    orders: List[Order] = []

    # Market taking
    if len(order_depth.sell_orders) != 0:
        best_ask = min(order_depth.sell_orders.keys())
        best_ask_amount = -1 * order_depth.sell_orders[best_ask]

        if abs(best_ask_amount) <= tomatoes.volume_thr:
            if best_ask <= tomatoes.fair_value - tomatoes.take_thr:
                quantity = min(
                    best_ask_amount, tomatoes.limit - tomatoes.position
                )  # max amt to buy
                if quantity > 0:
                    orders.append(Order(tomatoes.name, best_ask, quantity))
                    tomatoes.posted_buy_volume += quantity
                    order_depth.sell_orders[best_ask] += quantity
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]
    if len(order_depth.buy_orders) != 0:
        best_bid = max(order_depth.buy_orders.keys())
        best_bid_amount = order_depth.buy_orders[best_bid]

        if abs(best_bid_amount) <= tomatoes.volume_thr:
            if best_bid >= tomatoes.fair_value + tomatoes.take_thr:
                quantity = min(
                    best_bid_amount, tomatoes.limit + tomatoes.position
                )  # should be the max we can sell
                if quantity > 0:
                    orders.append(Order(tomatoes.name, best_bid, -1 * quantity))
                    tomatoes.posted_sell_volume += quantity
                    order_depth.buy_orders[best_bid] -= quantity
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]

    # Position clearance
    position_after_take = tomatoes.position + tomatoes.posted_buy_volume - tomatoes.posted_sell_volume
    fair_for_bid = round(tomatoes.fair_value - tomatoes.clear_thr)
    fair_for_ask = round(tomatoes.fair_value + tomatoes.clear_thr)
    buy_quantity = tomatoes.limit - (tomatoes.position + tomatoes.posted_buy_volume)
    sell_quantity = tomatoes.limit + (tomatoes.position - tomatoes.posted_sell_volume)

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
            orders.append(Order(tomatoes.name, fair_for_ask, -abs(sent_quantity)))
            tomatoes.posted_sell_volume += abs(sent_quantity)

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
            orders.append(Order(tomatoes.name, fair_for_bid, abs(sent_quantity)))
            tomatoes.posted_buy_volume += abs(sent_quantity)

    # Market making
    asks_above_fair = [
        price
        for price in order_depth.sell_orders.keys()
        if price > tomatoes.fair_value + tomatoes.disregard_thr
    ]
    bids_below_fair = [
        price
        for price in order_depth.buy_orders.keys()
        if price < tomatoes.fair_value - tomatoes.disregard_thr
    ]
    best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
    best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

    ask = round(tomatoes.fair_value + tomatoes.default_thr)
    if best_ask_above_fair is not None:
        if abs(best_ask_above_fair - tomatoes.fair_value) <= tomatoes.join_thr:
            ask = best_ask_above_fair  # join
        else:
            ask = best_ask_above_fair - 1  # penny

    bid = round(tomatoes.fair_value - tomatoes.default_thr)
    if best_bid_below_fair is not None:
        if abs(tomatoes.fair_value - best_bid_below_fair) <= tomatoes.join_thr:
            bid = best_bid_below_fair
        else:
            bid = best_bid_below_fair + 1

    buy_quantity = tomatoes.limit - (tomatoes.position + tomatoes.posted_buy_volume)
    if buy_quantity > 0:
        orders.append(Order(tomatoes.name, round(bid), buy_quantity))  # Buy order

    sell_quantity = tomatoes.limit + (tomatoes.position - tomatoes.posted_sell_volume)
    if sell_quantity > 0:
        orders.append(Order(tomatoes.name, round(ask), -sell_quantity))  # Sell order

    return orders


class Trader:
    def run(self, state: TradingState):
        conversions = 0
        tomatoes_fair_value = None

        result = {}
        for product_name in state.order_depths:
            position = state.position.get(product_name, 0)
            print(f'{product_name} position: {position}')
            orders: List[Order] = []
            if product_name == 'EMERALDS':
                product = Emeralds(position=position)
                orders.extend(trade_emeralds(state, product))
            if product_name == 'TOMATOES':
                tomatoes_fair_value = calc_tomatoes_fair_value(state)
                product = Tomatoes(position=position, fair_value=tomatoes_fair_value)
                orders.extend(trade_tomatoes(state, product))

            result[product_name] = orders
            print('---')

        trader_data = jsonpickle.encode({
            'tomatoes_last_price': tomatoes_fair_value,
        })
        return result, conversions, trader_data