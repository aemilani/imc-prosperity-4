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
class Pepper(Product):
    name: str = 'INTARIAN_PEPPER_ROOT'
    limit: int = 80
    take_thr: int = 1
    clear_thr: int = 0
    disregard_thr: int = 2
    join_thr: int = 3
    default_thr: int = 5
    soft_pos_limit: int = 60


@dataclass
class Osmium(Product):
    name: str = 'ASH_COATED_OSMIUM'
    limit: int = 80
    take_thr: int = 1
    clear_thr: int = 0
    disregard_thr: int = 1
    join_thr: int = 3
    default_thr: int = 7
    volume_thr: int = 20


# def calc_pepper_fair_value(state: TradingState, day=0) -> float:
#     return 10000 + (state.timestamp + (day + 2) * 1e6) / 1000


def calc_pepper_fair_value(state: TradingState) -> float:
    previous_price = None
    if state.traderData:
        previous_state = jsonpickle.decode(state.traderData)
        previous_price = previous_state.get('pepper_last_price', None)

    order_depth: OrderDepth = state.order_depths['INTARIAN_PEPPER_ROOT']

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
            next_logr = curr_logr * -0.47  # mean-reversion param
            return fair_value * np.exp(next_logr)
    else:
        return previous_price


def calc_osmium_fair_value(state: TradingState) -> float:
    previous_price = None
    if state.traderData:
        previous_state = jsonpickle.decode(state.traderData)
        previous_price = previous_state.get('osmium_last_price', None)

    order_depth: OrderDepth = state.order_depths['ASH_COATED_OSMIUM']

    if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())

        filtered_asks = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 20]
        filtered_bids = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 20]
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
            next_logr = curr_logr * -0.42  # mean-reversion param
            return fair_value * np.exp(next_logr)
    else:
        return previous_price


def trade_pepper(state: TradingState, pepper: Pepper) -> List[Order]:
    order_depth: OrderDepth = state.order_depths['INTARIAN_PEPPER_ROOT']
    orders: List[Order] = []

    if not pepper.fair_value:
        return orders

    # Market taking
    if len(order_depth.sell_orders) != 0:
        best_ask = min(order_depth.sell_orders.keys())
        best_ask_amount = -1 * order_depth.sell_orders[best_ask]

        if best_ask <= pepper.fair_value - pepper.take_thr:
            quantity = min(
                best_ask_amount, pepper.limit - pepper.position
            )  # max amt to buy
            if quantity > 0:
                orders.append(Order(pepper.name, best_ask, quantity))
                pepper.posted_buy_volume += quantity
                order_depth.sell_orders[best_ask] += quantity
                if order_depth.sell_orders[best_ask] == 0:
                    del order_depth.sell_orders[best_ask]
    if len(order_depth.buy_orders) != 0:
        best_bid = max(order_depth.buy_orders.keys())
        best_bid_amount = order_depth.buy_orders[best_bid]

        if best_bid >= pepper.fair_value + pepper.take_thr:
            quantity = min(
                best_bid_amount, pepper.limit + pepper.position
            )  # should be the max we can sell
            if quantity > 0:
                orders.append(Order(pepper.name, best_bid, -1 * quantity))
                pepper.posted_sell_volume += quantity
                order_depth.buy_orders[best_bid] -= quantity
                if order_depth.buy_orders[best_bid] == 0:
                    del order_depth.buy_orders[best_bid]

    # Position clearance
    position_after_take = pepper.position + pepper.posted_buy_volume - pepper.posted_sell_volume
    fair_for_bid = round(pepper.fair_value - pepper.clear_thr)
    fair_for_ask = round(pepper.fair_value + pepper.clear_thr)
    buy_quantity = pepper.limit - (pepper.position + pepper.posted_buy_volume)
    sell_quantity = pepper.limit + (pepper.position - pepper.posted_sell_volume)

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
            orders.append(Order(pepper.name, fair_for_ask, -abs(sent_quantity)))
            pepper.posted_sell_volume += abs(sent_quantity)

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
            orders.append(Order(pepper.name, fair_for_bid, abs(sent_quantity)))
            pepper.posted_buy_volume += abs(sent_quantity)

    # Market making
    asks_above_fair = [
        price
        for price in order_depth.sell_orders.keys()
        if price > pepper.fair_value + pepper.disregard_thr
    ]
    bids_below_fair = [
        price
        for price in order_depth.buy_orders.keys()
        if price < pepper.fair_value - pepper.disregard_thr
    ]
    best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
    best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

    ask = round(pepper.fair_value + pepper.default_thr)
    if best_ask_above_fair is not None:
        if abs(best_ask_above_fair - pepper.fair_value) <= pepper.join_thr:
            ask = best_ask_above_fair  # join
        else:
            ask = best_ask_above_fair - 1  # penny

    bid = round(pepper.fair_value - pepper.default_thr)
    if best_bid_below_fair is not None:
        if abs(pepper.fair_value - best_bid_below_fair) <= pepper.join_thr:
            bid = best_bid_below_fair
        else:
            bid = best_bid_below_fair + 1

    if pepper.position > pepper.soft_pos_limit:
        ask -= 1
    elif pepper.position < -1 * pepper.soft_pos_limit:
        bid += 1

    buy_quantity = pepper.limit - (pepper.position + pepper.posted_buy_volume)
    if buy_quantity > 0:
        orders.append(Order(pepper.name, round(bid), buy_quantity))  # Buy order

    sell_quantity = pepper.limit + (pepper.position - pepper.posted_sell_volume)
    if sell_quantity > 0:
        orders.append(Order(pepper.name, round(ask), -sell_quantity))  # Sell order

    return orders


def trade_osmium(state: TradingState, osmium:Osmium) -> List[Order]:
    order_depth: OrderDepth = state.order_depths['ASH_COATED_OSMIUM']
    orders: List[Order] = []

    if not osmium.fair_value:
        return orders

    # Market taking
    if len(order_depth.sell_orders) != 0:
        best_ask = min(order_depth.sell_orders.keys())
        best_ask_amount = -1 * order_depth.sell_orders[best_ask]

        if abs(best_ask_amount) <= osmium.volume_thr:
            if best_ask <= osmium.fair_value - osmium.take_thr:
                quantity = min(
                    best_ask_amount, osmium.limit - osmium.position
                )  # max amt to buy
                if quantity > 0:
                    orders.append(Order(osmium.name, best_ask, quantity))
                    osmium.posted_buy_volume += quantity
                    order_depth.sell_orders[best_ask] += quantity
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]
    if len(order_depth.buy_orders) != 0:
        best_bid = max(order_depth.buy_orders.keys())
        best_bid_amount = order_depth.buy_orders[best_bid]

        if abs(best_bid_amount) <= osmium.volume_thr:
            if best_bid >= osmium.fair_value + osmium.take_thr:
                quantity = min(
                    best_bid_amount, osmium.limit + osmium.position
                )  # should be the max we can sell
                if quantity > 0:
                    orders.append(Order(osmium.name, best_bid, -1 * quantity))
                    osmium.posted_sell_volume += quantity
                    order_depth.buy_orders[best_bid] -= quantity
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]

    # Position clearance
    position_after_take = osmium.position + osmium.posted_buy_volume - osmium.posted_sell_volume
    fair_for_bid = round(osmium.fair_value - osmium.clear_thr)
    fair_for_ask = round(osmium.fair_value + osmium.clear_thr)
    buy_quantity = osmium.limit - (osmium.position + osmium.posted_buy_volume)
    sell_quantity = osmium.limit + (osmium.position - osmium.posted_sell_volume)

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
            orders.append(Order(osmium.name, fair_for_ask, -abs(sent_quantity)))
            osmium.posted_sell_volume += abs(sent_quantity)

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
            orders.append(Order(osmium.name, fair_for_bid, abs(sent_quantity)))
            osmium.posted_buy_volume += abs(sent_quantity)

    # Market making
    asks_above_fair = [
        price
        for price in order_depth.sell_orders.keys()
        if price > osmium.fair_value + osmium.disregard_thr
    ]
    bids_below_fair = [
        price
        for price in order_depth.buy_orders.keys()
        if price < osmium.fair_value - osmium.disregard_thr
    ]
    best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
    best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

    ask = round(osmium.fair_value + osmium.default_thr)
    if best_ask_above_fair is not None:
        if abs(best_ask_above_fair - osmium.fair_value) <= osmium.join_thr:
            ask = best_ask_above_fair  # join
        else:
            ask = best_ask_above_fair - 1  # penny

    bid = round(osmium.fair_value - osmium.default_thr)
    if best_bid_below_fair is not None:
        if abs(osmium.fair_value - best_bid_below_fair) <= osmium.join_thr:
            bid = best_bid_below_fair
        else:
            bid = best_bid_below_fair + 1

    buy_quantity = osmium.limit - (osmium.position + osmium.posted_buy_volume)
    if buy_quantity > 0:
        orders.append(Order(osmium.name, round(bid), buy_quantity))  # Buy order

    sell_quantity = osmium.limit + (osmium.position - osmium.posted_sell_volume)
    if sell_quantity > 0:
        orders.append(Order(osmium.name, round(ask), -sell_quantity))  # Sell order

    return orders


class Trader:
    def run(self, state: TradingState):
        conversions = 0
        pepper_fair_value = None
        osmium_fair_value = None

        result = {}
        for product_name in state.order_depths:
            position = state.position.get(product_name, 0)
            print(f'{product_name} position: {position}')
            orders: List[Order] = []
            if product_name == 'INTARIAN_PEPPER_ROOT':
                pepper_fair_value = calc_pepper_fair_value(state)
                product = Pepper(position=position, fair_value=pepper_fair_value)
                orders.extend(trade_pepper(state, product))
            if product_name == 'ASH_COATED_OSMIUM':
                osmium_fair_value = calc_osmium_fair_value(state)
                product = Osmium(position=position, fair_value=osmium_fair_value)
                orders.extend(trade_osmium(state, product))

            result[product_name] = orders
            print('---')

        trader_data = jsonpickle.encode({
            'pepper_last_price': pepper_fair_value,
            'osmium_last_price': osmium_fair_value,
        })
        return result, conversions, trader_data