import jsonpickle
import numpy as np
from dataclasses import dataclass
from datamodel import OrderDepth, TradingState, Order
from typing import List


# TODO: Update mean and std of Hydrogel price based on each new price data


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
    join_thr: int = 3
    default_thr: int = 8
    volume_thr: int = 20
    soft_pos_limit: int = 180
    price_mean: float = 9990
    price_std: float = 32
    z_score_take_thr: float = 1
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
            next_logr = curr_logr * -0.03  # mean-reversion param
            return fair_value * np.exp(next_logr)
    else:
        return previous_price


def trade_hydrogel(state: TradingState, hydrogel:Hydrogel) -> List[Order]:
    order_depth: OrderDepth = state.order_depths['HYDROGEL_PACK']
    orders: List[Order] = []

    if not hydrogel.fair_value:
        return orders

    z_score = (hydrogel.fair_value - hydrogel.price_mean) / hydrogel.price_std

    # Market taking
    if len(order_depth.sell_orders) != 0:  # BUY
        best_ask = min(order_depth.sell_orders.keys())
        best_ask_amount = -1 * order_depth.sell_orders[best_ask]

        if abs(best_ask_amount) <= hydrogel.volume_thr:
            standard_take = best_ask <= hydrogel.fair_value - hydrogel.take_thr

            dynamic_premium = 0
            if z_score < -hydrogel.z_score_take_thr:
                excess_z = abs(z_score) - hydrogel.z_score_take_thr
                dynamic_premium = int(excess_z * 8)

            mean_reversion_trade = (z_score < -hydrogel.z_score_take_thr) and (
                    best_ask <= hydrogel.fair_value + dynamic_premium
            )

            mean_reversion_clear = (
                    z_score < hydrogel.z_score_clear_thr
                    and hydrogel.position < 0
                    and best_ask <= hydrogel.fair_value + 1
            )

            if standard_take or mean_reversion_trade or mean_reversion_clear:
                quantity = min(
                    best_ask_amount, hydrogel.limit - hydrogel.position
                )  # max amt to buy
                if quantity > 0:
                    orders.append(Order(hydrogel.name, best_ask, quantity))
                    hydrogel.posted_buy_volume += quantity
                    order_depth.sell_orders[best_ask] += quantity
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]

    if len(order_depth.buy_orders) != 0:  # SELL
        best_bid = max(order_depth.buy_orders.keys())
        best_bid_amount = order_depth.buy_orders[best_bid]

        if abs(best_bid_amount) <= hydrogel.volume_thr:
            standard_take = best_bid >= hydrogel.fair_value + hydrogel.take_thr

            dynamic_premium = 0
            if z_score > hydrogel.z_score_take_thr:
                excess_z = abs(z_score) - hydrogel.z_score_take_thr
                dynamic_premium = int(excess_z * 8)

            mean_reversion_trade = (z_score > hydrogel.z_score_take_thr) and (
                    best_bid >= hydrogel.fair_value - dynamic_premium
            )

            mean_reversion_clear = (
                    z_score > -hydrogel.z_score_clear_thr
                    and hydrogel.position > 0
                    and best_bid >= hydrogel.fair_value - 1
            )

            if standard_take or mean_reversion_trade or mean_reversion_clear:
                quantity = min(
                    best_bid_amount, hydrogel.limit + hydrogel.position
                )  # should be the max we can sell
                if quantity > 0:
                    orders.append(Order(hydrogel.name, best_bid, -1 * quantity))
                    hydrogel.posted_sell_volume += quantity
                    order_depth.buy_orders[best_bid] -= quantity
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]

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
    asks_above_fair = [
        price
        for price in order_depth.sell_orders.keys()
        if price > hydrogel.fair_value + hydrogel.disregard_thr
    ]
    bids_below_fair = [
        price
        for price in order_depth.buy_orders.keys()
        if price < hydrogel.fair_value - hydrogel.disregard_thr
    ]
    best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
    best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

    ask = round(hydrogel.fair_value + hydrogel.default_thr)
    if best_ask_above_fair is not None:
        if abs(best_ask_above_fair - hydrogel.fair_value) <= hydrogel.join_thr:
            ask = best_ask_above_fair  # join
        else:
            ask = best_ask_above_fair - 1  # penny

    bid = round(hydrogel.fair_value - hydrogel.default_thr)
    if best_bid_below_fair is not None:
        if abs(hydrogel.fair_value - best_bid_below_fair) <= hydrogel.join_thr:
            bid = best_bid_below_fair
        else:
            bid = best_bid_below_fair + 1

    if hydrogel.position > hydrogel.soft_pos_limit:
        bid -= 1
        ask -= 1
    elif hydrogel.position < -hydrogel.soft_pos_limit:
        bid += 1
        ask += 1

    buy_quantity = hydrogel.limit - (hydrogel.position + hydrogel.posted_buy_volume)
    if buy_quantity > 0:
        orders.append(Order(hydrogel.name, round(bid), buy_quantity))  # Buy order

    sell_quantity = hydrogel.limit + (hydrogel.position - hydrogel.posted_sell_volume)
    if sell_quantity > 0:
        orders.append(Order(hydrogel.name, round(ask), -sell_quantity))  # Sell order

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