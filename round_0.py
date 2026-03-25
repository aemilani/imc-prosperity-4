import numpy as np
from datamodel import OrderDepth, TradingState, Order
from typing import List


def trade(product: str, state: TradingState) -> List[Order] | None:
    if product == 'RAINFOREST_RESIN':
        return trade_resin(state)
    elif product == 'KELP':
        return trade_kelp(state)


def trade_resin(state: TradingState) -> List[Order]:
    curr_position = state.position.get('RAINFOREST_RESIN', 0)
    print(f'RAINFOREST_RESIN position: {curr_position}')

    max_position = 50
    max_buy_size = min(max_position, max_position - curr_position)
    max_sell_size = max(-max_position, -max_position - curr_position)

    fair_value = 10000
    print(f'RAINFOREST_RESIN fair value: {fair_value}')

    thr_l = fair_value - 2
    thr_h = fair_value + 2
    buy_price = thr_l
    sell_price = thr_h

    resin_orders = []
    if max_buy_size > 0:
        print(f'BUY: {max_buy_size} @ {buy_price}')
        resin_orders.append(Order('RAINFOREST_RESIN', buy_price, max_buy_size))
    if max_sell_size < 0:
        print(f'SELL: {max_sell_size} @ {sell_price}')
        resin_orders.append(Order('RAINFOREST_RESIN', sell_price, max_sell_size))

    return resin_orders


def trade_kelp(state: TradingState) -> List[Order]:
    curr_position = state.position.get('KELP', 0)
    print(f'KELP position: {curr_position}')

    order_depth: OrderDepth = state.order_depths['KELP']

    max_position = 50
    max_buy_size = min(max_position, max_position - curr_position)
    max_sell_size = max(-max_position, -max_position - curr_position)

    buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
    sell_orders = sorted(order_depth.sell_orders.items())

    popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
    popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

    fair_value = round((popular_buy_price + popular_sell_price) / 2)
    print(f'KELP fair value: {fair_value}')

    kelp_orders = []
    if max_buy_size > 0:
        buy_price = int(np.floor(fair_value)) - 1
        print(f'BUY: {max_buy_size} @ {buy_price}')
        kelp_orders.append(Order('KELP', buy_price, max_buy_size))
    if max_sell_size < 0:
        sell_price = int(np.ceil(fair_value)) + 1
        print(f'SELL: {max_sell_size} @ {sell_price}')
        kelp_orders.append(Order('KELP', sell_price, max_sell_size))

    return kelp_orders


class Trader:
    def run(self, state: TradingState):
        result = {}
        for product in state.order_depths:
            orders: List[Order] = trade(product, state)
            result[product] = orders
            print('---')

        trader_data = None
        conversions = 1
        return result, conversions, trader_data