import jsonpickle
from dataclasses import dataclass
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict


@dataclass
class Product:
    name: str
    limit: int
    fair_value: float = None
    position: int = 0
    best_bid: float = None
    best_ask: float = None


@dataclass
class CallOption(Product):
    strike_price: int = None


class Velvetfruit:
    def __init__(self):
        self.spot: Product = Product(name='VELVETFRUIT_EXTRACT', limit=200)
        self.strike_prices: List[int] = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
        self.call_options: List[CallOption] = [
            CallOption(name=f'VEV_{strike}', limit=300, strike_price=strike)
            for strike in self.strike_prices
        ]

        # Mean Reversion Tunables
        self.mr_underlying_window = 10
        self.mr_underlying_thr = 30

        self.mr_options_window = 30
        self.mr_options_thr = 10


def update_ema(state_dict: Dict, key: str, value: float, window: int) -> float:
    # If the key doesn't exist, initialize it with the first observed value
    old_mean = state_dict.get(key, value)
    alpha = 2 / (window + 1)
    new_mean = alpha * value + (1 - alpha) * old_mean
    state_dict[key] = new_mean
    return new_mean


def trade_mean_reversion(velvet: Velvetfruit, state_dict: Dict) -> Dict[str, List[Order]]:
    orders = {velvet.spot.name: []}
    for call in velvet.call_options:
        orders[call.name] = []

    # 1. Underlying Asset Mean Reversion
    if velvet.spot.fair_value:
        ema_u = update_ema(state_dict, 'ema_u', velvet.spot.fair_value, velvet.mr_underlying_window)
        u_dev = velvet.spot.fair_value - ema_u

        if u_dev > velvet.mr_underlying_thr and velvet.spot.best_bid:
            sell_qty = velvet.spot.limit + velvet.spot.position
            if sell_qty > 0:
                orders[velvet.spot.name].append(Order(velvet.spot.name, round(velvet.spot.best_bid), -sell_qty))

        elif u_dev < -velvet.mr_underlying_thr and velvet.spot.best_ask:
            buy_qty = velvet.spot.limit - velvet.spot.position
            if buy_qty > 0:
                orders[velvet.spot.name].append(Order(velvet.spot.name, round(velvet.spot.best_ask), buy_qty))

    # 2. Deep ITM Option Proxy Hedge (Strike 4000)
    deep_itm_call = next((c for c in velvet.call_options if c.strike_price == 4000), None)
    if deep_itm_call and deep_itm_call.fair_value:
        ema_o = update_ema(state_dict, 'ema_o', deep_itm_call.fair_value, velvet.mr_options_window)
        o_dev = deep_itm_call.fair_value - ema_o

        if o_dev > velvet.mr_options_thr and deep_itm_call.best_bid:
            sell_qty = deep_itm_call.limit + deep_itm_call.position
            if sell_qty > 0:
                orders[deep_itm_call.name].append(Order(deep_itm_call.name, round(deep_itm_call.best_bid), -sell_qty))

        elif o_dev < -velvet.mr_options_thr and deep_itm_call.best_ask:
            buy_qty = deep_itm_call.limit - deep_itm_call.position
            if buy_qty > 0:
                orders[deep_itm_call.name].append(Order(deep_itm_call.name, round(deep_itm_call.best_ask), buy_qty))

    return orders


class Trader:
    def run(self, state: TradingState):
        result = {}
        previous_state = {}

        if state.traderData:
            try:
                previous_state = jsonpickle.decode(state.traderData)
            except Exception:
                pass

        if 'VELVETFRUIT_EXTRACT' in state.order_depths:
            velvet = Velvetfruit()

            # Update Spot Position and Orderbook
            velvet.spot.position = state.position.get(velvet.spot.name, 0)
            if state.order_depths[velvet.spot.name].buy_orders and state.order_depths[velvet.spot.name].sell_orders:
                velvet.spot.best_bid = max(state.order_depths[velvet.spot.name].buy_orders.keys())
                velvet.spot.best_ask = min(state.order_depths[velvet.spot.name].sell_orders.keys())
                velvet.spot.fair_value = (velvet.spot.best_bid + velvet.spot.best_ask) / 2

            # Update Options Positions and Orderbook
            for call in velvet.call_options:
                call.position = state.position.get(call.name, 0)
                if call.name in state.order_depths and state.order_depths[call.name].buy_orders and state.order_depths[
                    call.name].sell_orders:
                    call.best_bid = max(state.order_depths[call.name].buy_orders.keys())
                    call.best_ask = min(state.order_depths[call.name].sell_orders.keys())
                    call.fair_value = (call.best_bid + call.best_ask) / 2

            # --- EXECUTE PURE MEAN REVERSION ---
            mr_orders = trade_mean_reversion(velvet, previous_state)

            # Assign orders to result
            for product_name, p_orders in mr_orders.items():
                if p_orders:
                    result[product_name] = p_orders

        # Encode and pass the state dictionary to the next tick
        trader_data = jsonpickle.encode(previous_state)

        return result, 0, trader_data