import jsonpickle
import numpy as np
from dataclasses import dataclass
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
from math import log, sqrt
from statistics import NormalDist


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
    time_to_expiry: float = None
    implied_vol: float = None
    delta: float = None
    vega: float = None
    moneyness: float = None
    theo_iv: float = None
    theo_price: float = None


class Velvetfruit:
    def __init__(self):
        self.spot: Product = Product(name='VELVETFRUIT_EXTRACT', limit=200)
        self.strike_prices: List[int] = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
        self.call_options: List[CallOption] = [
            CallOption(name=f'VEV_{strike}', limit=300, strike_price=strike, time_to_expiry=3 / 250)
            for strike in self.strike_prices]
        self.vol_window = 30

        # Winning Strategy Tunables
        self.mr_underlying_window = 10
        self.mr_underlying_thr = 15

        self.mr_options_window = 30
        self.mr_options_thr = 5

        self.theo_norm_window = 20
        self.iv_scalping_window = 100
        self.iv_scalping_thr = 0.7  # MAD threshold to activate scalping
        self.thr_open = 0.5


norm = NormalDist()


class BlackScholes:
    def __init__(self, spot, strike, time_to_expiry):
        self.spot = spot
        self.strike = strike
        self.time_to_expiry = max(time_to_expiry, 1e-8)

    def call_price(self, volatility):
        volatility = max(volatility, 1e-8)
        d1 = (log(self.spot / self.strike) + (0.5 * volatility * volatility) * self.time_to_expiry) / (
                    volatility * sqrt(self.time_to_expiry))
        d2 = d1 - volatility * sqrt(self.time_to_expiry)
        return self.spot * norm.cdf(d1) - self.strike * norm.cdf(d2)

    def delta(self, volatility):
        volatility = max(volatility, 1e-8)
        d1 = (log(self.spot / self.strike) + (0.5 * volatility * volatility) * self.time_to_expiry) / (
                    volatility * sqrt(self.time_to_expiry))
        return norm.cdf(d1)

    def vega(self, volatility):
        volatility = max(volatility, 1e-8)
        d1 = (log(self.spot / self.strike) + (0.5 * volatility * volatility) * self.time_to_expiry) / (
                    volatility * sqrt(self.time_to_expiry))
        return norm.pdf(d1) * (self.spot * sqrt(self.time_to_expiry)) / 100

    def implied_volatility(self, market_call_price, max_iterations=100, tolerance=1e-5):
        low_vol, high_vol = 0.0001, 5.0
        intrinsic_value = max(0.0, self.spot - self.strike)
        if market_call_price < intrinsic_value:
            return 0.0001

        volatility = (low_vol + high_vol) / 2.0
        for _ in range(max_iterations):
            estimated_price = self.call_price(volatility)
            diff = estimated_price - market_call_price
            if abs(diff) < tolerance:
                break
            elif diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility


def calc_time_to_expiry(day, ts):
    days_left = (8 - day) - (ts / 1_000_000)
    return max(days_left / 250, 1e-8)


def calc_moneyness(spot, strike, tte):
    return np.log(strike / spot) / np.sqrt(tte)


def update_ema(state_dict: Dict, key: str, value: float, window: int) -> float:
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


def trade_iv_scalping(velvet: Velvetfruit, state_dict: Dict) -> Dict[str, List[Order]]:
    orders = {call.name: [] for call in velvet.call_options}

    for call in velvet.call_options:
        # Skip the deepest ITM call which is used for Mean Reversion proxy
        if call.strike_price == 4000 or not call.fair_value or not call.theo_price:
            continue

        current_theo_diff = call.fair_value - call.theo_price

        # EMA of the difference
        mean_theo_diff = update_ema(state_dict, f'mean_diff_{call.name}', current_theo_diff, velvet.theo_norm_window)

        # Mean Absolute Deviation (MAD) to detect if there is enough volatility to scalp
        mad = abs(current_theo_diff - mean_theo_diff)
        switch_mean = update_ema(state_dict, f'switch_{call.name}', mad, velvet.iv_scalping_window)

        if switch_mean >= velvet.iv_scalping_thr:
            # Low vega options require a higher threshold to be worth scalping
            adj_thr = velvet.thr_open + (0.5 if call.vega <= 1.0 else 0)

            signal = current_theo_diff - mean_theo_diff

            # Scalp Overpriced (Sell)
            if signal >= adj_thr and call.best_bid:
                sell_qty = call.limit + call.position
                if sell_qty > 0:
                    orders[call.name].append(Order(call.name, round(call.best_bid), -sell_qty))

            # Scalp Underpriced (Buy)
            elif signal <= -adj_thr and call.best_ask:
                buy_qty = call.limit - call.position
                if buy_qty > 0:
                    orders[call.name].append(Order(call.name, round(call.best_ask), buy_qty))

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

        m_list = previous_state.get('m_list', [])
        v_list = previous_state.get('v_list', [])

        if 'VELVETFRUIT_EXTRACT' in state.order_depths:
            velvet = Velvetfruit()

            # Update Position and Orderbook Limits
            velvet.spot.position = state.position.get(velvet.spot.name, 0)
            if state.order_depths[velvet.spot.name].buy_orders and state.order_depths[velvet.spot.name].sell_orders:
                velvet.spot.best_bid = max(state.order_depths[velvet.spot.name].buy_orders.keys())
                velvet.spot.best_ask = min(state.order_depths[velvet.spot.name].sell_orders.keys())
                velvet.spot.fair_value = (velvet.spot.best_bid + velvet.spot.best_ask) / 2

            for call in velvet.call_options:
                call.position = state.position.get(call.name, 0)
                if call.name in state.order_depths and state.order_depths[call.name].buy_orders and state.order_depths[
                    call.name].sell_orders:
                    call.best_bid = max(state.order_depths[call.name].buy_orders.keys())
                    call.best_ask = min(state.order_depths[call.name].sell_orders.keys())
                    call.fair_value = (call.best_bid + call.best_ask) / 2

            # Calculate Greeks and gather Smile Data
            tte = calc_time_to_expiry(day=3, ts=state.timestamp)

            if velvet.spot.fair_value:
                for call in velvet.call_options:
                    if call.fair_value:
                        call.time_to_expiry = tte
                        bs = BlackScholes(velvet.spot.fair_value, call.strike_price, tte)

                        call.implied_vol = bs.implied_volatility(call.fair_value)
                        call.vega = bs.vega(call.implied_vol)
                        call.moneyness = calc_moneyness(velvet.spot.fair_value, call.strike_price, tte)

                        # Only add options with actual time value to the Smile fitting curve
                        if call.vega > 1e-6:
                            m_list.append(call.moneyness)
                            v_list.append(call.implied_vol)

            # Maintain moving window for polynomial fit
            max_len = velvet.vol_window * len(velvet.call_options)
            m_list = m_list[-max_len:]
            v_list = v_list[-max_len:]

            # --- DYNAMIC VOLATILITY SMILE FITTING ---
            if len(m_list) >= 10 and velvet.spot.fair_value:
                # Fit the smile parabola
                params = np.polyfit(m_list, v_list, 2)
                smile_func = np.poly1d(params)

                for call in velvet.call_options:
                    if call.moneyness is not None:
                        # 1. Evaluate theoretical IV at this exact strike's moneyness
                        call.theo_iv = smile_func(call.moneyness)
                        # 2. Calculate Theoretical Black-Scholes Price
                        bs = BlackScholes(velvet.spot.fair_value, call.strike_price, tte)
                        call.theo_price = bs.call_price(call.theo_iv)

                # --- EXECUTE HYBRID STRATEGY ---
                # mr_orders = trade_mean_reversion(velvet, previous_state)
                iv_orders = trade_iv_scalping(velvet, previous_state)

                # Combine orders
                # for product_name, p_orders in mr_orders.items():
                #     if p_orders:
                #         result[product_name] = result.get(product_name, []) + p_orders

                for product_name, p_orders in iv_orders.items():
                    if p_orders:
                        result[product_name] = result.get(product_name, []) + p_orders

        # Save state
        previous_state['m_list'] = m_list
        previous_state['v_list'] = v_list
        trader_data = jsonpickle.encode(previous_state)

        return result, 0, trader_data