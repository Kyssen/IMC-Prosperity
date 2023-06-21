from typing import Dict, List, Tuple
from datamodel import OrderDepth, TradingState, Order
import statistics


# CONST for all products
TRADING_LIMITS = {'PEARLS': 20,
                    'BANANAS': 20,
                    'COCONUTS': 600,
                    'PINA_COLADAS': 300,
                    'BERRIES': 250,
                    'DIVING_GEAR': 50,
                    'BAGUETTE': 150,
                    'DIP': 300,
                    'UKULELE': 70,
                    'PICNIC_BASKET': 70}
LEVEL: int = 2

# CONST for banana
rolling_mid_banana = [] # rolling avg of mid for n iterations
ROLLING_MID_N_BANANA = 10
POSITION_SCALAR_BANANA = 0 # or 0.05

# CONST for pearl
rolling_mid_pearl = []
ROLLING_MID_N_PEARL = 10
POSITION_SCALAR_PEARL = 0

# CONST for PC and C
PINA_COLADA_COCONUT_RATIO = 1.875
ETF_ADJ_FACTOR = 0.1 # if etf price is 20, adjust price by 2 This may be too high.
POSITION_ADJ_FACTOR = 1.5 # if (PC * 1.875 / C) ratio is 2, adjust price by 2
IMBALANCE_TOLERANCE = 0
SPEED_FACTOR = 2
# FRACTIONAL_POS_LIMIT = {-1: 0.1, 20: 0.5, 40: 0.85, 60: 0.95, 80: 1.0} # |etf| > key => can hold up to value * total pos_lim.

FRACTIONAL_POS_LIMIT = {-1: 0.1, 20: 0.5, 40: 1.0} # |etf| > key => can hold up to value * total pos_lim.
START_LIQUIDATING_CUTOFF_PCC = 10
min_diff = 20

# CONST for berries
rolling_mid_berries = []
ROLLING_MID_N_BERRY = 10
POSITION_SCALAR_BERRY = 0 # or 0.05 (tested for berries if we dont want to hold net pos) # note this being zero => alows for trend riding.
REGIME_CUTOFFS = [390000, 500000, 610000]
# REGIME_CUTOFFS = [30000, 50000, 70000]
liquidated_flag = True

# CONST for gear
buy_que = []
sell_que = []
rolling_mid_gear = []
rolling_rolling_mid_gear = []
ROLLING_MID_N_GEAR = 120
ROLLING_ROLLING_MID_N_GEAR = 20
shorting_flag = False
longing_flag = False
prev_dolf = -1
DOLPHIN_DIFF_CUTOFF = 5
time_since_last_event = 0
MIN_MOMENTUM_TIME = 200 # or 150?

# CONST for basket and underlyings
ETF_PREMIUM = 350
FRACTIONAL_POS_LIMIT_BASKET = {-1: 0.0, 90: 0.5, 120: 0.75, 200: 0.95, 300: 1.0} # |diff| > key => can hold up to value * total pos_lim.
# FRACTIONAL_POS_LIMIT_BASKET = {-1: 0.0, 90: 0.0, 100: 0.9, 200: 1.0, 300: 1.0}
# FRACTIONAL_POS_LIMIT_BASKET = {-1: 0.0, 90: 0.0, 100: 1.0, 200: 1.0, 300: 1.0}
SPEED_FACTOR_BASKET = 0.1
basket_min_diff = 90
START_LIQUIDATING_CUTOFF = 15
liquidated_from_olivia_strat = True


# MONEKY CONSTANTS
olivia_indicator = {'BANANAS': 0,
                    'BERRIES': 0,
                    'UKULELE': 0}
olivia_position = {'BANANAS': 0,
                   'BERRIES': 0,
                   'UKULELE': 0}


import json
from datamodel import Order, ProsperityEncoder, TradingState, Symbol, Trade
from typing import Any


class Logger:
    def __init__(self) -> None:
        self.logs = ""

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]]) -> None:
        print(json.dumps({
            "state": self.compress_state(state),
            "orders": self.compress_orders(orders),
            "logs": self.logs,
        }, cls=ProsperityEncoder, separators=(",", ":"), sort_keys=True))

        self.logs = ""

    def compress_state(self, state: TradingState) -> dict[str, Any]:
        listings = []
        for listing in state.listings.values():
            listings.append([listing["symbol"], listing["product"], listing["denomination"]])

        order_depths = {}
        for symbol, order_depth in state.order_depths.items():
            order_depths[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return {
            "t": state.timestamp,
            "l": listings,
            "od": order_depths,
            "ot": self.compress_trades(state.own_trades),
            "mt": self.compress_trades(state.market_trades),
            "p": state.position,
            "o": state.observations,
        }

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.buyer,
                    trade.seller,
                    trade.price,
                    trade.quantity,
                    trade.timestamp,
                ])

        return compressed

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed


# logger = Logger(local=True)

class Trader:
    logger = Logger()

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {}

        # # DEBUG
        # if state.timestamp == 61000:
        #     state.market_trades['BERRIES'].append(Trade('BERRIES', 50, 3, 'Caesar', 'Olivia', 61000))
        # if state.timestamp == 81000:
        #     state.market_trades['BERRIES'].append(Trade('BERRIES', 50, 2, 'Olivia', 'Caesar', 81000))
        # if state.timestamp == 15000:
        #     state.market_trades['BERRIES'].append(Trade('BERRIES', 50, 1, 'Olivia', 'Caesar', 15000))

        # if state.timestamp == 21000:
        #     if 'UKULELE' in state.market_trades:
        #         state.market_trades['UKULELE'].append(Trade('UKULELE', 50, 3, 'Caesar', 'Olivia', 21000)) # olivia sold 3
        #     else:
        #         state.market_trades['UKULELE'] = [Trade('UKULELE', 50, 3, 'Caesar', 'Olivia', 21000)]
        # if state.timestamp == 61000:
        #     if 'UKULELE' in state.market_trades:
        #         state.market_trades['UKULELE'].append(Trade('UKULELE', 50, 3, 'Olivia', 'Caesar', 61000)) # olivia bought 3
        #     else:
        #         state.market_trades['UKULELE'] = [Trade('UKULELE', 50, 3, 'Olivia', 'Caesar', 61000)]


        # if state.timestamp == 81000:
        #     if 'UKULELE' in state.market_trades:
        #         state.market_trades['UKULELE'].append(Trade('UKULELE', 50, 2, 'Olivia', 'Caesar', 81000)) # olivia bought 2
        #     else:
        #         state.market_trades['UKULELE'] = [Trade('UKULELE', 50, 2, 'Olivia', 'Caesar', 81000)]
        
        self.get_monkey_indocators(state)  

        # Iterate over all the keys (the available products) contained in the order dephts
        for product in state.order_depths.keys():     

            # if product == 'BANANAS':    # limit orders
            #     orders = self.get_orders_bananas(state)
            #     result[product] = orders

            # if product == 'PEARLS':     # limit orders
            #     orders = self.get_orders_pearls(state)
            #     result[product] = orders

            if (product == 'COCONUTS') and ('PINA_COLADAS' in state.order_depths):  # limit orders
                orders_coconut, orders_pina_colada = self.get_orders_coconut_pina_colada(state)
                result['COCONUTS'] = orders_coconut
                result['PINA_COLADAS'] = orders_pina_colada

            # if product == 'BERRIES':
            #     orders = self.get_orders_berries(state)
            #     result[product] = orders

            # if product == 'DIVING_GEAR':
            #     orders = self.get_orders_gear(state)
            #     result[product] = orders

            # if (product == 'PICNIC_BASKET') and ('DIP' in state.order_depths) and ('BAGUETTE' in state.order_depths) and ('UKULELE' in state.order_depths):
            #     orders_basket, orders_baguette, orders_dip, orders_ukulele = self.get_orders_picnic_dip_baguette_ukulele(state)
            #     result['PICNIC_BASKET'] = orders_basket
            #     result['BAGUETTE'] = orders_baguette
            #     result['DIP'] = orders_dip
            #     result['UKULELE'] = orders_ukulele
            
        self.logger.flush(state, result) # TODO: comment out

        return result

    def get_book(self, order_depth: OrderDepth) -> Tuple[Dict[int, int], Dict[int, int], int, int, List[int], List[int]]:
        bid_book = order_depth.buy_orders
        ask_book = {key: order_depth.sell_orders[key] * -1 for key in order_depth.sell_orders.keys()}
        bid_level_cutoff = min(LEVEL, len(bid_book))
        ask_level_cutoff = min(LEVEL, len(ask_book))
        bid_keys_sorted = sorted(bid_book.keys(), reverse=True)
        ask_keys_sorted = sorted(ask_book.keys())
        return bid_book, ask_book, bid_level_cutoff, ask_level_cutoff, bid_keys_sorted, ask_keys_sorted
    
    def get_orders_bananas(self, state) -> List[Order]:
        global olivia_indicator, olivia_position
        # Retrieve the Order Depth and positions
        product = 'BANANAS'
        order_depth: OrderDepth = state.order_depths[product]
        if product in state.position:
            positions: int = state.position[product]
        else:
            positions: int = 0

        # going forward book not empty
        bid_book, ask_book, bid_level_cutoff, ask_level_cutoff, bid_keys_sorted, ask_keys_sorted = self.get_book(order_depth)
        # undervalued_index = adjusted_theo - mid 
        orders = []
        remaining_bid_quote_volume = TRADING_LIMITS[product] - positions
        remaining_ask_quote_volume = TRADING_LIMITS[product] + positions

        mid = statistics.mean([bid_keys_sorted[0], ask_keys_sorted[0]])
        rolling_mid_banana.append(mid)
        if len(rolling_mid_banana) > ROLLING_MID_N_BANANA:
            rolling_mid_banana.pop(0)

        best_bid = bid_keys_sorted[0]
        best_ask = ask_keys_sorted[0]
        rolling_mid_avg = statistics.mean(rolling_mid_banana)

        # this turned out useless
        # rolling_mid_avg += olivia_indicator[product]

        # buy orders
        if remaining_bid_quote_volume > 0:

            # method 2
            if best_ask - best_bid <= 5: # squeeze
                if abs(mid - rolling_mid_avg) < 1: # if rolling and mid is similar
                    orders.append(Order(product, round((mid - 2) - positions * POSITION_SCALAR_BANANA), remaining_bid_quote_volume))
                else: # rolling and mid not similar (at a spike)
                    if mid > rolling_mid_avg: # up spike (green)
                        orders.append(Order(product, round((rolling_mid_avg - 3) - positions * POSITION_SCALAR_BANANA), remaining_bid_quote_volume))
                    else: # down spike (red)
                        # pick up the red littl guy(s) (buy up all the sells that are less than rolling - 0.5)
                        bids = {} # price as key, order as value
                        for price in ask_keys_sorted:
                            if price < rolling_mid_avg - 0.5:
                                order = Order(product, price, min(remaining_bid_quote_volume, ask_book[price]))
                                remaining_bid_quote_volume -= min(remaining_bid_quote_volume, ask_book[price])
                                bids[price] = order
                        if remaining_bid_quote_volume > 0:
                            bid_price = round((best_bid + 1) - positions * POSITION_SCALAR_BANANA)
                            if bid_price in bids:
                                order = Order(product, bid_price, bids[bid_price].quantity + remaining_bid_quote_volume)
                            else:
                                order = Order(product, bid_price, remaining_bid_quote_volume) # submit order as normal for remaining available bid volume
                            bids[bid_price] = order
                            for price in bids: # iterate through all the bids and append to orders list
                                orders.append(bids[price])
            else: # no squeeze, submit as usual
                orders.append(Order(product, round((best_bid + 1) - positions * POSITION_SCALAR_BANANA), remaining_bid_quote_volume))
        
        # sell orders
        if remaining_ask_quote_volume > 0:
    
            # method 2
            if best_ask - best_bid <= 5: # squeeze
                if abs(mid - rolling_mid_avg) < 1: # if rolling and mid is similar
                    orders.append(Order(product, round((mid + 2) - positions * POSITION_SCALAR_BANANA), -remaining_ask_quote_volume))
                else: # rolling and mid not similar (at a spike)
                    if mid < rolling_mid_avg: # down spike (red)
                        orders.append(Order(product, round((rolling_mid_avg + 3) - positions * POSITION_SCALAR_BANANA), -remaining_ask_quote_volume))
                    else: # up spike (green)
                        asks = {}
                        for price in bid_keys_sorted:
                            if price > rolling_mid_avg + 0.5:
                                order = Order(product, price, -min(remaining_ask_quote_volume, bid_book[price]))
                                remaining_ask_quote_volume -= min(remaining_ask_quote_volume, bid_book[price])
                                asks[price] = order
                        if remaining_ask_quote_volume > 0:
                            ask_price = round((best_ask - 1) - positions * POSITION_SCALAR_BANANA)
                            if ask_price in asks:
                                order = Order(product, ask_price, asks[ask_price].quantity - remaining_ask_quote_volume)
                            else:
                                order = Order(product, ask_price, -remaining_ask_quote_volume) # submit order as normal for remaining available ask volume
                            asks[ask_price] = order
                            for price in asks: # iterate through all the asks and append to orders list
                                orders.append(asks[price])
            else: # no squeeze, submit as usual
                orders.append(Order(product, round((best_ask - 1) - positions * POSITION_SCALAR_BANANA), -remaining_ask_quote_volume))
            
        return orders


    def get_orders_pearls(self, state):
        # Retrieve the Order Depth and positions
        product = 'PEARLS'
        order_depth: OrderDepth = state.order_depths[product]
        if product in state.position:
            positions: int = state.position[product]
        else:
            positions: int = 0
        
        bid_book, ask_book, bid_level_cutoff, ask_level_cutoff, bid_keys_sorted, ask_keys_sorted = self.get_book(order_depth)
        orders = []
        remaining_bid_quote_volume = TRADING_LIMITS[product] - positions
        remaining_ask_quote_volume = TRADING_LIMITS[product] + positions

        mid = statistics.mean([bid_keys_sorted[0], ask_keys_sorted[0]])
        rolling_mid_pearl.append(mid)
        if len(rolling_mid_pearl) > ROLLING_MID_N_PEARL:
            rolling_mid_pearl.pop(0)

        best_bid = bid_keys_sorted[0]
        best_ask = ask_keys_sorted[0] 
        rolling_mid_avg = statistics.mean(rolling_mid_pearl)

        # buy orders
        if remaining_bid_quote_volume > 0: 

            # method 3
            if best_ask - best_bid <= 4: # squeeze
                if abs(mid - rolling_mid_avg) < 1: # if rolling and mid is similar
                    orders.append(Order(product, round((mid - 2) - positions * POSITION_SCALAR_PEARL), remaining_bid_quote_volume)) 
                else: # rolling and mid not similar (at a spike)
                    if mid > rolling_mid_avg: # up spike (green)
                        orders.append(Order(product, round((rolling_mid_avg - 3) - positions * POSITION_SCALAR_PEARL), remaining_bid_quote_volume))
                    else: # down spike (red)
                        # pick up the red littl guy(s) (buy up all the sells that are less than rolling - 0.5)
                        bids = {} # price as key, order as value
                        for price in ask_keys_sorted:
                            if price < rolling_mid_avg - 0.5:
                                order = Order(product, price, min(remaining_bid_quote_volume, ask_book[price]))
                                remaining_bid_quote_volume -= min(remaining_bid_quote_volume, ask_book[price])
                                bids[price] = order
                        # place big one
                        if remaining_bid_quote_volume > 0:
                            bid_price = round((best_bid + 1) - positions * POSITION_SCALAR_PEARL)
                            if bid_price in bids:
                                order = Order(product, bid_price, bids[bid_price].quantity + remaining_bid_quote_volume)
                            else:
                                order = Order(product, bid_price, remaining_bid_quote_volume) # submit order as normal for remaining available bid volume
                            bids[bid_price] = order
                            for price in bids: # iterate through all the bids and append to orders list
                                orders.append(bids[price])
            else: # no squeeze, submit as usual
                orders.append(Order(product, round((best_bid + 1) - positions * POSITION_SCALAR_PEARL), remaining_bid_quote_volume))

        # sell orders
        if remaining_ask_quote_volume > 0: 

            # method 3
            if best_ask - best_bid <= 4: # squeeze
                if abs(mid - rolling_mid_avg) < 1: # if rolling and mid is similar
                    orders.append(Order(product, round((mid + 2) - positions * POSITION_SCALAR_PEARL), -remaining_ask_quote_volume))
                else: # rolling and mid not similar (at a spike)
                    if mid < rolling_mid_avg: # down spike (red)
                        orders.append(Order(product, round((rolling_mid_avg + 3) - positions * POSITION_SCALAR_PEARL), -remaining_ask_quote_volume))
                    else: # up spike (green)
                        asks = {}
                        for price in bid_keys_sorted:
                            if price > rolling_mid_avg + 0.5:
                                order = Order(product, price, -min(remaining_ask_quote_volume, bid_book[price]))
                                remaining_ask_quote_volume -= min(remaining_ask_quote_volume, bid_book[price])
                                asks[price] = order
                        if remaining_ask_quote_volume > 0:
                            ask_price = round((best_ask - 1) - positions * POSITION_SCALAR_PEARL)
                            if ask_price in asks:
                                order = Order(product, ask_price, asks[ask_price].quantity - remaining_ask_quote_volume)
                            else:
                                order = Order(product, ask_price, -remaining_ask_quote_volume) # submit order as normal for remaining available ask volume
                            asks[ask_price] = order
                            for price in asks: # iterate through all the asks and append to orders list
                                orders.append(asks[price])
            else: # no squeeze, submit as usual
                orders.append(Order(product, round((best_ask - 1) - positions * POSITION_SCALAR_PEARL), -remaining_ask_quote_volume))

        return orders

    def get_orders_coconut_pina_colada(self, state):
        # Retrieve the Order Depth and positions
        orders_c_m1, orders_pc_m1 = self.get_orders_pc_c_method0(state)
        return orders_c_m1, orders_pc_m1

    def get_best_etf_bid_ask(self, order_depth_c, order_depth_pc): # from market perspective
        bid_book_c, ask_book_c, bid_level_cutoff_c, ask_level_cutoff_c, bid_keys_sorted_c, ask_keys_sorted_c = self.get_book(order_depth_c)
        bid_book_pc, ask_book_pc, bid_level_cutoff_pc, ask_level_cutoff_pc, bid_keys_sorted_pc, ask_keys_sorted_pc = self.get_book(order_depth_pc)
        
        # TODO: optimize this with while loop?
        best_bid = bid_keys_sorted_pc[0] - ask_keys_sorted_c[0] * PINA_COLADA_COCONUT_RATIO
        best_bid_volume = min(bid_book_pc[bid_keys_sorted_pc[0]], round(ask_book_c[ask_keys_sorted_c[0]] / PINA_COLADA_COCONUT_RATIO))

        best_ask = ask_keys_sorted_pc[0] - bid_keys_sorted_c[0] * PINA_COLADA_COCONUT_RATIO
        best_ask_volume = min(ask_book_pc[ask_keys_sorted_pc[0]], round(bid_book_c[bid_keys_sorted_c[0]] / PINA_COLADA_COCONUT_RATIO))

        return best_bid, best_ask, best_bid_volume, best_ask_volume
    
    def get_orders_pc_c_method0(self, state):
        # get stuff needed
        product_c = 'COCONUTS'
        product_pc = 'PINA_COLADAS'
        order_depth_c = state.order_depths[product_c]
        order_depth_pc = state.order_depths[product_pc]
        if product_c in state.position:
            positions_c: int = state.position[product_c]
        else:
            positions_c: int = 0
        
        if product_pc in state.position:
            positions_pc: int = state.position[product_pc]
        else:
            positions_pc: int = 0
        
        bid_book_c, ask_book_c, bid_level_cutoff_c, ask_level_cutoff_c, bid_keys_sorted_c, ask_keys_sorted_c = self.get_book(order_depth_c)
        bid_book_pc, ask_book_pc, bid_level_cutoff_pc, ask_level_cutoff_pc, bid_keys_sorted_pc, ask_keys_sorted_pc = self.get_book(order_depth_pc)

        orders_c = []
        orders_pc = []

        remaining_bid_quote_volume_c = round((TRADING_LIMITS[product_c] - positions_c))
        remaining_ask_quote_volume_c = round((TRADING_LIMITS[product_c] + positions_c))
        remaining_bid_quote_volume_pc = round((TRADING_LIMITS[product_pc] - positions_pc))
        remaining_ask_quote_volume_pc = round((TRADING_LIMITS[product_pc] + positions_pc))

        c_best_bid = bid_keys_sorted_c[0]
        c_best_ask = ask_keys_sorted_c[0]
        c_mid = (c_best_bid + c_best_ask) / 2
        pc_best_bid = bid_keys_sorted_pc[0]
        pc_best_ask = ask_keys_sorted_pc[0]
        pc_mid = (pc_best_bid + pc_best_ask) / 2
        diff = pc_mid - PINA_COLADA_COCONUT_RATIO * c_mid
        
        # -----------------------------------
        # # if we are converging back to zero (back to ETF_PREMIUM), then liquidate FAST TODO:(with both market and limit?)
        # if diff <= START_LIQUIDATING_CUTOFF_PCC and (positions_pc < 0 or positions_c > 0): # diff just dropped back down to 0
        #     print('diff just dropped back down to near 0, trying to liquidate')
        #     # buy pc, sell c
        #     if positions_pc < 0:
        #         orders_pc.append(Order(product_pc, round(pc_best_ask), -positions_pc)) # buy pc
        #     if positions_c > 0:
        #         orders_c.append(Order(product_c, round(c_best_bid), -positions_c)) # sell c

        # elif diff >= -START_LIQUIDATING_CUTOFF_PCC and (positions_pc > 0 or positions_c < 0): # diff just increased back up to 0
        #     # sell basket, buy underlying
        #     print('diff just increased back up to near 0, trying to liquidate')
        #     if positions_pc > 0:
        #         orders_pc.append(Order(product_pc, round(pc_best_bid), -positions_pc)) # sell basket
        #     if positions_c < 0:
        #         orders_c.append(Order(product_c, round(c_best_ask), -positions_c)) # buy underlying
            
        # # otherwise (building up position), then trade max amount allowed by fractional restriction
        # else:
        #     print('difference is large or all of the underlyings are liquidated OR we didnt finish liquidating but now diff is on the other side')
        #     fractional_pos_lim = FRACTIONAL_POS_LIMIT[self.largest_number_less_than(list(FRACTIONAL_POS_LIMIT.keys()), abs(diff))]

        #     bid_volume_pc = round(min(remaining_bid_quote_volume_pc, abs(diff) * SPEED_FACTOR, max(int(TRADING_LIMITS[product_pc] * fractional_pos_lim) - positions_pc, 0)))
        #     ask_volume_pc = round(min(remaining_ask_quote_volume_pc, abs(diff) * SPEED_FACTOR, max(positions_pc + int(TRADING_LIMITS[product_pc] * fractional_pos_lim), 0)))

        #     bid_volume_c = round(min(remaining_bid_quote_volume_c, abs(diff) * SPEED_FACTOR * PINA_COLADA_COCONUT_RATIO, max(int(TRADING_LIMITS[product_c] * fractional_pos_lim) - positions_c, 0)))
        #     ask_volume_c = round(min(remaining_ask_quote_volume_c, abs(diff) * SPEED_FACTOR * PINA_COLADA_COCONUT_RATIO, max(positions_c + int(TRADING_LIMITS[product_c] * fractional_pos_lim), 0)))

        #     if diff > min_diff: 
        #         print('diff is very positive. we want to sell basket and buy underlying')
        #         # we want to sell basket and buy underlying
        #         # number of market trades we can make
        #         num_pc_tradable = int(min(bid_book_pc[pc_best_bid], ask_book_c[c_best_ask] / PINA_COLADA_COCONUT_RATIO))
        #         orders_pc.append(Order(product_pc, pc_best_bid, -min(num_pc_tradable, ask_volume_pc))) # sell basket
        #         orders_c.append(Order(product_c, c_best_ask, int(min(num_pc_tradable * PINA_COLADA_COCONUT_RATIO, bid_volume_c)))) # buy underlying
                
        #     elif diff < -min_diff:
        #         print('diff is very negative. we want to buy basket and sell underlying')
        #         # we want to buy basket and sell underlying
        #         # number of market trades we can make
        #         num_pc_tradable = int(min(ask_book_pc[pc_best_ask], bid_book_c[c_best_bid] / PINA_COLADA_COCONUT_RATIO))
        #         orders_pc.append(Order(product_pc, pc_best_ask, min(num_pc_tradable, bid_volume_pc))) # buy basket
        #         orders_c.append(Order(product_c, c_best_bid, -min(num_pc_tradable * PINA_COLADA_COCONUT_RATIO, ask_volume_c))) # sell underlying
        #     else: 
        #         print('diff is small. do not trade')
        #         # Market make
        #         pass
        #         # return self.get_orders_picnic_method2(state)

        # if we are converging back to zero (back to ETF_PREMIUM), then liquidate FAST TODO:(with both market and limit?)


        if diff <= START_LIQUIDATING_CUTOFF_PCC and (positions_pc < 0 or positions_c > 0): # diff just dropped back down to 0
            print('diff just dropped back down to near 0, trying to liquidate')
            # buy pc, sell c
            if positions_pc < 0:
                orders_pc.append(Order(product_pc, round(pc_best_bid + 1), -positions_pc)) # buy pc
            if positions_c > 0:
                orders_c.append(Order(product_c, round(c_best_ask - 1), -positions_c)) # sell c

        elif diff >= -START_LIQUIDATING_CUTOFF_PCC and (positions_pc > 0 or positions_c < 0): # diff just increased back up to 0
            # sell basket, buy underlying
            print('diff just increased back up to near 0, trying to liquidate')
            if positions_pc > 0:
                orders_pc.append(Order(product_pc, round(pc_best_ask - 1), -positions_pc)) # sell basket
            if positions_c < 0:
                orders_c.append(Order(product_c, round(c_best_bid + 1), -positions_c)) # buy underlying
            
        # otherwise (building up position), then trade max amount allowed by fractional restriction
        else:
            print('difference is large or all of the underlyings are liquidated OR we didnt finish liquidating but now diff is on the other side')
            fractional_pos_lim = FRACTIONAL_POS_LIMIT[self.largest_number_less_than(list(FRACTIONAL_POS_LIMIT.keys()), abs(diff))]

            bid_volume_pc = round(min(remaining_bid_quote_volume_pc, max(int(TRADING_LIMITS[product_pc] * fractional_pos_lim) - positions_pc, 0)))
            ask_volume_pc = round(min(remaining_ask_quote_volume_pc, max(positions_pc + int(TRADING_LIMITS[product_pc] * fractional_pos_lim), 0)))

            bid_volume_c = round(min(remaining_bid_quote_volume_c, max(int(TRADING_LIMITS[product_c] * fractional_pos_lim) - positions_c, 0)))
            ask_volume_c = round(min(remaining_ask_quote_volume_c, max(positions_c + int(TRADING_LIMITS[product_c] * fractional_pos_lim), 0)))

            if diff > min_diff: 
                print('diff is very positive. we want to sell basket and buy underlying')
                # we want to sell basket and buy underlying
                # number of market trades we can make
                orders_pc.append(Order(product_pc, pc_best_ask - 1, -ask_volume_pc)) # sell basket
                orders_c.append(Order(product_c, c_best_bid + 1, bid_volume_c)) # buy underlying
                
            elif diff < -min_diff:
                print('diff is very negative. we want to buy basket and sell underlying')
                # we want to buy basket and sell underlying
                # number of market trades we can make
                orders_pc.append(Order(product_pc, pc_best_bid + 1, bid_volume_pc)) # buy basket
                orders_c.append(Order(product_c, c_best_ask - 1, -ask_volume_c)) # sell underlying
            else: 
                print('diff is small. do not trade')
                # Market make
                pass
                # return self.get_orders_picnic_method2(state)

        return orders_c, orders_pc
        

    def get_orders_pc_c_method1(self, state):
        # get stuff needed
        product_c = 'COCONUTS'
        product_pc = 'PINA_COLADAS'
        order_depth_c = state.order_depths[product_c]
        order_depth_pc = state.order_depths[product_pc]
        if product_c in state.position:
            positions_c: int = state.position[product_c]
        else:
            positions_c: int = 0
        
        if product_pc in state.position:
            positions_pc: int = state.position[product_pc]
        else:
            positions_pc: int = 0
        
        bid_book_c, ask_book_c, bid_level_cutoff_c, ask_level_cutoff_c, bid_keys_sorted_c, ask_keys_sorted_c = self.get_book(order_depth_c)
        bid_book_pc, ask_book_pc, bid_level_cutoff_pc, ask_level_cutoff_pc, bid_keys_sorted_pc, ask_keys_sorted_pc = self.get_book(order_depth_pc)

        orders_c = []
        orders_pc = []

        remaining_bid_quote_volume_c = round((TRADING_LIMITS[product_c] - positions_c))
        remaining_ask_quote_volume_c = round((TRADING_LIMITS[product_c] + positions_c))
        remaining_bid_quote_volume_pc = round((TRADING_LIMITS[product_pc] - positions_pc))
        remaining_ask_quote_volume_pc = round((TRADING_LIMITS[product_pc] + positions_pc))

        c_best_bid = bid_keys_sorted_c[0]
        c_best_ask = ask_keys_sorted_c[0]
        c_mid = (c_best_bid + c_best_ask) / 2
        pc_best_bid = bid_keys_sorted_pc[0]
        pc_best_ask = ask_keys_sorted_pc[0]
        pc_mid = (pc_best_bid + pc_best_ask) / 2
        etf_price = pc_mid - PINA_COLADA_COCONUT_RATIO * c_mid
        # stategy--------------------------------------------------------------------
        # my ETF quotes
        bid_price_c = c_best_bid
        ask_price_c = c_best_ask
        bid_price_pc = pc_best_bid
        ask_price_pc = pc_best_ask
        if etf_price > 0:
            # we want to sell PC and buy C (avoid buying pc and selling c)
            bid_price_pc -= 2
            ask_price_c += 2
        if etf_price < 0:
            # we want to buy PC and sell C (avoid selling PC and buying C)
            ask_price_pc += 2
            bid_price_c -= 2

        # if we are converging back to zero, then liquidate FAST
        if ((etf_price < 0 and (positions_pc < 0 or positions_c > 0)) or (etf_price > 0 and (positions_pc > 0 or positions_c < 0))): # etf just converged back to 0
            bid_volume_c = round(min(remaining_bid_quote_volume_c, abs(positions_c))) 
            ask_volume_c = round(min(remaining_ask_quote_volume_c, abs(positions_c)))
            bid_volume_pc = round(min(remaining_bid_quote_volume_pc, abs(positions_pc)))
            ask_volume_pc = round(min(remaining_ask_quote_volume_pc, abs(positions_pc)))

        # # otherwise (building up position), then trade max amount allowed by fractional restriction
        else:
            fractional_pos_lim = FRACTIONAL_POS_LIMIT[self.largest_number_less_than(list(FRACTIONAL_POS_LIMIT.keys()), abs(etf_price))]

            bid_volume_pc = round(min(remaining_bid_quote_volume_pc, abs(etf_price) * SPEED_FACTOR, max(int(TRADING_LIMITS[product_pc] * fractional_pos_lim) - positions_pc, 0)))
            ask_volume_pc = round(min(remaining_ask_quote_volume_pc, abs(etf_price) * SPEED_FACTOR, max(positions_pc + int(TRADING_LIMITS[product_pc] * fractional_pos_lim), 0)))

            bid_volume_c = round(min(remaining_bid_quote_volume_c, abs(etf_price) * SPEED_FACTOR * PINA_COLADA_COCONUT_RATIO, max(int(TRADING_LIMITS[product_c] * fractional_pos_lim) - positions_c, 0)))
            ask_volume_c = round(min(remaining_ask_quote_volume_c, abs(etf_price) * SPEED_FACTOR * PINA_COLADA_COCONUT_RATIO, max(positions_c + int(TRADING_LIMITS[product_c] * fractional_pos_lim), 0)))

        # adjustment for etf (price difference between pc and ratio*c). 
        # The larger the (PC - Ratio * C), the more I am willing to sell the etf (at a lower price) (PC ask - , C bid + ; PC bid - , C ask + )
        # The smaller the (PC - Ratio * C), the more I am willing to buy the etf (at a higher price) (PC bid +, C ask -; PC ask +, C bid - )
        if abs(etf_price) < 20:
            ask_price_pc -= etf_price * ETF_ADJ_FACTOR
            bid_price_c += etf_price * ETF_ADJ_FACTOR
            bid_price_pc -= etf_price * ETF_ADJ_FACTOR
            ask_price_c += etf_price * ETF_ADJ_FACTOR
        else: #keeps the etf price's sign but with value 2
            ask_price_pc -= etf_price / abs(etf_price) * 2
            bid_price_c += etf_price / abs(etf_price) * 2
            bid_price_pc -= etf_price / abs(etf_price) * 2
            ask_price_c += etf_price / abs(etf_price) * 2

        # adjustment for positions (always want to have 1.875PC : -C)
        if etf_price > 0: # then we want PC negative and C positive
            if positions_pc < 0 and positions_c > 0:
                if -positions_pc * PINA_COLADA_COCONUT_RATIO + IMBALANCE_TOLERANCE < positions_c:
                    # sell c, sell pc
                    imbalance_score = abs(positions_c / (positions_pc * PINA_COLADA_COCONUT_RATIO))
                    ask_price_c -= imbalance_score * POSITION_ADJ_FACTOR
                    bid_price_c -= imbalance_score * POSITION_ADJ_FACTOR
                    ask_price_pc -= imbalance_score * POSITION_ADJ_FACTOR
                    bid_price_pc -= imbalance_score * POSITION_ADJ_FACTOR
                elif -positions_pc * PINA_COLADA_COCONUT_RATIO > positions_c + IMBALANCE_TOLERANCE:
                    # buy c, buy pc
                    imbalance_score = abs(positions_pc * PINA_COLADA_COCONUT_RATIO / positions_c)
                    ask_price_c += imbalance_score * POSITION_ADJ_FACTOR
                    bid_price_c += imbalance_score * POSITION_ADJ_FACTOR
                    ask_price_pc += imbalance_score * POSITION_ADJ_FACTOR
                    bid_price_pc += imbalance_score * POSITION_ADJ_FACTOR
            if positions_pc > 0:
                # sell PC
                ask_price_pc -= 1
                bid_price_pc -= 1
            if positions_c < 0:
                # buy C
                ask_price_c += 1
                bid_price_c += 1
        elif etf_price < 0: # then we want PC positive and C negative
            if positions_pc > 0 and positions_c < 0:
                if positions_pc * PINA_COLADA_COCONUT_RATIO + IMBALANCE_TOLERANCE < -positions_c:
                    # buy PC, buy C:
                    imblanace_score = abs(positions_c / (positions_pc * PINA_COLADA_COCONUT_RATIO))
                    ask_price_c += imblanace_score * POSITION_ADJ_FACTOR
                    bid_price_c += imblanace_score * POSITION_ADJ_FACTOR
                    ask_price_pc += imblanace_score * POSITION_ADJ_FACTOR
                    bid_price_pc += imblanace_score * POSITION_ADJ_FACTOR
                elif positions_pc * PINA_COLADA_COCONUT_RATIO > -positions_c + IMBALANCE_TOLERANCE:
                    # sell PC, sell C
                    imbalance_score = abs(positions_pc * PINA_COLADA_COCONUT_RATIO / positions_c)
                    ask_price_c -= imbalance_score * POSITION_ADJ_FACTOR
                    bid_price_c -= imbalance_score * POSITION_ADJ_FACTOR
                    ask_price_pc -= imbalance_score * POSITION_ADJ_FACTOR
                    bid_price_pc -= imbalance_score * POSITION_ADJ_FACTOR
            if positions_pc < 0:
                # buy PC
                ask_price_pc += 1
                bid_price_pc += 1
            if positions_c > 0:
                # sell C
                ask_price_c -= 1
                bid_price_c -= 1


        orders_pc.append(Order(product_pc, round(bid_price_pc), bid_volume_pc))
        orders_pc.append(Order(product_pc, round(ask_price_pc), -ask_volume_pc))
        orders_c.append(Order(product_c, round(bid_price_c), bid_volume_c))
        orders_c.append(Order(product_c, round(ask_price_c), -ask_volume_c))

        return orders_c, orders_pc

    def largest_number_less_than(self, numbers, target):
        # First, sort the list in descending order
        numbers.sort(reverse=True)
        # Then, loop through the list and return the first number that is less than the target
        for number in numbers:
            if number < target:
                return number
        # If no number is less than the target, return None
        return None

    def get_orders_berries(self, state):
        global liquidated_flag
        if olivia_indicator['BERRIES'] == 1:
            orders = self.get_orders_berries_method_long(state)
            liquidated_flag = False
        elif olivia_indicator['BERRIES'] == -1:
            orders = self.get_orders_berries_method_short(state)
            liquidated_flag = False
        else: # olivia not taking any position
            if state.timestamp < REGIME_CUTOFFS[0]:
                if liquidated_flag == False:
                    # liquidate positions to 0
                    orders = self.get_orders_berries_liquidate(state)
                else:
                    orders = self.get_orders_berries_method_mm(state)
            elif (state.timestamp >= REGIME_CUTOFFS[0] and state.timestamp < REGIME_CUTOFFS[1]):
                orders = self.get_orders_berries_method_long(state)
                liquidated_flag = False
            elif (state.timestamp >= REGIME_CUTOFFS[1] and state.timestamp < REGIME_CUTOFFS[2]):
                orders = self.get_orders_berries_method_short(state)
                liquidated_flag = False
            # need liquidate position
            else:
                if liquidated_flag == False:
                    # liquidate position to 0
                    orders = self.get_orders_berries_liquidate(state)
                else:
                    orders = self.get_orders_berries_method_mm(state)
        return orders

    def get_orders_berries_method_mm(self, state):
        # Retrieve the Order Depth and positions
        product = 'BERRIES'
        order_depth: OrderDepth = state.order_depths[product]
        if product in state.position:
            positions: int = state.position[product]
        else:
            positions: int = 0

        # going forward book not empty
        bid_book, ask_book, bid_level_cutoff, ask_level_cutoff, bid_keys_sorted, ask_keys_sorted = self.get_book(order_depth)
        # undervalued_index = adjusted_theo - mid 
        orders = []
        remaining_bid_quote_volume = TRADING_LIMITS[product] - positions
        remaining_ask_quote_volume = TRADING_LIMITS[product] + positions

        mid = statistics.mean([bid_keys_sorted[0], ask_keys_sorted[0]])
        rolling_mid_berries.append(mid)
        if len(rolling_mid_berries) > ROLLING_MID_N_BERRY:
            rolling_mid_berries.pop(0)

        best_bid = bid_keys_sorted[0]
        best_ask = ask_keys_sorted[0]
        rolling_mid_avg = statistics.mean(rolling_mid_berries)

        # buy orders
        if remaining_bid_quote_volume > 0:

            # method 2
            if best_ask - best_bid <= 5: # squeeze
                if abs(mid - rolling_mid_avg) < 1: # if rolling and mid is similar
                    orders.append(Order(product, round((mid - 2) - positions * POSITION_SCALAR_BERRY), remaining_bid_quote_volume))
                else: # rolling and mid not similar (at a spike)
                    if mid > rolling_mid_avg: # up spike (green)
                        orders.append(Order(product, round((rolling_mid_avg - 3) - positions * POSITION_SCALAR_BERRY), remaining_bid_quote_volume))
                    else: # down spike (red)
                        # pick up the red littl guy(s) (buy up all the sells that are less than rolling - 0.5)
                        bids = {} # price as key, order as value
                        for price in ask_keys_sorted:
                            if price < rolling_mid_avg - 0.5:
                                order = Order(product, price, min(remaining_bid_quote_volume, ask_book[price]))
                                remaining_bid_quote_volume -= min(remaining_bid_quote_volume, ask_book[price])
                                bids[price] = order
                        if remaining_bid_quote_volume > 0:
                            bid_price = round((best_bid + 1) - positions * POSITION_SCALAR_BERRY)
                            if bid_price in bids:
                                order = Order(product, bid_price, bids[bid_price].quantity + remaining_bid_quote_volume)
                            else:
                                order = Order(product, bid_price, remaining_bid_quote_volume) # submit order as normal for remaining available bid volume
                            bids[bid_price] = order
                            for price in bids: # iterate through all the bids and append to orders list
                                orders.append(bids[price])
            else: # no squeeze, submit as usual
                orders.append(Order(product, round((best_bid + 1) - positions * POSITION_SCALAR_BERRY), remaining_bid_quote_volume))
        
        # sell orders
        if remaining_ask_quote_volume > 0:
    
            # method 2
            if best_ask - best_bid <= 5: # squeeze
                if abs(mid - rolling_mid_avg) < 1: # if rolling and mid is similar
                    orders.append(Order(product, round((mid + 2) - positions * POSITION_SCALAR_BERRY), -remaining_ask_quote_volume))
                else: # rolling and mid not similar (at a spike)
                    if mid < rolling_mid_avg: # down spike (red)
                        orders.append(Order(product, round((rolling_mid_avg + 3) - positions * POSITION_SCALAR_BERRY), -remaining_ask_quote_volume))
                    else: # up spike (green)
                        asks = {}
                        for price in bid_keys_sorted:
                            if price > rolling_mid_avg + 0.5:
                                order = Order(product, price, -min(remaining_ask_quote_volume, bid_book[price]))
                                remaining_ask_quote_volume -= min(remaining_ask_quote_volume, bid_book[price])
                                asks[price] = order
                        if remaining_ask_quote_volume > 0:
                            ask_price = round((best_ask - 1) - positions * POSITION_SCALAR_BERRY)
                            if ask_price in asks:
                                order = Order(product, ask_price, asks[ask_price].quantity - remaining_ask_quote_volume)
                            else:
                                order = Order(product, ask_price, -remaining_ask_quote_volume) # submit order as normal for remaining available ask volume
                            asks[ask_price] = order
                            for price in asks: # iterate through all the asks and append to orders list
                                orders.append(asks[price])
            else: # no squeeze, submit as usual
                orders.append(Order(product, round((best_ask - 1) - positions * POSITION_SCALAR_BERRY), -remaining_ask_quote_volume))
            
        return orders
    
    def get_orders_berries_method_long(self, state):
        # Retrieve the Order Depth and positions
        product = 'BERRIES'
        order_depth: OrderDepth = state.order_depths[product]
        if product in state.position:
            positions: int = state.position[product]
        else:
            positions: int = 0

        # going forward book not empty
        bid_book, ask_book, bid_level_cutoff, ask_level_cutoff, bid_keys_sorted, ask_keys_sorted = self.get_book(order_depth)
        # undervalued_index = adjusted_theo - mid 
        orders = []
        remaining_bid_quote_volume = TRADING_LIMITS[product] - positions
        remaining_ask_quote_volume = TRADING_LIMITS[product] + positions

        mid = statistics.mean([bid_keys_sorted[0], ask_keys_sorted[0]])
        rolling_mid_berries.append(mid)
        if len(rolling_mid_berries) > ROLLING_MID_N_BERRY:
            rolling_mid_berries.pop(0)

        best_bid = bid_keys_sorted[0]
        best_ask = ask_keys_sorted[0]
        rolling_mid_avg = statistics.mean(rolling_mid_berries)

        # buy max amount (by crossing the book)
        order = Order(product, best_ask, min(remaining_bid_quote_volume, ask_book[best_ask]))
        # # buy max amount by limit order
        # order = Order(product, best_bid + 3, remaining_bid_quote_volume)
        orders.append(order)

        # # buy max amount (by limit order)
        # order = Order(product, best_bid + 1, remaining_bid_quote_volume)
        # orders.append(order)

        return orders

    def get_orders_berries_method_short(self, state):
        # Retrieve the Order Depth and positions
        product = 'BERRIES'
        order_depth: OrderDepth = state.order_depths[product]
        if product in state.position:
            positions: int = state.position[product]
        else:
            positions: int = 0

        # going forward book not empty
        bid_book, ask_book, bid_level_cutoff, ask_level_cutoff, bid_keys_sorted, ask_keys_sorted = self.get_book(order_depth)
        # undervalued_index = adjusted_theo - mid 
        orders = []
        remaining_bid_quote_volume = TRADING_LIMITS[product] - positions
        remaining_ask_quote_volume = TRADING_LIMITS[product] + positions

        mid = statistics.mean([bid_keys_sorted[0], ask_keys_sorted[0]])
        rolling_mid_berries.append(mid)
        if len(rolling_mid_berries) > ROLLING_MID_N_BERRY:
            rolling_mid_berries.pop(0)

        best_bid = bid_keys_sorted[0]
        best_ask = ask_keys_sorted[0]
        rolling_mid_avg = statistics.mean(rolling_mid_berries)

        # sell max amount (by crossing the book)
        order = Order(product, best_bid, -min(remaining_ask_quote_volume, bid_book[best_bid]))
        # # sell max amount by limit order
        # order = Order(product, best_ask - 3, -remaining_ask_quote_volume)
        orders.append(order)

        # # sell max amount (by limit order)
        # order = Order(product, best_ask - 1, -remaining_ask_quote_volume)
        # orders.append(order)

        return orders
    
    def get_orders_berries_liquidate(self, state):
        global liquidated_flag
        # Retrieve the Order Depth and positions
        product = 'BERRIES'
        order_depth: OrderDepth = state.order_depths[product]
        if product in state.position:
            positions: int = state.position[product]
        else:
            positions: int = 0

        # going forward book not empty
        bid_book, ask_book, bid_level_cutoff, ask_level_cutoff, bid_keys_sorted, ask_keys_sorted = self.get_book(order_depth)
        # undervalued_index = adjusted_theo - mid 
        orders = []
        remaining_bid_quote_volume = TRADING_LIMITS[product] - positions
        remaining_ask_quote_volume = TRADING_LIMITS[product] + positions

        mid = statistics.mean([bid_keys_sorted[0], ask_keys_sorted[0]])
        rolling_mid_berries.append(mid)
        if len(rolling_mid_berries) > ROLLING_MID_N_BERRY:
            rolling_mid_berries.pop(0)

        best_bid = bid_keys_sorted[0]
        best_ask = ask_keys_sorted[0]
        rolling_mid_avg = statistics.mean(rolling_mid_berries)

        # buy max amount (by crossing the book)
        if positions < 0:
            order = Order(product, best_ask, min(remaining_bid_quote_volume, ask_book[best_ask], abs(positions)))
            orders.append(order)
        elif positions > 0:
            order = Order(product, best_bid, min(remaining_ask_quote_volume, bid_book[best_bid], -abs(positions)))
            orders.append(order)
        else: #positions == 0
            liquidated_flag = True

        return orders

    def get_orders_gear(self, state):
        global longing_flag, shorting_flag, prev_dolf, time_since_last_event
        assert(longing_flag == False or shorting_flag == False)
        #get stuff needed
        orders = []
        product = 'DIVING_GEAR'
        order_depth = state.order_depths[product]
        bid_book, ask_book, bid_level_cutoff, ask_level_cutoff, bid_keys_sorted, ask_keys_sorted = self.get_book(order_depth)
        best_ask = ask_keys_sorted[0]
        best_bid = bid_keys_sorted[0]

        mid = statistics.mean([bid_keys_sorted[0], ask_keys_sorted[0]])
        rolling_mid_gear.append(mid)
        if len(rolling_mid_gear) > ROLLING_MID_N_GEAR:
            rolling_mid_gear.pop(0)
        
        # calculate slope based on rolling rolling mid avg to avoid being impacted by small + or - slopes
        rolling_mid_avg = statistics.mean(rolling_mid_gear)
        rolling_rolling_mid_gear.append(rolling_mid_avg)
        if len(rolling_rolling_mid_gear) > ROLLING_ROLLING_MID_N_GEAR:
            rolling_rolling_mid_gear.pop(0)
        
        slope = (rolling_rolling_mid_gear[-1] - rolling_rolling_mid_gear[0]) / len(rolling_rolling_mid_gear)

        if product in state.position:
            positions: int = state.position[product]
        else:
            positions: int = 0
        
        remaining_bid_quote_volume = TRADING_LIMITS[product] - positions
        remaining_ask_quote_volume = TRADING_LIMITS[product] + positions

        dolf = state.observations["DOLPHIN_SIGHTINGS"]
        # #debug
        # if state.timestamp == 100:
        #     dolf = prev_dolf + 6
        # -------------------- strat
        if prev_dolf >= 0 and abs(dolf - prev_dolf) > DOLPHIN_DIFF_CUTOFF: # if peak or drop in dolphins 
            time_since_last_event = 0
            if dolf - prev_dolf > DOLPHIN_DIFF_CUTOFF:
                # we buy
                longing_flag = True
                shorting_flag = False
            else:
                # we sell
                longing_flag = False
                shorting_flag = True

        else: # no event
            time_since_last_event += 1
            if time_since_last_event >= MIN_MOMENTUM_TIME:
                if longing_flag == True:
                    if slope < 0:
                        longing_flag = False # stop buying

                if shorting_flag == True:
                    if slope > 0:
                        shorting_flag = False # stop selling
        
        # submit orders based on our longing/shorting flags
        if longing_flag == True:
            order = Order(product, best_ask, remaining_bid_quote_volume)
            orders.append(order)
        if shorting_flag == True:
            order = Order(product, best_bid, -remaining_ask_quote_volume)
            orders.append(order)
        if longing_flag == False and shorting_flag == False:
            # if positions not equal to 0, we liquidate back to 0
            if positions > 0:
                # sell
                order = Order(product, best_bid, -positions)
                orders.append(order)
            elif positions < 0:
                order = Order(product, best_ask, -positions)
                orders.append(order)
        
        prev_dolf = dolf

        return orders

    def get_orders_picnic_dip_baguette_ukulele(self, state):
        # orders_basket, orders_baguette, orders_dip, orders_ukulele = self.get_orders_picnic_method1(state)
        orders_basket, orders_baguette, orders_dip, orders_ukulele = self.get_orders_picnic_method1(state)
        return orders_basket, orders_baguette, orders_dip, orders_ukulele

    def get_orders_picnic_method1(self, state):
        global liquidated_from_olivia_strat, olivia_indicator, olivia_position
        #get stuff needed
        # get etf stuff
        orders_basket = []
        product_basket = 'PICNIC_BASKET'
        order_depth_basket = state.order_depths[product_basket]
        bid_book_basket, ask_book_basket, bid_level_cutoff_basket, ask_level_cutoff_basket, bid_keys_sorted_basket, ask_keys_sorted_basket = self.get_book(order_depth_basket)
        best_ask_basket = ask_keys_sorted_basket[0]
        best_bid_basket = bid_keys_sorted_basket[0]

        mid_basket = statistics.mean([best_bid_basket, best_ask_basket])

        # mid_basket = statistics.mean([bid_keys_sorted_basket[0], ask_keys_sorted_basket[0]])
        # rolling_mid_basket.append(mid_basket)
        # if len(rolling_mid_basket) > ROLLING_MID_N_BASKET:
        #     rolling_mid_basket.pop(0)

        if product_basket in state.position:
            positions_basket: int = state.position[product_basket]
        else:
            positions_basket: int = 0

        remaining_bid_quote_volume_basket = TRADING_LIMITS[product_basket] - positions_basket
        remaining_ask_quote_volume_basket = TRADING_LIMITS[product_basket] + positions_basket
        
        # get DIP stuff
        orders_dip = []
        product_dip = 'DIP'
        order_depth_dip = state.order_depths[product_dip]
        bid_book_dip, ask_book_dip, bid_level_cutoff_dip, ask_level_cutoff_dip, bid_keys_sorted_dip, ask_keys_sorted_dip = self.get_book(order_depth_dip)
        best_ask_dip = ask_keys_sorted_dip[0]
        best_bid_dip = bid_keys_sorted_dip[0]

        mid_dip = statistics.mean([bid_keys_sorted_dip[0], ask_keys_sorted_dip[0]])

        if product_dip in state.position:
            positions_dip: int = state.position[product_dip]
        else:
            positions_dip: int = 0
        
        remaining_bid_quote_volume_dip = TRADING_LIMITS[product_dip] - positions_dip
        remaining_ask_quote_volume_dip = TRADING_LIMITS[product_dip] + positions_dip

        # get BAGUETTE stuff
        orders_baguette = []
        product_baguette = 'BAGUETTE'
        order_depth_baguette = state.order_depths[product_baguette]
        bid_book_baguette, ask_book_baguette, bid_level_cutoff_baguette, ask_level_cutoff_baguette, bid_keys_sorted_baguette, ask_keys_sorted_baguette = self.get_book(order_depth_baguette)
        best_ask_baguette = ask_keys_sorted_baguette[0]
        best_bid_baguette = bid_keys_sorted_baguette[0]

        mid_baguette = statistics.mean([bid_keys_sorted_baguette[0], ask_keys_sorted_baguette[0]])

        if product_baguette in state.position:
            positions_baguette: int = state.position[product_baguette]
        else:
            positions_baguette: int = 0
        
        remaining_bid_quote_volume_baguette = TRADING_LIMITS[product_baguette] - positions_baguette
        remaining_ask_quote_volume_baguette = TRADING_LIMITS[product_baguette] + positions_baguette

        # get UKULELE stuff
        orders_ukulele = []
        product_ukulele = 'UKULELE'
        order_depth_ukulele = state.order_depths[product_ukulele]
        bid_book_ukulele, ask_book_ukulele, bid_level_cutoff_ukulele, ask_level_cutoff_ukulele, bid_keys_sorted_ukulele, ask_keys_sorted_ukulele = self.get_book(order_depth_ukulele)
        best_ask_ukulele = ask_keys_sorted_ukulele[0]
        best_bid_ukulele = bid_keys_sorted_ukulele[0]

        mid_ukulele = statistics.mean([bid_keys_sorted_ukulele[0], ask_keys_sorted_ukulele[0]])

        if product_ukulele in state.position:
            positions_ukulele: int = state.position[product_ukulele]
        else:
            positions_ukulele: int = 0
        
        remaining_bid_quote_volume_ukulele = TRADING_LIMITS[product_ukulele] - positions_ukulele
        remaining_ask_quote_volume_ukulele = TRADING_LIMITS[product_ukulele] + positions_ukulele

        #---------------------
        best_ask_equivalent_basket = 2 * best_ask_baguette + 4 * best_ask_dip + best_ask_ukulele # can buy equivalent basket for
        best_bid_equivalent_basket = 2 * best_bid_baguette + 4 * best_bid_dip + best_bid_ukulele # can sell equivalent basket for
        mid_equivalent_basket = (best_ask_equivalent_basket + best_bid_equivalent_basket) / 2

        diff = mid_basket - mid_equivalent_basket - ETF_PREMIUM
        print('diff = {}'.format(diff))
        
        if olivia_indicator['UKULELE'] != 0: # if Olivia has a position
            print('Olivia has position')
            # follow Olivia's position for ukulele
            if olivia_indicator['UKULELE'] == 1:
                orders_ukulele.append(Order(product_ukulele, best_ask_ukulele, remaining_bid_quote_volume_ukulele))
            elif olivia_indicator['UKULELE'] == -1:
                orders_ukulele.append(Order(product_ukulele, best_bid_ukulele, -remaining_ask_quote_volume_ukulele))
            
            # trade the difference for all other products
            # if we are converging back to zero (back to ETF_PREMIUM), then liquidate FAST TODO:(with both market and limit?)
            if diff <= START_LIQUIDATING_CUTOFF and (positions_basket < 0 or positions_baguette > 0 or positions_dip > 0): # diff just dropped back down to 0
                # buy basket, sell underlying
                if positions_basket < 0:
                    orders_basket.append(Order(product_basket, round(best_ask_basket), -positions_basket)) # buy basket
                if positions_baguette > 0:
                    orders_baguette.append(Order(product_baguette, round(best_bid_baguette), -positions_baguette)) # sell underlying
                if positions_dip > 0:
                    orders_dip.append(Order(product_dip, round(best_bid_dip), -positions_dip)) # sell underlying
                # if positions_ukulele > 0:
                #     orders_ukulele.append(Order(product_ukulele, round(best_bid_ukulele), -positions_ukulele)) # sell underlying

            elif diff >= -START_LIQUIDATING_CUTOFF and (positions_basket > 0 or positions_baguette < 0 or positions_dip < 0): # diff just increased back up to 0
                # sell basket, buy underlying
                if positions_basket > 0:
                    orders_basket.append(Order(product_basket, round(best_bid_basket), -positions_basket)) # sell basket
                if positions_baguette < 0:
                    orders_baguette.append(Order(product_baguette, round(best_ask_baguette), -positions_baguette)) # buy underlying
                if positions_dip < 0:
                    orders_dip.append(Order(product_dip, round(best_ask_dip), -positions_dip)) # buy underlying
                # if positions_ukulele < 0:
                #     orders_ukulele.append(Order(product_ukulele, round(best_ask_ukulele), -positions_ukulele)) # buy underlying

            # otherwise (building up position), then trade max amount allowed by fractional restriction
            else:
                fractional_pos_lim = FRACTIONAL_POS_LIMIT_BASKET[self.largest_number_less_than(list(FRACTIONAL_POS_LIMIT_BASKET.keys()), abs(diff))]
                print('fractional positional limit = {}'.format(fractional_pos_lim))

                bid_volume_basket = round(min(remaining_bid_quote_volume_basket, abs(diff) * SPEED_FACTOR_BASKET, max(int(TRADING_LIMITS[product_basket] * fractional_pos_lim) - positions_basket, 0)))
                ask_volume_basket = round(min(remaining_ask_quote_volume_basket, abs(diff) * SPEED_FACTOR_BASKET, max(positions_basket + int(TRADING_LIMITS[product_basket] * fractional_pos_lim), 0)))

                bid_volume_baguette = round(min(remaining_bid_quote_volume_baguette, abs(diff) * SPEED_FACTOR_BASKET * 2, max(int(TRADING_LIMITS[product_baguette] * fractional_pos_lim) - positions_baguette, 0)))
                ask_volume_baguette = round(min(remaining_ask_quote_volume_baguette, abs(diff) * SPEED_FACTOR_BASKET * 2, max(positions_baguette + int(TRADING_LIMITS[product_baguette] * fractional_pos_lim), 0)))

                bid_volume_dip = round(min(remaining_bid_quote_volume_dip, abs(diff) * SPEED_FACTOR_BASKET * 4, max(int(TRADING_LIMITS[product_dip] * fractional_pos_lim) - positions_dip, 0)))
                ask_volume_dip = round(min(remaining_ask_quote_volume_dip, abs(diff) * SPEED_FACTOR_BASKET * 4, max(positions_dip + int(TRADING_LIMITS[product_dip] * fractional_pos_lim), 0)))

                if diff > basket_min_diff: 
                    # we want to sell basket and buy underlying
                    # number of market trades we can make
                    num_baskets_tradable = int(min(bid_book_basket[best_bid_basket], ask_book_baguette[best_ask_baguette] / 2, ask_book_dip[best_ask_dip] / 4))
                    print('num_baskets_tradable = {}'.format(num_baskets_tradable))
                    orders_basket.append(Order(product_basket, best_bid_basket, -min(num_baskets_tradable, ask_volume_basket))) # sell basket
                    orders_baguette.append(Order(product_baguette, best_ask_baguette, min(num_baskets_tradable * 2, bid_volume_baguette))) # buy underlying
                    orders_dip.append(Order(product_dip, best_ask_dip, min(num_baskets_tradable * 4, bid_volume_dip))) # buy underlying

                elif diff < -basket_min_diff:
                    # we want to buy basket and sell underlying
                    # number of market trades we can make
                    num_baskets_tradable = int(min(ask_book_basket[best_ask_basket], bid_book_baguette[best_bid_baguette] / 2, bid_book_dip[best_bid_dip] / 4))
                    orders_basket.append(Order(product_basket, best_ask_basket, min(num_baskets_tradable, bid_volume_basket))) # buy basket
                    orders_baguette.append(Order(product_baguette, best_bid_baguette, -min(num_baskets_tradable * 2, ask_volume_baguette))) # sell underlying
                    orders_dip.append(Order(product_dip, best_bid_dip, -min(num_baskets_tradable * 4, ask_volume_dip))) # sell underlying
                else: 
                    # Market make
                    pass
                    # return self.get_orders_picnic_method2(state)
            
            liquidated_from_olivia_strat = False

        else: # Olivia does not have a position
            print("Olivia does not have a position")
            #if ukulele position is not the same as the others then reduce to the same (while others wait)
            if liquidated_from_olivia_strat == False:
                print('liquidated olivia flag is false at the beginning of this iteration')
                if positions_ukulele < -positions_basket: # if ukeklele position is less than what it is suppsoed to be at
                    # buy ukulele
                    orders_ukulele.append(Order(product_ukulele, round(best_ask_ukulele), -positions_basket - positions_ukulele)) # buy ukulele
                elif positions_ukulele > -positions_basket: # if ukulele position is more than what it is supposed to be at
                    # sell ukulele
                    orders_ukulele.append(Order(product_ukulele, round(best_bid_ukulele), -(positions_ukulele + positions_basket))) # sell ukulele
                else: # ukulele position is as supposed to be
                    liquidated_from_olivia_strat = True

            if liquidated_from_olivia_strat == True: # only execute diff trading strat if olivia strat when ukulele position is correct
                print('liquidated from olivia strat')
                # if we are converging back to zero (back to ETF_PREMIUM), then liquidate FAST TODO:(with both market and limit?)
                if diff <= START_LIQUIDATING_CUTOFF and (positions_basket < 0 or positions_ukulele > 0 or positions_baguette > 0 or positions_dip > 0): # diff just dropped back down to 0
                    print('diff just dropped back down to near 0, trying to liquidate')
                    # buy basket, sell underlying
                    if positions_basket < 0:
                        orders_basket.append(Order(product_basket, round(best_ask_basket), -positions_basket)) # buy basket
                    if positions_baguette > 0:
                        orders_baguette.append(Order(product_baguette, round(best_bid_baguette), -positions_baguette)) # sell underlying
                    if positions_dip > 0:
                        orders_dip.append(Order(product_dip, round(best_bid_dip), -positions_dip)) # sell underlying
                    if positions_ukulele > 0:
                        orders_ukulele.append(Order(product_ukulele, round(best_bid_ukulele), -positions_ukulele)) # sell underlying

                elif diff >= -START_LIQUIDATING_CUTOFF and (positions_basket > 0 or positions_ukulele < 0 or positions_baguette < 0 or positions_dip < 0): # diff just increased back up to 0
                    # sell basket, buy underlying
                    print('diff just increased back up to near 0, trying to liquidate')
                    if positions_basket > 0:
                        orders_basket.append(Order(product_basket, round(best_bid_basket), -positions_basket)) # sell basket
                    if positions_baguette < 0:
                        orders_baguette.append(Order(product_baguette, round(best_ask_baguette), -positions_baguette)) # buy underlying
                    if positions_dip < 0:
                        orders_dip.append(Order(product_dip, round(best_ask_dip), -positions_dip)) # buy underlying
                    if positions_ukulele < 0:
                        orders_ukulele.append(Order(product_ukulele, round(best_ask_ukulele), -positions_ukulele)) # buy underlying

                # otherwise (building up position), then trade max amount allowed by fractional restriction
                else:
                    print('difference is large or all of the underlyings are liquidated OR we didnt finish liquidating but now diff is on the other side')
                    fractional_pos_lim = FRACTIONAL_POS_LIMIT_BASKET[self.largest_number_less_than(list(FRACTIONAL_POS_LIMIT_BASKET.keys()), abs(diff))]

                    bid_volume_basket = round(min(remaining_bid_quote_volume_basket, abs(diff) * SPEED_FACTOR_BASKET, max(int(TRADING_LIMITS[product_basket] * fractional_pos_lim) - positions_basket, 0)))
                    ask_volume_basket = round(min(remaining_ask_quote_volume_basket, abs(diff) * SPEED_FACTOR_BASKET, max(positions_basket + int(TRADING_LIMITS[product_basket] * fractional_pos_lim), 0)))

                    bid_volume_baguette = round(min(remaining_bid_quote_volume_baguette, abs(diff) * SPEED_FACTOR_BASKET * 2, max(int(TRADING_LIMITS[product_baguette] * fractional_pos_lim) - positions_baguette, 0)))
                    ask_volume_baguette = round(min(remaining_ask_quote_volume_baguette, abs(diff) * SPEED_FACTOR_BASKET * 2, max(positions_baguette + int(TRADING_LIMITS[product_baguette] * fractional_pos_lim), 0)))

                    bid_volume_dip = round(min(remaining_bid_quote_volume_dip, abs(diff) * SPEED_FACTOR_BASKET * 4, max(int(TRADING_LIMITS[product_dip] * fractional_pos_lim) - positions_dip, 0)))
                    ask_volume_dip = round(min(remaining_ask_quote_volume_dip, abs(diff) * SPEED_FACTOR_BASKET * 4, max(positions_dip + int(TRADING_LIMITS[product_dip] * fractional_pos_lim), 0)))

                    bid_volume_ukulele = round(min(remaining_bid_quote_volume_ukulele, abs(diff) * SPEED_FACTOR_BASKET * 1, max(int(TRADING_LIMITS[product_ukulele] * fractional_pos_lim) - positions_ukulele, 0)))
                    ask_volume_ukulele = round(min(remaining_ask_quote_volume_ukulele, abs(diff) * SPEED_FACTOR_BASKET * 1, max(positions_ukulele + int(TRADING_LIMITS[product_ukulele] * fractional_pos_lim), 0)))

                    if diff > basket_min_diff: 
                        print('diff is very positive. we want to sell basket and buy underlying')
                        # we want to sell basket and buy underlying
                        # number of market trades we can make
                        num_baskets_tradable = int(min(bid_book_basket[best_bid_basket], ask_book_baguette[best_ask_baguette] / 2, ask_book_dip[best_ask_dip] / 4, ask_book_ukulele[best_ask_ukulele]))
                        orders_basket.append(Order(product_basket, best_bid_basket, -min(num_baskets_tradable, ask_volume_basket))) # sell basket
                        orders_baguette.append(Order(product_baguette, best_ask_baguette, min(num_baskets_tradable * 2, bid_volume_baguette))) # buy underlying
                        orders_dip.append(Order(product_dip, best_ask_dip, min(num_baskets_tradable * 4, bid_volume_dip))) # buy underlying
                        orders_ukulele.append(Order(product_ukulele, best_ask_ukulele, min(num_baskets_tradable * 1, bid_volume_ukulele))) # buy underlying

                    elif diff < -basket_min_diff:
                        print('diff is very negative. we want to buy basket and sell underlying')
                        # we want to buy basket and sell underlying
                        # number of market trades we can make
                        num_baskets_tradable = int(min(ask_book_basket[best_ask_basket], bid_book_baguette[best_bid_baguette] / 2, bid_book_dip[best_bid_dip] / 4, bid_book_ukulele[best_bid_ukulele]))
                        orders_basket.append(Order(product_basket, best_ask_basket, min(num_baskets_tradable, bid_volume_basket))) # buy basket
                        orders_baguette.append(Order(product_baguette, best_bid_baguette, -min(num_baskets_tradable * 2, ask_volume_baguette))) # sell underlying
                        orders_dip.append(Order(product_dip, best_bid_dip, -min(num_baskets_tradable * 4, ask_volume_dip))) # sell underlying
                        orders_ukulele.append(Order(product_ukulele, best_bid_ukulele, -min(num_baskets_tradable * 1, ask_volume_ukulele))) # sell underlying
                    else: 
                        print('diff is small. do not trade')
                        # Market make
                        pass
                        # return self.get_orders_picnic_method2(state)

        return orders_basket, orders_baguette, orders_dip, orders_ukulele
    
    def get_monkey_indocators(self, state):
        global olivia_indicator, olivia_position
        for product, trades in state.market_trades.items():
            print(product)
            print(trades)
            if product in olivia_indicator:
                for trade in trades:
                    print("AAAAAAAAAAAAAAAA")
                    print(trade.timestamp, state.timestamp, trade.buyer, trade.seller)
                    if trade.timestamp == state.timestamp and trade.buyer == 'Olivia':
                        print("OLIVIA IS BUYER")
                        olivia_position[product] += trade.quantity
                    if trade.timestamp == state.timestamp and trade.seller == 'Olivia':
                        print('OLIVIA IS SELLER')
                        olivia_position[product] -= trade.quantity
                if olivia_position[product] > 0:
                    olivia_indicator[product] = 1
                elif olivia_position[product] < 0:
                    olivia_indicator[product] = -1
                else: # zero position
                    olivia_indicator[product] = 0
                    
        for product, trades in state.own_trades.items():
            if product in olivia_indicator:
                for trade in trades:
                    # print(trade.timestamp, state.timestamp, trade.buyer, trade.seller)
                    if (trade.timestamp == state.timestamp - 100) and (trade.buyer == 'Olivia'):
                        # print("OLIVIA IS BUYER")
                        olivia_position[product] += trade.quantity
                    if (trade.timestamp == state.timestamp - 100) and (trade.seller == 'Olivia'):
                        # print('OLIVIA IS SELLER')
                        olivia_position[product] -= trade.quantity
                if olivia_position[product] > 0:
                    olivia_indicator[product] = 1
                elif olivia_position[product] < 0:
                    olivia_indicator[product] = -1
                else: # zero position
                    olivia_indicator[product] = 0
        print(olivia_indicator)


        


        
