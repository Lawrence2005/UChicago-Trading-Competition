from typing import Dict, Optional
import numpy as np
import abc
from utcxchangelib import xchange_client
# import argparse
import asyncio

class Asset:
    """Base class for all assets (stocks/ETFs)."""
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.position = 0
        self.price = None

    def update_price(self, p: float) -> None:
        self.price = p

    @abc.abstractmethod
    def check_arbitrage(self) -> Optional[dict[str, int]]:
        pass

class APT(Asset):
    PE_RATIO: float = 10.0

    """Large-cap stock with earnings announcements."""
    def __init__(self):
        super().__init__("APT")


class DLR(Asset):
    """
    DLR: Mid-cap stock tied to a binary petition outcome.
    Fair value is 100 if >=100,000 signatures are collected by the end of day 10, else 0.
    Signature growth follows a lognormal process:
        S_i ~ LogNormal(log(alpha) + log(S_{i-1}), sigma^2)
    """

    def __init__(self):
        self.symbol = "DLR"
        self.position = 0
        self.price = None  # market price
        self.current_signatures = 5000  # initial value
        self.alpha = 1.0630449594499
        self.sigma = 0.006
        self.time_step = 0  # current update index
        self.history = [self.current_signatures]

    def update_price(self, price: float):
        """Update market price from ticker feed."""
        self.price = price

    def update_signatures(self, new_signatures: int):
        """
        Update current signature count based on news release.
        Append to history and increment time step.
        """
        self.current_signatures += new_signatures
        self.history.append(self.current_signatures)
        self.time_step += 1

    def simulate_signature_paths(self, num_days_left: int, num_simulations: int = 1000) -> float:
        """
        Monte Carlo simulation of signature growth to estimate probability of hitting 100,000.
        """
        final_counts = []
        
        for _ in range(num_simulations):
            count = self.current_signatures
            for _ in range(num_days_left * 5):
                mu = np.log(self.alpha) + np.log(count)
                count = np.random.lognormal(mean=mu, sigma=self.sigma)
            final_counts.append(count)

        # Probability of reaching 100,000 signatures
        success_prob = np.mean(np.array(final_counts) >= 100000)

        return success_prob

    def compute_fair_value(self, current_day: int) -> float:
        """
        Estimate fair value as expected payout (100 or 0).
        """
        days_left = 10 - current_day
        if days_left <= 0:
            return 100.0 if self.current_signatures >= 100000 else 0.0
        prob = self.simulate_signature_paths(days_left)

        return 100 * prob
    
    def get_market_making_quotes(self, fair_value, spread=1.0):
        bid = fair_value - 0.5 * spread
        ask = fair_value + 0.5 * spread

        return bid, ask





class MKJ(Asset):
#     """Small-cap stock with unstructured news."""
#     def __init__(self):
#         super().__init__("MKJ")

# class ETF(Asset):
#     FEE: float

#     """Base class for ETFs (AKAV, AKIM)."""
#     def __init__(self, symbol: str):
#         super().__init__(symbol)
#         self.FEE = 100

# class AKAV(ETF):
#     """ETF composed of APT, DLR, MKJ."""
#     components: list[str]

#     def __init__(self):
#         super().__init__("AKAV")
#         self.components = ["APT", "DLR", "MKJ"]

#     def calculate_nav(self, assets: Dict[str, Asset]) -> float:
#         return sum(assets[symbol].price for symbol in self.components)

#     def check_arbitrage(self, assets: Dict[str, Asset]) -> Optional[Dict[str, int]]:
#         """Return trades if arbitrage exists."""
#         nav = self.calculate_nav(assets)
#         if self.price > nav + self.fee:
#             return {self.symbol: -1, **{symbol: 1 for symbol in self.components}}
#         elif self.price < nav - self.fee:
#             return {self.symbol: 1, **{symbol: -1 for symbol in self.components}}
#         return None

# class AKIM(ETF):
#     def __init__(self):
#         super().__init__("AKIM")

#     ##TODO##

# class TradingBot:
#     """Orchestrates all trading logic."""

#     MAX_ORDER_SIZE: int = 40
#     MAX_OPEN_ORDERS: int = 50
#     MAX_OUTSTANDING_VOLUME: int = 120
#     MAX_ABSOLUTE_POSITION: int = 200

#     assets: dict[str, Asset]
#     open_orders: list[tuple[str, int]]

#     def __init__(self):
#         self.assets = {
#             "APT": Asset("APT"),
#             "DLR": Asset("DLR"),  # Mid-cap stock
#             "MKJ": Asset("MKJ"),  # Small-cap stock
#             "AKAV": AKAV(),
#             "AKIM": AKIM()   # Inverse ETF
#         }
#         self.open_orders = []

    def update_signatures_for_dlr(self, new_signatures: int, current_day: int):
        dlr: DLR = self.assets["DLR"]
        dlr.update_signatures(new_signatures)
        fair_value = dlr.compute_fair_value(current_day)
        bid, ask = dlr.get_market_making_quotes(fair_value)
        print(f"[DLR STRATEGY] Fair Value = {fair_value:.2f}, Bid = {bid:.2f}, Ask = {ask:.2f}, Signatures = {dlr.current_signatures}")
        # Optional: place bid/ask orders here
    

#     def update_market_data(self, prices: Dict[str, float]) -> None:
#         """Update all asset prices."""
#         for symbol, price in prices.items():
#             self.assets[symbol].update_price(price)

#     def execute_trades(self, trades: Dict[str, int]) -> bool:
#         """Update positions after trades."""

#         if len(self.open_orders) + 1 > self.MAX_OPEN_ORDERS:
#             print(f"[RISK] Blocked: MAX_OPEN_ORDERS ({self.MAX_OPEN_ORDERS}) reached")
#             return False
        
#         new_volume = 0
#         for symbol, qty in trades.items():
#             if abs(qty) > self.MAX_ORDER_SIZE:
#                 print(f"[RISK] Blocked: Order for {symbol} exceeds MAX_ORDER_SIZE ({self.MAX_ORDER_SIZE})")
#                 return False

#             if abs(self.assets[symbol].position + qty) > self.MAX_ABSOLUTE_POSITION:
#                 print(f"[RISK] Blocked: Position for {symbol} would exceed MAX_ABSOLUTE_POSITION ({self.MAX_ABSOLUTE_POSITION})")
#                 return False
            
#             new_volume += abs(qty)
        
#         current_volume = sum(abs(o[1]) for o in self.open_orders)
#         if current_volume + new_volume > self.MAX_OUTSTANDING_VOLUME:
#             print(f"[RISK] Blocked: MAX_OUTSTANDING_VOLUME ({self.MAX_OUTSTANDING_VOLUME}) exceeded")
#             return False
    
#         for symbol, amt in trades.items():
#             self.assets[symbol].position += amt
#         print(f"[EXECUTED] Trades: {trades}")
#         return True

#     def run_arbitrage(self) -> None:
#         """Check and execute AKAV arbitrage."""
#         trades = self.assets["AKAV"].check_arbitrage(self.assets)
#         if trades:
#             valid = self.execute_trades(trades)
#             if not valid:
#                 print("[INFO] Arbitrage opportunity skipped due to risk limits")

#     def run(self, prices: Dict[str, float]) -> None:
#         """Main loop."""
#         self.update_market_data(prices)
#         self.run_arbitrage()
#         print(f"Positions: {[f'{a.symbol}: {a.position}' for a in self.assets.values()]}")

class MyXchangeClient(xchange_client.XChangeClient):
    '''A shell client with the methods that can be implemented to interact with the xchange.'''

    def __init__(self, host: str, username: str, password: str):
        super().__init__(host, username, password)
        self.pe_ratio = 10 
        self.signature_count = 0
        self.DLR_std_dev = .006
        self.apt_theo: int
        self.dlr_theo: int
        self.mkj_theo: int
        self.akav_theo: int
        self.akim_theo: int
    
    async def bot_handle_cancel_response(self, order_id: str, success: bool, error: Optional[str]) -> None:
        order = self.open_orders[order_id]
        print(f"{'Market' if order[2] else 'Limit'} Order ID {order_id} cancelled, {order[1]} unfilled")

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        print("order fill", self.positions)

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        print("order rejected because of ", reason)

    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int):
        pass

    async def bot_handle_book_update(self, symbol: str) -> None:
        pass

    async def bot_handle_swap_response(self, swap: str, qty: int, success: bool):
        pass

    async def bot_handle_news(self, news_release: dict):
        # Parsing the message based on what type was received
        timestamp = news_release["timestamp"] # This is in exchange ticks not ISO or Epoch
        news_type = news_release['kind']
        news_data = news_release["new_data"]
        print(news_release)
        if news_type == "structured":
            subtype = news_data["structured_subtype"]
            symb = news_data["asset"]
            if subtype == "earnings":
                earnings = news_data["value"]
                self.apt_theo = self.pe_ratio * earnings

                # GONNA NEED TO UNFUCK THIS, INIT FAIR PRICES FOR EACH SECURITY AND UPDATE W/EACH FN, TRADE IN ONLY ONE FN AND 
                #await self.place_order(symb, 5, xchange_client.Side.BUY, new_price - 50) # 50 is arbitrary constant, make dynamic later
                #await self.place_order(symb,5,xchange_client.Side.SELL, new_price + 50)
                
            else:
                new_signatures = news_data["new_signatures"]
                cumulative = news_data["cumulative"]
                ### Do something with this data ###
        else:
            for i in self.open_orders.keys():
                self.bot_handle_cancel_response()

    async def view_books(self):
        while True:
            await asyncio.sleep(3)
            for security, book in self.order_books.items():
                sorted_bids = sorted((k,v) for k,v in book.bids.items() if v != 0)
                sorted_asks = sorted((k,v) for k,v in book.asks.items() if v != 0)
                print(f"Bids for {security}:\n{sorted_bids}")
                print(f"Asks for {security}:\n{sorted_asks}")

    async def start(self, user_interface):
        asyncio.create_task(self.trade())

        # This is where Phoenixhood will be launched if desired. There is no need to change these
        # lines, you can either remove the if or delete the whole thing depending on your purposes.
        if user_interface:
            self.launch_user_interface()
            asyncio.create_task(self.handle_queued_messages())

        await self.connect()

    async def trade(self):
        """    
        Examples of various XChangeClient actions (limit orders, order cancel, swaps, and market orders)
        """
        # Pause for 5 seconds before starting the trading sequence
        await asyncio.sleep(5)
        print("attempting to trade")

        # Place a BUY limit order for 3 units of APT at price 5
        await self.place_order("APT", 3, xchange_client.Side.BUY, 5)

        # Place a SELL limit order for 3 units of APT at price 7
        await self.place_order("APT", 3, xchange_client.Side.SELL, 7)

        # Pause for 5 seconds to allow orders to be processed
        await asyncio.sleep(5)

        # Cancel the first open order by retrieving its ID from open_orders
        if self.open_orders:
            await self.cancel_order(list(self.open_orders.keys())[0])

        # Place a swap order to swap 1 unit to AKAV
        await self.place_swap_order('toAKAV', 1)

        # Pause for 5 seconds to allow swap to process
        await asyncio.sleep(5)

        # Place a swap order to swap 1 unit from AKAV
        await self.place_swap_order('fromAKAV', 1)

        # Pause for 5 seconds after the swap
        await asyncio.sleep(5)

        # Place a large SELL limit order of 1000 units of APT at price 7
        await self.place_order("APT", 1000, xchange_client.Side.SELL, 7)

        # Pause for 5 seconds before placing a market order
        await asyncio.sleep(5)

        # Place a SELL market order for 10 units of APT
        # Market orders do not have a price, so the price parameter is omitted
        market_order_id = await self.place_order("APT", 10, xchange_client.Side.SELL)

        # Print the ID of the market order for reference
        print("MARKET ORDER ID:", market_order_id)

        # Pause for 5 seconds to allow order to settle
        await asyncio.sleep(5)

        # Print the current positions held after the sequence of trades
        print("my positions:", self.positions)

        for security, book in self.order_books.items():
            sorted_bids = sorted((k,v) for k,v in book.bids.items() if v != 0)
            sorted_asks = sorted((k,v) for k,v in book.asks.items() if v != 0)
            print(f"Bids for {security}:\n{sorted_bids}")
            print(f"Asks for {security}:\n{sorted_asks}")
            best_bid = sorted_bids[0][0]
            best_ask = sorted_asks[0][0]
            best_bid_vol = sorted_bids[0][1]
            best_ask_vol = sorted_asks[0][1]

            mkt_implied_price = (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol+best_ask_vol)



async def main():
    SERVER = '3.138.154.148:3333'
    my_client = MyXchangeClient(SERVER,"chicago7","^DmqJY6UUp")
    await my_client.start(None)
    return

if __name__ == "__main__":
    """....."""
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(main())