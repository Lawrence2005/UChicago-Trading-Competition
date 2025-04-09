from typing import Dict, Optional
import numpy as np
import abc
from utcxchangelib import xchange_client
#import xchange_client
import argparse
import asyncio

import utcxchangelib.xchange_client

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
    """Mid-cap stock dependent on petition signatures."""
    def __init__(self):
        super().__init__("DLR")

class MKJ(Asset):
    """Small-cap stock with unstructured news."""
    def __init__(self):
        super().__init__("MKJ")

class ETF(Asset):
    FEE: float

    """Base class for ETFs (AKAV, AKIM)."""
    def __init__(self, symbol: str):
        super().__init__(symbol)
        self.FEE = 100

class AKAV(ETF):
    """ETF composed of APT, DLR, MKJ."""
    components: list[str]

    def __init__(self):
        super().__init__("AKAV")
        self.components = ["APT", "DLR", "MKJ"]

    def calculate_nav(self, assets: Dict[str, Asset]) -> float:
        return sum(assets[symbol].price for symbol in self.components)

    def check_arbitrage(self, assets: Dict[str, Asset]) -> Optional[Dict[str, int]]:
        """Return trades if arbitrage exists."""
        nav = self.calculate_nav(assets)
        if self.price > nav + self.fee:
            return {self.symbol: -1, **{symbol: 1 for symbol in self.components}}
        elif self.price < nav - self.fee:
            return {self.symbol: 1, **{symbol: -1 for symbol in self.components}}
        return None

class AKIM(ETF):
    def __init__(self):
        super().__init__("AKIM")

    ##TODO##

class TradingBot:
    """Orchestrates all trading logic."""

    MAX_ORDER_SIZE: int = 40
    MAX_OPEN_ORDERS: int = 50
    MAX_OUTSTANDING_VOLUME: int = 120
    MAX_ABSOLUTE_POSITION: int = 200

    assets: dict[str, Asset]
    open_orders: list[tuple[str, int]]

    def __init__(self):
        self.assets = {
            "APT": Asset("APT"),
            "DLR": Asset("DLR"),  # Mid-cap stock
            "MKJ": Asset("MKJ"),  # Small-cap stock
            "AKAV": AKAV(),
            "AKIM": AKIM()   # Inverse ETF
        }
        self.open_orders = []

    def update_market_data(self, prices: Dict[str, float]) -> None:
        """Update all asset prices."""
        for symbol, price in prices.items():
            self.assets[symbol].update_price(price)

    def execute_trades(self, trades: Dict[str, int]) -> bool:
        """Update positions after trades."""

        if len(self.open_orders) + 1 > self.MAX_OPEN_ORDERS:
            print(f"[RISK] Blocked: MAX_OPEN_ORDERS ({self.MAX_OPEN_ORDERS}) reached")
            return False
        
        new_volume = 0
        for symbol, qty in trades.items():
            if abs(qty) > self.MAX_ORDER_SIZE:
                print(f"[RISK] Blocked: Order for {symbol} exceeds MAX_ORDER_SIZE ({self.MAX_ORDER_SIZE})")
                return False

            if abs(self.assets[symbol].position + qty) > self.MAX_ABSOLUTE_POSITION:
                print(f"[RISK] Blocked: Position for {symbol} would exceed MAX_ABSOLUTE_POSITION ({self.MAX_ABSOLUTE_POSITION})")
                return False
            
            new_volume += abs(qty)
        
        current_volume = sum(abs(o[1]) for o in self.open_orders)
        if current_volume + new_volume > self.MAX_OUTSTANDING_VOLUME:
            print(f"[RISK] Blocked: MAX_OUTSTANDING_VOLUME ({self.MAX_OUTSTANDING_VOLUME}) exceeded")
            return False
    
        for symbol, amt in trades.items():
            self.assets[symbol].position += amt
        print(f"[EXECUTED] Trades: {trades}")
        return True

    def run_arbitrage(self) -> None:
        """Check and execute AKAV arbitrage."""
        trades = self.assets["AKAV"].check_arbitrage(self.assets)
        if trades:
            valid = self.execute_trades(trades)
            if not valid:
                print("[INFO] Arbitrage opportunity skipped due to risk limits")

    def run(self, prices: Dict[str, float]) -> None:
        """Main loop."""
        self.update_market_data(prices)
        self.run_arbitrage()
        print(f"Positions: {[f'{a.symbol}: {a.position}' for a in self.assets.values()]}")

class MyXchangeClient(xchange_client.XChangeClient):
    '''A shell client with the methods that can be implemented to interact with the xchange.'''

    def __init__(self, host: str, username: str, password: str):
        super().__init__(host, username, password)
    
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

        if news_type == "structured":
            subtype = news_data["structured_subtype"]
            symb = news_data["asset"]
            if subtype == "earnings":
                earnings = news_data["value"]
                ### Do something with this data ###
            else:
                new_signatures = news_data["new_signatures"]
                cumulative = news_data["cumulative"]
                ### Do something with this data ###
        else:
            ### Not sure what you would do with unstructured data.... ###
            pass

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

        # This is where Phoenixhood will be launched if desired. There is no need to change these lines, you can either remove the if or delete the whole thing depending on your purposes.
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

    # for security, book in self.order_books.items():
    #     sorted_bids = sorted((k,v) for k,v in book.bids.items() if v != 0)
    #     sorted_asks = sorted((k,v) for k,v in book.asks.items() if v != 0)
    #     print(f"Bids for {security}:\n{sorted_bids}")
    #     print(f"Asks for {security}:\n{sorted_asks}")

    # print("My positions:", self.positions)

async def main():
    SERVER = '3.138.154.148:3333'
    my_client = MyXchangeClient(SERVER,"chicago7","^DmqJY6UUp")
    await my_client.start(None)
    return

if __name__ == "__main__":
    """....."""
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(main())