from typing import Dict, Optional
import numpy as np
import abc
from utcxchangelib import xchange_client
import argparse
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
    """Enhanced trading bot with position tracking and risk management."""

    def __init__(self):
        self.assets = {
            "APT": Asset("APT"),
            "DLR": Asset("DLR"),
            "MKJ": Asset("MKJ"),
            "AKAV": AKAV(),
            "AKIM": AKIM()
        }
        self.open_orders = []
        self.position_updates = {}  # Track pending position changes
        
        # Risk parameters
        self.MAX_ORDER_SIZE = 40
        self.MAX_OPEN_ORDERS = 50
        self.MAX_OUTSTANDING_VOLUME = 120
        self.MAX_ABSOLUTE_POSITION = 200
        
        # Trading parameters
        self.min_spread = 0.01  # Minimum spread to trade
        self.max_trade_size = 10  # Maximum size per trade
        
    def update_market_data(self, prices: Dict[str, float]) -> None:
        """Update all asset prices and calculate indicators."""
        for symbol, price in prices.items():
            if symbol in self.assets:
                self.assets[symbol].update_price(price)
                
                # Calculate technical indicators
                self.assets[symbol].update_indicators(price)
    
    def check_risk_limits(self, trades: Dict[str, int]) -> bool:
        """Check proposed trades against risk limits."""
        if len(self.open_orders) + len(trades) > self.MAX_OPEN_ORDERS:
            print(f"[RISK] Blocked: MAX_OPEN_ORDERS ({self.MAX_OPEN_ORDERS}) reached")
            return False
        
        new_volume = sum(abs(qty) for qty in trades.values())
        if new_volume > self.MAX_OUTSTANDING_VOLUME:
            print(f"[RISK] Blocked: MAX_OUTSTANDING_VOLUME ({self.MAX_OUTSTANDING_VOLUME}) exceeded")
            return False
            
        for symbol, qty in trades.items():
            if abs(qty) > self.MAX_ORDER_SIZE:
                print(f"[RISK] Blocked: Order for {symbol} exceeds MAX_ORDER_SIZE ({self.MAX_ORDER_SIZE})")
                return False
                
            new_position = self.assets[symbol].position + qty
            if abs(new_position) > self.MAX_ABSOLUTE_POSITION:
                print(f"[RISK] Blocked: Position for {symbol} would exceed MAX_ABSOLUTE_POSITION ({self.MAX_ABSOLUTE_POSITION})")
                return False
                
        return True
    
    def run_arbitrage(self) -> Dict[str, int]:
        """Run arbitrage strategies and return proposed trades."""
        trades = {}
        
        # AKAV arbitrage
        akav_trades = self.assets["AKAV"].check_arbitrage(self.assets)
        if akav_trades and self.check_risk_limits(akav_trades):
            trades.update(akav_trades)
        
        # Add other strategies here...
        
        return trades
    
    def run_market_making(self, order_book: Dict[str, Any]) -> Dict[str, int]:
        """Run market making strategies."""
        trades = {}
        
        for symbol, asset in self.assets.items():
            if symbol not in order_book:
                continue
                
            book = order_book[symbol]
            if not book.bids or not book.asks:
                continue
                
            best_bid = max(book.bids.keys())
            best_ask = min(book.asks.keys())
            spread = best_ask - best_bid
            
            if spread < self.min_spread:
                continue  # Spread too tight to market make
                
            # Simple market making logic - adjust as needed
            mid_price = (best_bid + best_ask) / 2
            our_bid = mid_price - spread/2
            our_ask = mid_price + spread/2
            
            # Calculate position-adjusted sizes
            position = self.assets[symbol].position
            max_buy = min(self.max_trade_size, self.MAX_ABSOLUTE_POSITION - position)
            max_sell = min(self.max_trade_size, self.MAX_ABSOLUTE_POSITION + position)
            
            if max_buy > 0:
                trades[symbol] = max_buy
            elif max_sell > 0:
                trades[symbol] = -max_sell
                
        return trades
    
class MyXchangeClient(xchange_client.XChangeClient):
    def __init__(self, host: str, username: str, password: str):
        super().__init__(host, username, password)
        self.trading_bot = TradingBot()  # Initialize your trading bot
        self.last_book_update = {}  # Track last book updates
    
    async def bot_handle_book_update(self, symbol: str) -> None:
        """Handle order book updates and trigger trading logic"""
        # Get current order book
        book = self.order_books.get(symbol)
        if not book:
            return
            
        # Convert book to prices dictionary format expected by TradingBot
        prices = {
            symbol: (min(book.asks.keys()) + max(book.bids.keys())) / 2  # Mid price
        }
        
        # Update trading bot with latest prices
        self.trading_bot.update_market_data(prices)
        
        # Store book update time
        self.last_book_update[symbol] = time.time()
        
        # Run trading logic
        await self.execute_bot_strategy(symbol)
    
    async def execute_bot_strategy(self, symbol: str):
        """Execute the trading bot's strategy for a given symbol"""
        # Get current positions from exchange
        positions = {sym: qty for sym, qty in self.positions.items() if sym in self.trading_bot.assets}
        
        # Update bot positions to match exchange
        for sym, qty in positions.items():
            self.trading_bot.assets[sym].position = qty
        
        # Run arbitrage checks
        self.trading_bot.run_arbitrage()
        
        # Get target positions from bot
        target_positions = {a.symbol: a.position for a in self.trading_bot.assets.values()}
        
        # Calculate needed trades
        trades = {}
        for sym, target_qty in target_positions.items():
            current_qty = positions.get(sym, 0)
            delta = target_qty - current_qty
            if delta != 0:
                trades[sym] = delta
        
        # Execute trades through exchange
        await self.execute_trades(trades)
    
    async def execute_trades(self, trades: Dict[str, int]):
        """Execute trades through the exchange"""
        for symbol, qty in trades.items():
            if qty == 0:
                continue
                
            side = xchange_client.Side.BUY if qty > 0 else xchange_client.Side.SELL
            qty = abs(qty)
            
            # Get current market price
            book = self.order_books.get(symbol)
            if not book:
                continue
                
            # Place limit order at best bid/ask
            if side == xchange_client.Side.BUY:
                price = min(book.asks.keys())  # Pay the ask
            else:
                price = max(book.bids.keys())  # Get the bid
                
            # Place order
            await self.place_order(symbol, qty, side, price)
    
    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        """Handle order fills and update bot positions"""
        print(f"Order {order_id} filled: {qty} @ {price}")
        
        # Get order details
        order = self.open_orders.get(order_id)
        if not order:
            return
            
        symbol, quantity, side, price = order
        
        # Update bot position
        if symbol in self.trading_bot.assets:
            if side == xchange_client.Side.BUY:
                self.trading_bot.assets[symbol].position += quantity
            else:
                self.trading_bot.assets[symbol].position -= quantity
    
    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        """Handle order rejections"""
        print(f"Order {order_id} rejected: {reason}")
        
        # Optionally retry or adjust strategy
        order = self.open_orders.get(order_id)
        if order:
            symbol, quantity, side, price = order
            print(f"Retrying order for {symbol}...")
            await self.place_order(symbol, quantity, side, price * 0.99 if side == xchange_client.Side.BUY else price * 1.01)
    
    async def trade(self):
        """Main trading loop"""
        # Initial pause to receive market data
        await asyncio.sleep(5)
        
        while True:
            try:
                # Check all symbols periodically
                for symbol in self.trading_bot.assets.keys():
                    if symbol in self.order_books:
                        await self.execute_bot_strategy(symbol)
                
                # Small delay to prevent excessive trading
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"Error in trading loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying

# class MyXchangeClient(xchange_client.XChangeClient):
#     '''A shell client with the methods that can be implemented to interact with the xchange.'''

#     def __init__(self, host: str, username: str, password: str):
#         super().__init__(host, username, password)
    
#     async def bot_handle_cancel_response(self, order_id: str, success: bool, error: Optional[str]) -> None:
#         order = self.open_orders[order_id]
#         print(f"{'Market' if order[2] else 'Limit'} Order ID {order_id} cancelled, {order[1]} unfilled")

#     async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
#         print("order fill", self.positions)

#     async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
#         print("order rejected because of ", reason)

#     async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int):
#         pass

#     async def bot_handle_book_update(self, symbol: str) -> None:
#         pass

#     async def bot_handle_swap_response(self, swap: str, qty: int, success: bool):
#         pass

#     async def bot_handle_news(self, news_release: dict):
#         # Parsing the message based on what type was received
#         timestamp = news_release["timestamp"] # This is in exchange ticks not ISO or Epoch
#         news_type = news_release['kind']
#         news_data = news_release["new_data"]

#         if news_type == "structured":
#             subtype = news_data["structured_subtype"]
#             symb = news_data["asset"]
#             if subtype == "earnings":
#                 earnings = news_data["value"]
#                 ### Do something with this data ###
#             else:
#                 new_signatures = news_data["new_signatures"]
#                 cumulative = news_data["cumulative"]
#                 ### Do something with this data ###
#         else:
#             ### Not sure what you would do with unstructured data.... ###
#             pass

#     async def view_books(self):
#         while True:
#             await asyncio.sleep(3)
#             for security, book in self.order_books.items():
#                 sorted_bids = sorted((k,v) for k,v in book.bids.items() if v != 0)
#                 sorted_asks = sorted((k,v) for k,v in book.asks.items() if v != 0)
#                 print(f"Bids for {security}:\n{sorted_bids}")
#                 print(f"Asks for {security}:\n{sorted_asks}")

#     async def start(self, user_interface):
#         asyncio.create_task(self.trade())

#         # This is where Phoenixhood will be launched if desired. There is no need to change these lines, you can either remove the if or delete the whole thing depending on your purposes.
#         if user_interface:
#             self.launch_user_interface()
#             asyncio.create_task(self.handle_queued_messages())

#         await self.connect()

#     async def trade(self):
#         """    
#         Examples of various XChangeClient actions (limit orders, order cancel, swaps, and market orders)
#         """
#         # Pause for 5 seconds before starting the trading sequence
#         await asyncio.sleep(5)
#         print("attempting to trade")

#         # Place a BUY limit order for 3 units of APT at price 5
#         await self.place_order("APT", 3, xchange_client.Side.BUY, 5)

#         # Place a SELL limit order for 3 units of APT at price 7
#         await self.place_order("APT", 3, xchange_client.Side.SELL, 7)

#         # Pause for 5 seconds to allow orders to be processed
#         await asyncio.sleep(5)

#         # Cancel the first open order by retrieving its ID from open_orders
#         if self.open_orders:
#             await self.cancel_order(list(self.open_orders.keys())[0])

#         # Place a swap order to swap 1 unit to AKAV
#         await self.place_swap_order('toAKAV', 1)

#         # Pause for 5 seconds to allow swap to process
#         await asyncio.sleep(5)

#         # Place a swap order to swap 1 unit from AKAV
#         await self.place_swap_order('fromAKAV', 1)

#         # Pause for 5 seconds after the swap
#         await asyncio.sleep(5)

#         # Place a large SELL limit order of 1000 units of APT at price 7
#         await self.place_order("APT", 1000, xchange_client.Side.SELL, 7)

#         # Pause for 5 seconds before placing a market order
#         await asyncio.sleep(5)

#         # Place a SELL market order for 10 units of APT
#         # Market orders do not have a price, so the price parameter is omitted
#         market_order_id = await self.place_order("APT", 10, xchange_client.Side.SELL)

#         # Print the ID of the market order for reference
#         print("MARKET ORDER ID:", market_order_id)

#         # Pause for 5 seconds to allow order to settle
#         await asyncio.sleep(5)

#         # Print the current positions held after the sequence of trades
#         print("my positions:", self.positions)

#     # for security, book in self.order_books.items():
#     #     sorted_bids = sorted((k,v) for k,v in book.bids.items() if v != 0)
#     #     sorted_asks = sorted((k,v) for k,v in book.asks.items() if v != 0)
#     #     print(f"Bids for {security}:\n{sorted_bids}")
#     #     print(f"Asks for {security}:\n{sorted_asks}")

#     # print("My positions:", self.positions)

async def main():
    SERVER = '3.138.154.148:3333'
    my_client = MyXchangeClient(SERVER,"chicago7","^DmqJY6UUp")
    await my_client.start(None)
    return

if __name__ == "__main__":
    """....."""
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(main())