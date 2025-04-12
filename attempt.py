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
    earnings: float
    SPREAD: float

    """Large-cap stock with earnings announcements."""
    def __init__(self):
        super().__init__("APT")
        self.earnings = None
        self.SPREAD = 50
    
    def calculate_fair_price(self, earning):
        self.price = self.PE_RATIO * earning

    def update_earnings(self, earnings):
        self.earnings = earnings

    def check_arbitrage(self, order_book) -> Optional[dict[str, tuple[int, int]]]:
        if not self.earnings or not self.price:
            return None

        trades = {}

        best_bid = max(order_book.bids.keys()) if order_book.bids else None
        if best_bid and best_bid > self.price:
            trades[self.symbol] = (-1, int(self.price + self.SPREAD))

        best_ask = min(order_book.asks.keys()) if order_book.asks else None
        if best_ask and best_ask < self.price:
            trades[self.symbol] = (1, int(self.price - self.SPREAD)) 
        
        return trades if trades else None

class DLR(Asset):
    """Mid-cap stock dependent on petition signatures.
    Fair = 100 if >=100,000 signatures are collected by the end of day 10, else 0.
    Signature growth follows a lognormal process:
        S_i ~ LogNormal(log(alpha) + log(S_{i-1}), sigma^2)
    """

    current_signatures: int
    alpha: float
    sigma: float
    time_step: int
    history: list[int]
    TOTAL_TICKS: int

    def __init__(self):
        super().__init__("DLR")
        self.current_signatures = 5000  # initial value
        self.alpha = 1.0630449594499
        self.sigma = 0.006
        self.time_step = 0  # current update index
        self.history = [self.current_signatures]
        self.TOTAL_TICKS = 50

    def update_signatures(self, new_signatures: int):
        """
        Update current signature count based on news release.
        Append to history and increment time step.
        """
        self.current_signatures += new_signatures
        self.history.append(self.current_signatures)
        self.time_step += 1

    def simulate_signature_paths(self, num_ticks_left: int, num_simulations: int = 1000) -> float:
        """
        Monte Carlo simulation of signature growth to estimate probability of hitting 100,000.
        """
        final_counts = []
        
        for _ in range(num_simulations):
            count = self.current_signatures
            for _ in range(num_ticks_left):
                mu = np.log(self.alpha) + np.log(count)
                count = np.random.lognormal(mean=mu, sigma=self.sigma)
            final_counts.append(count)

        # Probability of reaching 100,000 signatures
        success_prob = np.mean(np.array(final_counts) >= 100000)

        return success_prob

    def compute_fair_value(self) -> float:
        """
        Estimate fair value as expected payout (100 or 0).
        """
        ticks_left = self.TOTAL_TICKS - self.time_step
        if ticks_left <= 0:  # End of evaluation period
            return 100.0 if self.current_signatures >= 100000 else 0.0
        
        prob = self.simulate_signature_paths(ticks_left)
        print("DLR THEO: ", prob)
        return 100 * prob
    
    def get_market_making_quotes(self, fair_value, spread=100.0):
        bid = fair_value - 0.5 * spread
        ask = fair_value + 0.5 * spread

        return bid, ask

    def check_arbitrage(self) -> Optional[dict[str, tuple[int, int]]]:
        fair = self.compute_fair_value()
        bid, ask = self.get_market_making_quotes(fair)
        if self.price > fair:
            return {self.symbol: (-1, int(bid))}
        if self.price < fair:
            return {self.symbol: (1, int(ask))}
        return None

class MKJ(Asset):
    SPREAD_MULTIPLIER: float

    """Small-cap stock with unstructured news."""
    def __init__(self):
        super().__init__("MKJ")
        self.SPREAD_MULTIPLIER = 1.5

    def get_fair(self, mkt_implied_prices):
        return mkt_implied_prices[self.symbol]
    
    def get_market_making_quotes(self, fair_price: float = None) -> tuple[float, float]:
        """
        Generate bid/ask quotes centered around fair price with spread.
        """
        if fair_price is None:
            fair_price = self.get_fair(self._mkt_implied_prices)
        
        bid = fair_price - self.SPREAD_MULTIPLIER/2
        ask = fair_price + self.SPREAD_MULTIPLIER/2
        return bid, ask

    def check_arbitrage(self, order_book) -> Optional[dict[str, tuple[int, int]]]:
        if not hasattr(self, '_mkt_implied_prices') or self.price is None:
            return None

        fair_price = self.get_fair(self._mkt_implied_prices)
        bid, ask = self.get_market_making_quotes(fair_price)

        trades = {}

        if order_book.bids:
            best_bid = max(order_book.bids.keys())
            if best_bid > fair_price:
                trades[self.symbol] = (-1, int(bid))

        if order_book.asks:
            best_ask = min(order_book.asks.keys())
            if best_ask < fair_price:
                trades[self.symbol] = (1, int(ask))

        return trades if trades else None

class ETF(Asset):
    FEE: float

    """Base class for ETFs (AKAV, AKIM)."""
    def __init__(self, symbol: str):
        super().__init__(symbol)
        self.FEE = 5

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
        if self.price > nav + self.FEE:
            return {self.symbol: -1, **{symbol: 1 for symbol in self.components}}
        elif self.price < nav - self.FEE:
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

    def __init__(self, xchange_client=None):
        self.assets = {
            "APT": APT(),
            "DLR": DLR(),
            "MKJ": MKJ(),
            "AKAV": AKAV(),
            "AKIM": AKIM()
        }
        self.open_orders = []
        self.xclient = xchange_client
        if xchange_client:
            self.sync_positions()

    def update_market_data(self, prices: Dict[str, float]) -> None:
        """Update all asset prices."""
        for symbol, price in prices.items():
            self.assets[symbol].update_price(price)

    def execute_trades(self, trades: Dict[str, int]) -> bool:
        if len(self.open_orders) + 1 > self.MAX_OPEN_ORDERS:
            print(f"[RISK] Blocked: MAX_OPEN_ORDERS ({self.MAX_OPEN_ORDERS}) reached")
            return False
        
        new_volume = 0
        for symbol in trades:
            print(trades)
            qty = trades[symbol][0]
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
    
        self.sync_positions()
        print(f"[EXECUTED] Trades: {trades}")
        return True

    def sync_positions(self) -> None:
        for symbol, position in self.xclient.positions.items():
            if symbol in self.assets:
                self.assets[symbol].position = position

    def run_arbitrage(self, symbol: str) -> Optional[Dict[str, int]]:
        """Check and execute AKAV arbitrage."""
        if symbol == "APT":
            trades = self.assets[symbol].check_arbitrage(self.xclient.order_books["APT"]) 
        elif symbol == "MKJ":
            trades = self.assets[symbol].check_arbitrage(self.xclient.order_books["MKJ"])
        elif symbol == "AKAV":
            trades = self.assets[symbol].check_arbitrage(self.assets)
        else:
            trades = self.assets[symbol].check_arbitrage()

        if trades:
            valid = self.execute_trades(trades)
            if not valid:
                print("[INFO] Arbitrage opportunity skipped due to risk limits")
        return trades

    def run(self, prices: Dict[str, float], symbol: str) -> Optional[Dict[str, int]]:
        """Main loop."""
        self.update_market_data(prices)
        trades = self.run_arbitrage(symbol)
        print(f"POSITIONS: {[f'{a.symbol}: {a.position}' for a in self.assets.values()]}")
        return trades

class MyXchangeClient(xchange_client.XChangeClient):
    '''A shell client with the methods that can be implemented to interact with the xchange.'''

    def __init__(self, server: str, username: str, password: str):
        super().__init__(server, username, password)
        # Initialize TradingBot without modifying existing attributes
        self._trading_bot = TradingBot(self)
        self._last_prices = {}
        self._mkt_implied_prices ={}
    
    # Original interface methods remain exactly the same
    async def bot_handle_cancel_response(self, order_id: str, success: bool, error: Optional[str]) -> None:
        order = self.open_orders[order_id]
        print(f"{'Market' if order[2] else 'Limit'} Order ID {order_id} cancelled, {order[1]} unfilled")

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        # Update TradingBot positions while maintaining original print
        if order_id in self.open_orders:
            side = self.open_orders[order_id][0].side
            signed_qty = qty if side == xchange_client.Side.BUY else -qty
            self._trading_bot.execute_trades({self.open_orders[order_id][0].symbol: (signed_qty, price)})
        print("order fill", self.positions)

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        print("order rejected because of ", reason)

    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int):
        # Update last prices without modifying behavior
        self._last_prices[symbol] = price

    async def bot_handle_book_update(self, symbol: str) -> None:
        # Enhanced book handling while maintaining original empty implementation
        book = self.order_books[symbol]
        if book.bids and book.asks:
            vol_filter = 0
            best_bid = max(book.bids.keys())
            best_ask = min(book.asks.keys())

            if len([price for price in book.asks.keys() if abs(book.asks[price]) > vol_filter]) > 0 and len([price for price in book.bids.keys() if abs(book.bids[price]) > vol_filter]) > 0:
                best_ask = min([price for price in book.asks.keys() if abs(book.asks[price]) > vol_filter])
                best_bid = max([price for price in book.bids.keys() if abs(book.bids[price]) > vol_filter])

            best_bid_vol = book.bids[best_bid]
            best_ask_vol = book.asks[best_ask]

            self._last_prices[symbol] = (best_bid + best_ask) / 2
            self._mkt_implied_prices[symbol] = self._last_prices[symbol] 
            if best_bid_vol + best_ask_vol > 0:
                self._mkt_implied_prices[symbol] = (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)
            self._trading_bot.update_market_data(self._last_prices)

    async def bot_handle_swap_response(self, swap: str, qty: int, success: bool):
        pass 

    async def bot_handle_news(self, news_release: dict):
        print("NEWS: ", news_release)
        news_data = news_release["new_data"]
        if news_release["kind"] == "structured":
            if news_data["asset"] == "DLR":
                self._trading_bot.assets["DLR"].update_signatures(news_data["new_signatures"])
            elif news_data["asset"] == "APT":
                for order_id, order in list(self.open_orders.items()):
                    if order[0].symbol == "APT":
                        await self.cancel_order(order_id)
                        print(f"[CANCELLED] APT Order ID: {order_id}")

                self._trading_bot.assets["APT"].update_earnings(news_data["value"])
                trades = self._trading_bot.assets["APT"].check_arbitrage(self.order_books["APT"])

                if trades:
                    for symbol in trades:
                        qty = trades[symbol][0]
                        side = xchange_client.Side.BUY if qty > 0 else xchange_client.Side.SELL
                        await self.place_order(symbol, abs(qty), side, trades[symbol][1])

    async def view_books(self):
        while True:
            await asyncio.sleep(3)
            for security, book in self.order_books.items():
                sorted_bids = sorted((k,v) for k,v in book.bids.items() if v != 0)
                sorted_asks = sorted((k,v) for k,v in book.asks.items() if v != 0)
                print(f"Bids for {security}:\n{sorted_bids}")
                print(f"Asks for {security}:\n{sorted_asks}")

    async def start(self, user_interface):
        asyncio.create_task(self._enhanced_trade())  # Modified to use new trading method
        if user_interface:
            self.launch_user_interface()
            asyncio.create_task(self.handle_queued_messages())
        await self.connect()

    async def _enhanced_trade(self):
        """Enhanced trading logic that incorporates TradingBot while maintaining original behavior"""
        # Original trading sequence preserved
        await asyncio.sleep(2)
        print("attempting to trade")

        if self.open_orders:
            await self.cancel_order(list(self.open_orders.keys())[0])

        print("SYSTEM POSITIONS: ", self.positions)

        # New TradingBot integration
        while True:
            try:
                if self._last_prices:
                    # Run trading bot logic
                    for asset in self._trading_bot.assets:
                        arb_trades = self._trading_bot.run(self._last_prices, asset)
                        
                        if arb_trades:
                            for symbol in arb_trades:
                                qty = arb_trades[symbol][0]
                                side = (xchange_client.Side.BUY if qty > 0 
                                    else xchange_client.Side.SELL)
                                await self.place_order(
                                    symbol, 
                                    abs(qty), 
                                    side,
                                    arb_trades[symbol][1]
                                )
                
                        await asyncio.sleep(1)  # Throttle trading frequency
                
            except Exception as e:
                print(f"Error in trading loop: {e}")
                await asyncio.sleep(5)

    # for security, book in self.order_books.items():
    #     sorted_bids = sorted((k,v) for k,v in book.bids.items() if v != 0)
    #     sorted_asks = sorted((k,v) for k,v in book.asks.items() if v != 0)
    #     print(f"Bids for {security}:\n{sorted_bids}")
    #     print(f"Asks for {security}:\n{sorted_asks}")

    # print("My positions:", self.positions)

async def main():
    my_client = MyXchangeClient('server.uchicagotradingcompetition25.com:3333',"chicago7","^DmqJY6UUp")
    await my_client.start(None)
    return

if __name__ == "__main__":
    """....."""
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(main())