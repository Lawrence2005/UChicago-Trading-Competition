from typing import Dict, Optional
import numpy as np
import abc
from utcxchangelib import xchange_client
import asyncio

class Asset:
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

    def __init__(self):
        super().__init__("APT")

class DLR(Asset):
    def __init__(self):
        super().__init__("DLR")

class MKJ(Asset):
    def __init__(self):
        super().__init__("MKJ")

class ETF(Asset):
    FEE: float

    def __init__(self, symbol: str):
        super().__init__(symbol)
        self.FEE = 5

class AKAV(ETF):
    components: list[str]

    def __init__(self):
        super().__init__("AKAV")
        self.components = ["APT", "DLR", "MKJ"]

    def calculate_nav(self, assets: Dict[str, Asset]) -> float:
        return sum(assets[symbol].price for symbol in self.components)

    def check_arbitrage(self, assets: Dict[str, Asset]) -> Optional[Dict[str, int]]:
        nav = self.calculate_nav(assets)
        if self.price > nav + self.FEE:
            return {self.symbol: -1, **{symbol: 1 for symbol in self.components}}
        elif self.price < nav - self.FEE:
            return {self.symbol: 1, **{symbol: -1 for symbol in self.components}}
        return None

class AKIM(ETF):
    def __init__(self):
        super().__init__("AKIM")

class TradingBot:
    MAX_ORDER_SIZE = 40
    MAX_OPEN_ORDERS = 50
    MAX_OUTSTANDING_VOLUME = 120
    MAX_ABSOLUTE_POSITION = 200

    def __init__(self):
        self.assets = {
            "APT": Asset("APT"),
            "DLR": Asset("DLR"),
            "MKJ": Asset("MKJ"),
            "AKAV": AKAV(),
            "AKIM": AKIM()
        }
        self.open_orders: dict[str, tuple[str, xchange_client.Side, int]] = {}

    def update_market_data(self, prices: Dict[str, float]) -> None:
        for symbol, price in prices.items():
            self.assets[symbol].update_price(price)

    def execute_trades(self, trades: Dict[str, int]) -> bool:
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

        current_volume = sum(abs(qty) for _, (_, _, qty) in self.open_orders.items())
        if current_volume + new_volume > self.MAX_OUTSTANDING_VOLUME:
            print(f"[RISK] Blocked: MAX_OUTSTANDING_VOLUME ({self.MAX_OUTSTANDING_VOLUME}) exceeded")
            return False

        for symbol, amt in trades.items():
            self.assets[symbol].position += amt
        print(f"[EXECUTED] Trades: {trades}")
        return True

    def run_arbitrage(self) -> None:
        trades = self.assets["AKAV"].check_arbitrage(self.assets)
        if trades:
            valid = self.execute_trades(trades)
            if not valid:
                print("[INFO] Arbitrage opportunity skipped due to risk limits")

    def run(self, prices: Dict[str, float]) -> None:
        self.update_market_data(prices)
        self.run_arbitrage()
        print(f"Positions: {[f'{a.symbol}: {a.position}' for a in self.assets.values()]}")

class MyXchangeClient(xchange_client.XChangeClient):
    def __init__(self, server: str, username: str, password: str):
        super().__init__(server, username, password)
        self._trading_bot = TradingBot()
        self._last_prices = {}
        self._mkt_implied_prices = {}

    async def bot_handle_cancel_response(self, order_id: str, success: bool, error: Optional[str]) -> None:
        order = self._trading_bot.open_orders.get(order_id)
        if order:
            print(f"Order ID {order_id} cancelled, {order[2]} unfilled")

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        if order_id in self._trading_bot.open_orders:
            symbol, side, _ = self._trading_bot.open_orders[order_id]
            signed_qty = qty if side == xchange_client.Side.BUY else -qty
            self._trading_bot.execute_trades({symbol: signed_qty})
            print(f"[FILL] {qty} {symbol} @ {price} on side {side}")

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        print("order rejected because of ", reason)

    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int):
        self._last_prices[symbol] = price

    async def bot_handle_book_update(self, symbol: str) -> None:
        book = self.order_books[symbol]
        if book.bids and book.asks:
            vol_filter = 0
            best_bid = max(book.bids.keys())
            best_ask = min(book.asks.keys())

            filtered_bids = [price for price in book.bids.keys() if abs(book.bids[price]) > vol_filter]
            filtered_asks = [price for price in book.asks.keys() if abs(book.asks[price]) > vol_filter]

            if filtered_asks and filtered_bids:
                best_ask = min(filtered_asks)
                best_bid = max(filtered_bids)

            best_bid_vol = book.bids[best_bid]
            best_ask_vol = book.asks[best_ask]

            self._last_prices[symbol] = (best_bid + best_ask) / 2
            self._mkt_implied_prices[symbol] = self._last_prices[symbol]
            if best_bid_vol + best_ask_vol > 0:
                self._mkt_implied_prices[symbol] = (
                    best_bid * best_ask_vol + best_ask * best_bid_vol
                ) / (best_bid_vol + best_ask_vol)

            self._trading_bot.update_market_data(self._last_prices)

    async def bot_handle_swap_response(self, swap: str, qty: int, success: bool):
        pass

    async def bot_handle_news(self, news_release: dict):
        timestamp = news_release["timestamp"]
        news_type = news_release['kind']
        news_data = news_release["new_data"]
        if news_type == "structured":
            subtype = news_data["structured_subtype"]
            symb = news_data["asset"]
            if subtype == "earnings":
                earnings = news_data["value"]
            else:
                new_signatures = news_data["new_signatures"]
                cumulative = news_data["cumulative"]

    async def start(self, user_interface):
        asyncio.create_task(self._enhanced_trade())
        if user_interface:
            self.launch_user_interface()
            asyncio.create_task(self.handle_queued_messages())
        await self.connect()

    async def _enhanced_trade(self):
        await asyncio.sleep(5)
        print("attempting to trade")
        order_id = await self.place_order("APT", 3, xchange_client.Side.BUY, 5)
        self._trading_bot.open_orders[order_id] = ("APT", xchange_client.Side.BUY, 3)

        order_id = await self.place_order("APT", 3, xchange_client.Side.SELL, 7)
        self._trading_bot.open_orders[order_id] = ("APT", xchange_client.Side.SELL, 3)

        await asyncio.sleep(5)
        if self._trading_bot.open_orders:
            await self.cancel_order(next(iter(self._trading_bot.open_orders)))

        await self.place_swap_order('toAKAV', 1)
        await asyncio.sleep(5)
        await self.place_swap_order('fromAKAV', 1)
        await asyncio.sleep(5)

        order_id = await self.place_order("APT", 1000, xchange_client.Side.SELL, 7)
        self._trading_bot.open_orders[order_id] = ("APT", xchange_client.Side.SELL, 1000)
        await asyncio.sleep(5)

        market_order_id = await self.place_order("APT", 10, xchange_client.Side.SELL)
        self._trading_bot.open_orders[market_order_id] = ("APT", xchange_client.Side.SELL, 10)
        print("MARKET ORDER ID:", market_order_id)
        await asyncio.sleep(5)
        print("my positions:", self.positions)

        while True:
            try:
                if self._last_prices:
                    self._trading_bot.run(self._last_prices)
                    arb_trades = self._trading_bot.assets["AKAV"].check_arbitrage(
                        self._trading_bot.assets
                    )
                    if arb_trades:
                        for symbol, qty in arb_trades.items():
                            side = xchange_client.Side.BUY if qty > 0 else xchange_client.Side.SELL
                            order_id = await self.place_order(symbol, abs(qty), side)
                            self._trading_bot.open_orders[order_id] = (symbol, side, abs(qty))
                await asyncio.sleep(1)
            except Exception as e:
                print(f"Error in trading loop: {e}")
                await asyncio.sleep(5)

async def main():
    my_client = MyXchangeClient('3.138.154.148:3333',"chicago7","^DmqJY6UUp")
    await my_client.start(None)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(main())
