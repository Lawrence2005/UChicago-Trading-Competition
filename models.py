from typing import Dict, Optional
import numpy as np
import abc

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