import math
from datetime import datetime, date

class StockMathModel:
    def __init__(self, start_date=None):
        """
        מודל אחיד לכל החישובים המתמטיים בפרויקט
        """
        self.start_date = start_date or date.today()
        self.reset_participants()

        # זמן/תאריך ראשוני
        self.update_time(datetime.now())
    # ===============================
    # TIME & DATE METHODS
    # ===============================
    def update_time(self, dt=None):
        self.dt = dt or datetime.now()

        self.year   = self.dt.year
        self.month  = self.dt.month
        self.day    = self.dt.day
        self.hour   = self.dt.hour
        self.minute = self.dt.minute
        self.second = self.dt.second

        self.weekday     = self.dt.weekday()
        self.day_of_year = self.dt.timetuple().tm_yday
        self.quarter     = (self.month - 1) // 3 + 1

        self.is_fibo  = self._is_fibonacci(self.day)
        self.is_prime = self._is_prime(self.day)

    def day_index(self):
        return (date.today() - self.start_date).days

    @staticmethod
    def _is_fibonacci(n):
        def is_square(x):
            return int(math.sqrt(x))**2 == x
        return is_square(5*n*n + 4) or is_square(5*n*n - 4)

    @staticmethod
    def _is_prime(n):
        if n < 2: return False
        for i in range(2, int(math.sqrt(n))+1):
            if n % i == 0:
                return False
        return True

    # ===============================
    # PARTICIPANT INPUT METHODS
    # ===============================

    def reset_participants(self):
        self.buyers  = []
        self.sellers = []
        self.total_buy_percent  = 0
        self.total_sell_percent = 0

    def add_buyer(self, percent, price):
        if percent > 100 or self.total_buy_percent + percent > 100:
            raise ValueError("Invalid buyer percent")
        self.buyers.append((percent, price))
        self.total_buy_percent += percent

    def add_seller(self, percent, price):
        if percent > 100 or self.total_sell_percent + percent > 100:
            raise ValueError("Invalid seller percent")
        self.sellers.append((percent, price))
        self.total_sell_percent += percent

    def adjust_participants(self, data, factor):
        return [(p, pr * factor) for p, pr in data]

    @staticmethod
    def weighted_average(data):
        num = sum(p * pr for p, pr in data)
        den = sum(p for p, _ in data)
        return num / den if den != 0 else 0

    # ===============================
    # MARKET CALCULATIONS
    # ===============================

    def compute_market_factor(self, t=None, A=0.12, T=100, eps=0.04):
        t = t if t is not None else self.day_index()
        return 1 + A*math.sin((2*math.pi*t)/T) + eps*math.sin(t)*math.sin(math.sqrt(2)*t)

    def compute_daily_prices(self):
        factor = self.compute_market_factor()

        adj_buy  = self.adjust_participants(self.buyers,  factor)
        adj_sell = self.adjust_participants(self.sellers, factor)

        avg_buy  = self.weighted_average(adj_buy)
        avg_sell = self.weighted_average(adj_sell)

        close_price = (avg_buy + avg_sell)/2

        return {
            "factor"     : factor,
            "avg_buy"    : avg_buy,
            "avg_sell"   : avg_sell,
            "close_price": close_price
        }

    # ===============================
    # PERCENT CHANGE CALCULATOR
    # ===============================

    @staticmethod
    def percent_change(old, new):
        if new == 0:
            raise ValueError("New price cannot be zero")
        return ((new - old)/old) * 100

    # ===============================
    # PROBABILISTIC FUNCTIONS
    # ===============================

    @staticmethod
    def factorial(n):
        return 1 if n == 0 else n * StockMathModel.factorial(n-1)

    @classmethod
    def Ppoason(cls, g, k):
        return (g**k * math.exp(-g)) / cls.factorial(k)

    @classmethod
    def Pbinomi(cls, n, k, p):
        return cls.factorial(n)/(cls.factorial(k)*cls.factorial(n-k)) * p**k * (1-p)**(n-k)

    @staticmethod
    def Pnormal(x, mu, sigma):
        return (1/math.sqrt(2*math.pi*sigma*sigma)) * math.exp(-((x - mu)**2)/(2*sigma*sigma))

    # ===============================
    # DEBUG PRINT
    # ===============================

    def summary(self):
        res = self.compute_daily_prices()
        return (
            f"📊 יום #{self.day_index()} | "
            f"פקטור שוק: {res['factor']:.4f} | "
            f"ממוצע קנייה: {res['avg_buy']:.4f} | "
            f"ממוצע מכירה: {res['avg_sell']:.4f} | "
            f"מחיר סגירה: {res['close_price']:.4f}"
        )
