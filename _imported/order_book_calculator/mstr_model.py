"""
mstr_model.py  --  merged, hardened version of MSTR.py + MSTR1.py
================================================================
MSTR.py and MSTR1.py were two drafts of the SAME toy:

    * MSTR.py   -- procedural script: interactive buyer/seller input, a
                   deterministic "market factor" sine wave, a weighted-average
                   "close price", a percent-change / direction calculator and a
                   few standalone probability functions.
    * MSTR1.py  -- an OOP refactor of the market part into ``StockMathModel``;
                   cleaner, but it dropped the CLI and the direction calculator
                   and it kept the same numeric bugs.

This file merges them:  ``StockMathModel`` (pure, testable logic, from MSTR1)
+ a validated interactive CLI (``run_cli``, from MSTR.py) + numerically stable
probability helpers.  Every bug found in the two originals is fixed -- see
BUGS-FIXED at the bottom.

NOTE ON SCOPE:  the "market factor" is a deterministic sine wave with invented
constants and the "close price" is just the midpoint of quotes you type in.
This is a calculator / teaching tool, NOT a forecasting model, and it is not
connected to the CARN-X pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List, Optional, Sequence, Tuple

Number = float


# ===========================================================================
#  Pure numeric helpers  (no I/O, individually testable)
# ===========================================================================

def is_fibonacci(n: int) -> bool:
    """True iff ``n`` is a Fibonacci number. Uses integer sqrt -- exact for any n
    (the originals used ``int(math.sqrt(x))`` which loses precision for large x)."""
    if n < 0:
        return False

    def _is_square(x: int) -> bool:
        r = math.isqrt(x)
        return r * r == x

    return _is_square(5 * n * n + 4) or _is_square(5 * n * n - 4)


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    for i in range(3, math.isqrt(n) + 1, 2):
        if n % i == 0:
            return False
    return True


def factorial(n: int) -> int:
    """Iterative factorial -- no RecursionError, rejects negatives / non-ints
    (the originals recursed and would overflow the stack or loop forever)."""
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(f"factorial expects a non-negative int, got {n!r}")
    if n < 0:
        raise ValueError("factorial is undefined for negative numbers")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def _log_factorial(n: int) -> float:
    """log(n!) via lgamma -- stays finite where factorial(n) would overflow."""
    if n < 0:
        raise ValueError("log-factorial is undefined for negative numbers")
    return math.lgamma(n + 1)


def poisson_pmf(lam: Number, k: int) -> float:
    """P(X = k) for X ~ Poisson(lam).  Computed in log-space for stability."""
    if lam < 0:
        raise ValueError("lambda must be >= 0")
    if k < 0:
        return 0.0
    if lam == 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(k * math.log(lam) - lam - _log_factorial(k))


def binomial_pmf(n: int, k: int, p: Number) -> float:
    """P(X = k) for X ~ Binomial(n, p).  Log-space; validates its inputs."""
    if n < 0:
        raise ValueError("n must be >= 0")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    if k < 0 or k > n:
        return 0.0
    log_coeff = _log_factorial(n) - _log_factorial(k) - _log_factorial(n - k)
    log_p = (k * math.log(p)) if p > 0 else (0.0 if k == 0 else -math.inf)
    log_q = ((n - k) * math.log1p(-p)) if p < 1 else (0.0 if k == n else -math.inf)
    return math.exp(log_coeff + log_p + log_q)


def normal_pdf(x: Number, mu: Number, sigma: Number) -> float:
    """Gaussian probability density at ``x`` (was ``Pnormal``)."""
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


def percent_change(old: Number, new: Number) -> float:
    """Signed percent change from ``old`` to ``new``:  (new - old) / old * 100.

    MSTR.py had the sign inverted ( (old-new)/new ) so a price *rise* printed a
    negative number.  MSTR1.py guarded ``new == 0`` -- the wrong operand; the
    division is by ``old``.
    """
    if old == 0:
        raise ValueError("percent change is undefined when the old price is 0")
    return (new - old) / old * 100.0


def change_ratio(old: Number, new: Number) -> Tuple[float, float, float]:
    """Returns (pct_of_old, pct_of_new, ratio).

    pct_of_old = (new-old)/old*100   -- the standard percent change
    pct_of_new = (new-old)/new*100   -- the same move expressed against the new price
    ratio      = pct_of_old / pct_of_new  (== new/old);  undefined if new == 0
    """
    if old == 0 or new == 0:
        raise ValueError("change_ratio needs both prices non-zero")
    pct_of_old = (new - old) / old * 100.0
    pct_of_new = (new - old) / new * 100.0
    ratio = math.nan if pct_of_new == 0 else pct_of_old / pct_of_new
    return pct_of_old, pct_of_new, ratio


# ===========================================================================
#  Order book
# ===========================================================================

@dataclass(frozen=True)
class Quote:
    percent: float          # share of the side, 0 < percent <= 100
    price: float            # quoted price, > 0

    def __post_init__(self) -> None:
        if not (0.0 < self.percent <= 100.0):
            raise ValueError(f"percent must be in (0, 100], got {self.percent}")
        if not (self.price > 0.0):
            raise ValueError(f"price must be > 0, got {self.price}")


class OrderBook:
    """A one-day book of buy and sell quotes, each side summing to <= 100%."""

    def __init__(self) -> None:
        self.buyers: List[Quote] = []
        self.sellers: List[Quote] = []

    def _add(self, side: List[Quote], percent: float, price: float) -> None:
        quote = Quote(percent, price)
        running = sum(q.percent for q in side) + quote.percent
        if running > 100.0 + 1e-9:
            raise ValueError(f"side percent would exceed 100 (would be {running:.2f})")
        side.append(quote)

    def add_buyer(self, percent: float, price: float) -> None:
        self._add(self.buyers, percent, price)

    def add_seller(self, percent: float, price: float) -> None:
        self._add(self.sellers, percent, price)

    def reset(self) -> None:
        self.buyers.clear()
        self.sellers.clear()

    @staticmethod
    def weighted_average(quotes: Sequence[Quote]) -> Optional[float]:
        """Percent-weighted mean price, or ``None`` for an empty side
        (the originals raised ZeroDivisionError or silently returned 0)."""
        den = sum(q.percent for q in quotes)
        if den == 0:
            return None
        return sum(q.percent * q.price for q in quotes) / den


# ===========================================================================
#  The model
# ===========================================================================

@dataclass
class DailyPrices:
    day_index: int
    market_factor: float
    avg_buy: Optional[float]
    avg_sell: Optional[float]
    close_price: Optional[float]


class StockMathModel:
    """Unified home for every calculation the two drafts contained."""

    def __init__(self, start_date: Optional[date] = None) -> None:
        self.start_date: date = start_date or date.today()
        self.book = OrderBook()
        self.update_time(datetime.now())

    # ---- time -------------------------------------------------------------
    def update_time(self, dt: Optional[datetime] = None) -> None:
        self.dt = dt or datetime.now()
        self.year, self.month, self.day = self.dt.year, self.dt.month, self.dt.day
        self.hour, self.minute, self.second = self.dt.hour, self.dt.minute, self.dt.second
        self.weekday = self.dt.weekday()
        self.day_of_year = self.dt.timetuple().tm_yday
        self.quarter = (self.month - 1) // 3 + 1
        self.day_is_fibonacci = is_fibonacci(self.day)
        self.day_is_prime = is_prime(self.day)

    def day_index(self) -> int:
        return (date.today() - self.start_date).days

    # ---- market ---------------------------------------------------------
    @staticmethod
    def market_factor(t: int, A: float = 0.12, T: float = 100.0, eps: float = 0.04) -> float:
        """Deterministic bubble/crisis oscillator:  > 1 bubble, < 1 crisis.

        This is an *invented* wave (arbitrary A, T, eps) -- it has no predictive
        basis and is kept only for continuity with the originals.
        """
        if T == 0:
            raise ValueError("period T must be non-zero")
        return (
            1.0
            + A * math.sin((2.0 * math.pi * t) / T)
            + eps * math.sin(t) * math.sin(math.sqrt(2.0) * t)
        )

    def compute_daily_prices(self, t: Optional[int] = None) -> DailyPrices:
        t = self.day_index() if t is None else t
        factor = self.market_factor(t)

        def scale(quotes: Sequence[Quote]) -> List[Quote]:
            return [Quote(q.percent, q.price * factor) for q in quotes]

        avg_buy = OrderBook.weighted_average(scale(self.book.buyers))
        avg_sell = OrderBook.weighted_average(scale(self.book.sellers))

        if avg_buy is not None and avg_sell is not None:
            close = (avg_buy + avg_sell) / 2.0
        else:                       # only one side quoted -> use whichever exists
            close = avg_buy if avg_buy is not None else avg_sell

        return DailyPrices(t, factor, avg_buy, avg_sell, close)

    # ---- convenience passthroughs to the pure helpers -------------------
    percent_change = staticmethod(percent_change)
    change_ratio = staticmethod(change_ratio)
    poisson_pmf = staticmethod(poisson_pmf)
    binomial_pmf = staticmethod(binomial_pmf)
    normal_pdf = staticmethod(normal_pdf)
    factorial = staticmethod(factorial)

    def summary(self) -> str:
        r = self.compute_daily_prices()
        cp = "n/a" if r.close_price is None else f"{r.close_price:.4f}"
        return (
            f"day #{r.day_index} | factor {r.market_factor:.4f} | "
            f"avg_buy {r.avg_buy if r.avg_buy is None else round(r.avg_buy, 4)} | "
            f"avg_sell {r.avg_sell if r.avg_sell is None else round(r.avg_sell, 4)} | "
            f"close {cp}"
        )


# ===========================================================================
#  Interactive CLI  (the MSTR.py session, with real input validation)
# ===========================================================================

def _read_quotes(book: OrderBook, side: str) -> None:
    add = book.add_buyer if side == "buyers" else book.add_seller
    print(f"\nEnter '<percent> <price>' for {side.upper()} (type STOP to finish):")
    while True:
        raw = input(">>> ").strip()
        if raw.lower() == "stop":
            break
        parts = raw.split()
        if len(parts) != 2:
            print("  need exactly two numbers: <percent> <price>")
            continue
        try:
            percent, price = float(parts[0]), float(parts[1])
        except ValueError:
            print("  both values must be numbers")
            continue
        try:
            add(percent, price)
        except ValueError as e:
            print(f"  rejected: {e}")


def _read_float(prompt: str) -> float:
    while True:
        try:
            return float(input(prompt).strip())
        except ValueError:
            print("  not a number, try again")


def run_cli() -> None:
    model = StockMathModel()
    _read_quotes(model.book, "buyers")
    _read_quotes(model.book, "sellers")

    r = model.compute_daily_prices()
    print("\n=== daily summary ===")
    print(f"day index     : {r.day_index}")
    print(f"market factor : {r.market_factor:.4f}")
    print(f"avg buy price : {r.avg_buy if r.avg_buy is None else round(r.avg_buy, 4)}")
    print(f"avg sell price: {r.avg_sell if r.avg_sell is None else round(r.avg_sell, 4)}")
    print(f"close price   : {r.close_price if r.close_price is None else round(r.close_price, 4)}")

    if input("\nCompute a percent change? [y/N] ").strip().lower() == "y":
        old = _read_float("old price: ")
        new = _read_float("new price: ")
        try:
            pct_old, pct_new, ratio = change_ratio(old, new)
            direction = "up" if new > old else ("down" if new < old else "flat")
            print(f"  change      : {pct_old:+.4f}%  ({direction})")
            print(f"  vs new price : {pct_new:+.4f}%")
            print(f"  ratio        : {ratio:.4f}" if not math.isnan(ratio) else "  ratio        : n/a (no change)")
        except ValueError as e:
            print(f"  {e}")

    print("\nnormal_pdf(41, mu=5, sigma=15) =", round(normal_pdf(41, 5, 15), 8))


# ===========================================================================
#  Self-test  (runs when there is no interactive terminal / with --test)
# ===========================================================================

def _self_test() -> None:
    assert [n for n in range(30) if is_fibonacci(n)] == [0, 1, 2, 3, 5, 8, 13, 21]
    assert is_fibonacci(832040) and not is_fibonacci(832041)          # large-n precision
    assert [n for n in range(20) if is_prime(n)] == [2, 3, 5, 7, 11, 13, 17, 19]
    assert factorial(0) == 1 and factorial(6) == 720
    assert factorial(2000) > 0                                        # no RecursionError

    try:
        factorial(-1)
    except ValueError:
        pass
    else:
        raise AssertionError("factorial(-1) should raise")

    assert abs(sum(binomial_pmf(10, k, 0.3) for k in range(11)) - 1.0) < 1e-9
    assert abs(sum(poisson_pmf(4.0, k) for k in range(60)) - 1.0) < 1e-9
    assert binomial_pmf(5, 6, 0.5) == 0.0
    assert abs(normal_pdf(0, 0, 1) - 0.3989422804014327) < 1e-12

    assert abs(percent_change(100, 110) - 10.0) < 1e-9                 # a rise is positive
    assert abs(percent_change(100, 90) + 10.0) < 1e-9
    try:
        percent_change(0, 10)
    except ValueError:
        pass
    else:
        raise AssertionError("percent_change(0, ...) should raise")

    book = OrderBook()
    book.add_buyer(60, 100)
    book.add_buyer(40, 110)
    assert abs(OrderBook.weighted_average(book.buyers) - 104.0) < 1e-9
    assert OrderBook.weighted_average(book.sellers) is None           # empty side -> None
    try:
        book.add_buyer(10, 100)                                       # would exceed 100%
    except ValueError:
        pass
    else:
        raise AssertionError("over-100% add should raise")

    m = StockMathModel(start_date=date(2020, 1, 1))
    m.book.add_buyer(100, 50)
    m.book.add_seller(100, 60)
    dp = m.compute_daily_prices(t=10)
    assert dp.close_price is not None and dp.avg_buy is not None
    assert isinstance(m.summary(), str)

    print("mstr_model self-test: all checks passed")


if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        _self_test()
    elif sys.stdin is not None and sys.stdin.isatty():
        run_cli()
    else:
        _self_test()


# ===========================================================================
#  BUGS-FIXED  (relative to MSTR.py / MSTR1.py)
# ---------------------------------------------------------------------------
#  MSTR.py
#   * percent change had the sign inverted -- a price rise printed as negative
#   * p / q divided by zero whenever the price was unchanged (q == 0)
#   * empty buyer/seller list -> ZeroDivisionError in weighted_average
#   * bare `except:` swallowed every error (incl. Ctrl-C) and mislabeled it
#   * `int(input())` / `float(input())` crashed the program on bad input
#   * recursive factorial -> RecursionError / stack overflow for large n
#   * module-level input() ran on import; no `if __name__ == "__main__"` guard
#   * int(math.sqrt(x)) in is_fibonacci/is_prime -- wrong for large x
#   * no validation of negative percents / non-positive prices
#   * unused imports (time, datetime.datetime), dead is_fibonacci/is_prime
#  MSTR1.py
#   * percent_change guarded `new == 0`; the division is by `old`
#   * weighted_average returned 0 for an empty side -> silently wrong close price
#   * recursive factorial; Poisson/Binomial via factorial -> overflow for big args
#   * Pnormal: ZeroDivisionError when sigma == 0; no sigma > 0 check
#   * float-sqrt is_fibonacci/is_prime; no CLI / entry point / tests
# ===========================================================================
