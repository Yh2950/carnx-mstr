import math
import time
from datetime import datetime, date

def is_fibonacci(n):
    def is_square(x):
        return int(math.sqrt(x))**2 == x
    return is_square(5 * n * n + 4) or is_square(5 * n * n - 4)

def is_prime(n):
    if n < 2: return False
    for i in range(2, int(n**0.5)+1):
        if n % i == 0:
            return False
    return True

# ===============================
#  TIME & MARKET STATE
# ===============================

START_DATE = date.today()

def day_index():
    return (date.today() - START_DATE).days

def market_factor(t, A=0.12, T=100, eps=0.04):
    """
    מצב שוק דטרמיניסטי:
    >1 בועה | <1 משבר
    """
    return 1 + A * math.sin((2 * math.pi * t) / T) \
        + eps * math.sin(t) * math.sin(math.sqrt(2) * t)

def adjust_participants(data, factor):
    adjusted = []
    for percent, price in data:
        adjusted.append((percent, price * factor))
    return adjusted


buyers = []
total_percent1 = 0

print("Enter percent && Enter price (BUYERS)")
print("לסיום כתוב STOP\n")

while True:
    User_input = input(">>> ")
    if User_input.lower() == "stop":
        break
    try:
        percent, price = map(float, User_input.split())
        if percent > 100 or total_percent1 + percent > 100:
            raise ValueError("Invalid percent")

        buyers.append((percent, price))
        total_percent1 += percent

    except:
        print("פורמט אינו תקין")

sellers = []
total_percent2 = 0

print("\nEnter percent && Enter price (SELLERS)")
print("לסיום כתוב STOP\n")

while True:
    User_input = input(">>> ")
    if User_input.lower() == "stop":
        break
    try:
        percent, price = map(float, User_input.split())
        if percent > 100 or total_percent2 + percent > 100:
            raise ValueError("Invalid percent")

        sellers.append((percent, price))
        total_percent2 += percent

    except:
        print("פורמט אינו תקין")

# ===============================
#  DAILY MARKET CALCULATION
# ===============================

t = day_index()
factor = market_factor(t)

buyers_t  = adjust_participants(buyers, factor)
sellers_t = adjust_participants(sellers, factor)

def weighted_average(data):
    num = sum(p * pr for p, pr in data)
    den = sum(p for p, _ in data)
    return num / den

avg_buy  = weighted_average(buyers_t)
avg_sell = weighted_average(sellers_t)

Sp = (avg_buy + avg_sell) / 2

print("\n📊 סיכום יומי")
print("-" * 30)
print(f"Day index        : {t}")
print(f"Market factor   : {factor:.4f}")
print(f"Avg Buy Price   : {avg_buy:.4f}")
print(f"Avg Sell Price  : {avg_sell:.4f}")
print(f"📈 Close Price  : {Sp:.4f}")
print("-" * 30)
z = int(input("\nלחשב עלייה הזן 1 | ירידה הזן 2 : "))
x = float(input("Enter Old Price: "))
y = float(input("Enter New Price: "))

p = ((x - y) / y) * 100
q = ((x - y) / x) * 100

print("Result:", p if z == 1 else q)

print("\nיחס עלייה/ירידה:", p / q)

def factorial(n):
    return 1 if n == 0 else n * factorial(n - 1)

def nCk(n, k):
    return factorial(n) / (factorial(k) * factorial(n - k))

def Ppoason(g, k):
    return (g**k * math.exp(-g)) / factorial(k)

def Pbinomi(n, k, p):
    return nCk(n, k) * p**k * (1 - p)**(n - k)

def Pnormal(x, mu, sigma):
    return (1 / math.sqrt(2 * math.pi * sigma**2)) * \
           math.exp(-((x - mu)**2) / (2 * sigma**2))

print("\nNormal distribution example:", Pnormal(41, 5, 15))

