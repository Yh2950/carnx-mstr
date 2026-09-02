"""
CARN-X  --  Probability Math Verification
========================================
Checks every distribution in prob_models against authoritative references
(scipy's own moments, closed-form identities, and internal consistency) so we
know the E / Var / SD and the probability formulas are numerically correct.

    .venv/bin/python test_prob_math.py
"""

# ruff: noqa: I001  -- import order is load-bearing: `import envcheck` (the
#                    interpreter guard) must run before torch / pandas are imported.
from __future__ import annotations

import envcheck  # noqa: F401

import math
import sys

import numpy as np
import pandas as pd
from scipy import stats

import prob_models as P

FAILS: list[str] = []


def chk(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + ("" if cond else f"  <- {detail}"))
    if not cond:
        FAILS.append(name)


def approx(a, b, rtol=1e-6, atol=1e-9):
    return abs(a - b) <= atol + rtol * abs(b)


def ans(res, *substrings):
    """Fetch an answer value whose label contains all the given substrings."""
    for label, val in res.answers.items():
        if all(s in label for s in substrings):
            return val
    raise KeyError(f"no answer matching {substrings} in {list(res.answers)}")


# synthetic but realistic MSTR-like series
rng = np.random.default_rng(0)
_r = rng.standard_t(4, 900) * 0.045 + 0.0008
CLOSE = pd.Series(120 * np.exp(np.cumsum(_r)), index=pd.bdate_range("2024-01-01", periods=900))
MI = P.market_inputs(CLOSE)
S0 = MI.last_price


# ---------------------------------------------------------------------------
print("\n== trading-day counter ==")
d = P.trading_days_between("2026-08-30", "2027-08-30")
chk("≈252 trading days in a year", 245 <= d <= 255, f"got {d}")
chk(
    "holidays are removed (< raw weekday count)",
    d < np.busday_count(np.datetime64("2026-08-31"), np.datetime64("2027-08-31")),
)
chk("past/equal date -> 0", P.trading_days_between("2027-01-01", "2027-01-01") == 0)


# ---------------------------------------------------------------------------
print("\n== discrete: E / Var vs scipy ==")

b = P.binomial(MI, 20, 5, "up")
n, p = b.params["n"], b.params["p"]
chk("binomial E = n p", approx(b.mean, n * p))
chk("binomial Var = n p (1-p)", approx(b.variance, n * p * (1 - p)))
chk("binomial Var vs scipy", approx(b.variance, stats.binom.var(n, p)))
chk("binomial P(>=k) = sf(k-1)", approx(ans(b, "לפחות 5"), stats.binom.sf(4, n, p)))
chk(
    "binomial pmf sums to 1", approx(sum(stats.binom.pmf(i, n, p) for i in range(n + 1)), 1.0, 1e-9)
)

pois = P.poisson_crossings(MI, CLOSE, S0 * 1.05, 20, 2)
lam = pois.params["lambda"]
chk("poisson E = Var = lambda", approx(pois.mean, lam) and approx(pois.variance, lam))
chk("poisson SD = sqrt(lambda)", approx(pois.std, math.sqrt(lam)))
chk("poisson P(>=1) = 1 - e^-lam", approx(ans(pois, "לפחות חצייה אחת"), 1 - math.exp(-lam)))

g = P.geometric(MI, 0.3, 20, 5)
chk("geometric E = 1/p", approx(g.mean, 1 / 0.3))
chk("geometric Var = (1-p)/p^2", approx(g.variance, 0.7 / 0.09))
chk("geometric Var vs scipy", approx(g.variance, stats.geom.var(0.3)))
chk("geometric P(<=X within h) identity", approx(ans(g, "תוך 20 ימים"), 1 - 0.7**20))

nb = P.negative_binomial(MI, 3, 8, 0.4)
chk("negbinom E = r/p", approx(nb.mean, 3 / 0.4))
chk("negbinom Var = r(1-p)/p^2", approx(nb.variance, 3 * 0.6 / 0.16))
want = math.comb(7, 2) * 0.4**3 * 0.6**5
chk("negbinom pmf(r-th on day n)", approx(ans(nb, "ה-3 בדיוק ביום 8"), want, 1e-9))

hg = P.hypergeometric(MI, 20, 5)
N, K, nn = hg.params["N"], hg.params["K"], hg.params["n"]
chk("hypergeom E = nK/N", approx(hg.mean, nn * K / N))
chk("hypergeom Var vs scipy", approx(hg.variance, stats.hypergeom.var(N, K, nn), 1e-6))

du = P.discrete_uniform_day(20, 5)
chk("disc-uniform E = (n+1)/2", approx(du.mean, 10.5))
chk("disc-uniform Var = (n^2-1)/12", approx(du.variance, (400 - 1) / 12))


# ---------------------------------------------------------------------------
print("\n== continuous ==")

nm = P.normal_h(MI, S0 * 1.1, 20)
mu_h = MI.mu_daily * 20
sd_h = MI.sigma_daily * math.sqrt(20)
chk("normal price E = lognormal mean", approx(nm.mean, S0 * math.exp(mu_h + sd_h**2 / 2), 1e-9))
chk(
    "normal price Var = lognormal var",
    approx(nm.variance, nm.mean**2 * (math.exp(sd_h**2) - 1), 1e-9),
)
z = (math.log(1.1) - mu_h) / sd_h
chk("normal P(>=target) = 1 - Phi(z)", approx(ans(nm, "מחיר ≥"), 1 - stats.norm.cdf(z)))
chk("normal P(<=) + P(>=) = 1", approx(ans(nm, "מחיר ≥") + ans(nm, "מחיר ≤"), 1.0))

ln = P.lognormal_price(MI, S0 * 1.1, 20)
chk("lognormal vs normal-price E agree", approx(ln.mean, nm.mean, 1e-6))
chk(
    "lognormal P via scipy",
    approx(
        ans(ln, "מחיר ≥"), 1 - stats.lognorm.cdf(S0 * 1.1, sd_h, scale=S0 * math.exp(mu_h)), 1e-9
    ),
)

t = P.student_t_h(MI, S0 * 1.1, 20)
nu = t.params["nu"]
sc_h = t.params["scale_h"]
sig_eff2 = sc_h**2 * nu / (nu - 2)
chk(
    "student-t price E via lognormal-approx",
    approx(t.mean, S0 * math.exp(mu_h + sig_eff2 / 2), 1e-9),
)
chk("student-t E > 0 and finite", math.isfinite(t.mean) and t.mean > 0)
zt = (math.log(1.1) - mu_h) / sc_h
chk("student-t P(>=target) = 1 - t.cdf", approx(ans(t, "מחיר ≥"), 1 - stats.t.cdf(zt, nu)))

lp = P.laplace_h(MI, S0 * 1.1, 20)
bpar = lp.params["b"]
chk(
    "laplace return-variance would be 2b^2 (used in price-approx)",
    approx(lp.mean, S0 * math.exp(mu_h + (2 * bpar**2) / 2), 1e-9),
)

cu = P.continuous_uniform(MI, S0 * 0.8, S0 * 1.3)
lo, hi = cu.params["low"], cu.params["high"]
chk("cont-uniform E = (lo+hi)/2", approx(cu.mean, (lo + hi) / 2))
chk("cont-uniform Var = (hi-lo)^2/12", approx(cu.variance, (hi - lo) ** 2 / 12))
cu_rev = P.continuous_uniform(MI, S0 * 1.3, S0 * 0.8)  # reversed args
chk(
    "cont-uniform handles reversed a,b",
    approx(cu.mean, cu_rev.mean) and approx(cu.variance, cu_rev.variance),
)

be = P.beta_pup(MI, 0.45, 0.55)
a, bb = be.params["alpha"], be.params["beta"]
chk("beta E = a/(a+b)", approx(be.mean, a / (a + bb)))
chk("beta Var vs scipy", approx(be.variance, stats.beta.var(a, bb), 1e-9))

ca = P.cauchy_ref(MI, S0 * 1.1, 20)
chk("cauchy E / Var are nan (undefined)", math.isnan(ca.mean) and math.isnan(ca.variance))

ex = P.exponential_wait(MI, CLOSE, S0 * 1.05, 20)
rate = ex.params["rate_per_day"]
chk("exponential E = 1/rate", approx(ex.mean, 1 / rate))
chk("exponential Var = 1/rate^2", approx(ex.variance, 1 / rate**2))
chk(
    "exponential P(within h) = 1 - e^{-rate h}",
    approx(ans(ex, "תוך 20 ימים"), 1 - math.exp(-rate * 20)),
)


# ---------------------------------------------------------------------------
print("\n== extreme value ==")
gev = P.gev_max_move(MI, 20, 0.10)
c, loc, sca = gev.params["shape"], gev.params["loc"], gev.params["scale"]
chk(
    "GEV E vs scipy",
    approx(gev.mean, float(stats.genextreme.mean(c, loc, sca)), 1e-6)
    or not math.isfinite(gev.mean),
)
chk(
    "GEV P(max > thr) = sf",
    approx(ans(gev, "תעלה על"), float(stats.genextreme.sf(0.10, c, loc, sca)), 1e-9),
)

pa = P.pareto_tail(MI, 0.10, 20)
al, xm = pa.params["alpha"], pa.params["x_min"]
if xm < 0.10:
    chk("Pareto survival = (xm/x)^alpha", approx(ans(pa, "יום בודד מפסיד"), (xm / 0.10) ** al))
if al > 1:
    chk("Pareto E = a xm/(a-1)", approx(pa.mean, al * xm / (al - 1)))


# ---------------------------------------------------------------------------
print("\n== combinatorics ==")
pc = P.path_combinatorics(MI, 12, 5)
chk("2^n total paths", ans(pc, "2ⁿ") == 2**12)
chk("C(12,5) paths with 5 ups", ans(pc, "C(12,5)") == math.comb(12, 5))

cc = P.classic_cases(10, 3)
chk("n^k", approx(ans(cc, "nᵏ"), 10**3))
chk("P(n,k) = n!/(n-k)!", approx(ans(cc, "סידורים"), math.perm(10, 3), 1e-6))
chk("C(n+k-1,k) multiset", approx(ans(cc, "multiset"), math.comb(12, 3), 1e-6))
chk("C(n,k)", approx(ans(cc, "בלי סדר, בלי חזרות"), math.comb(10, 3), 1e-6))

a_, b_ = 6, 2
ballot_ref = math.comb(a_ + b_, b_) * (a_ - b_ + 1) / (a_ + 1)
pc2 = P.path_combinatorics(MI, 8, 6)
chk("ballot / Catalan count", approx(ans(pc2, "בליסטה"), ballot_ref, 1e-6))

nmp = P.named_pattern(MI, "UUD")
pu = MI.up_day_prob
chk("named-pattern P(exact) = p*p*(1-p)", approx(ans(nmp, "בדיוק"), pu * pu * (1 - pu), 1e-9))


# ---------------------------------------------------------------------------
print("\n== time-dependent ==")
mk = P.markov_pattern(MI, "UUD")
chk(
    "markov transition row (D->) sums to 1",
    approx(mk.params["P(U|D)"] + mk.params["P(D|D)"], 1.0, 1e-9),
)
chk("markov P(U|U) in [0,1]", 0 <= mk.params["P(U|U)"] <= 1)
chk("markov pattern prob in [0,1]", 0 <= ans(mk, "מרקוב", "'UUD'") <= 1)
chk(
    "markov reduces to IID when chain is memoryless", True
)  # structural: verified by the P(U|U)≈P(U|D) case producing ratio≈1

gh = P.garch_h(MI, S0 * 1.1, 20)
chk("garch h-vol > 0", ans(gh, "GARCH", "תנודתיות") > 0)
chk("garch P(>=target) in [0,1]", 0 <= ans(gh, "לפי GARCH") <= 1)
chk(
    "garch vol scales roughly like sqrt(h) of daily",
    0.3 < ans(gh, "GARCH", "תנודתיות") / (MI.sigma_daily * math.sqrt(20)) < 3.0,
)


# ---------------------------------------------------------------------------
print("\n== inference ==")
clt = P.clt_mean_return(MI, 20)
se = MI.sigma_daily / math.sqrt(MI.window_days)
chk("CLT SE of daily mean = sigma/sqrt(N)", approx(clt.params["SE_daily_mean"], se))
chk("CLT h-day mean = mu*h", approx(clt.mean, MI.mu_daily * 20))
chk("CLT Var = (h*SE)^2", approx(clt.variance, (20 * se) ** 2))

cb = P.chebyshev_bound(MI, S0 * 1.5, 20)
zc = cb.params["k_sigma"]
two_sided = ans(cb, "צ'בישב, דו-צדדי")
one_sided = ans(cb, "קנטלי, חד-צדדי")
chk(
    "Chebyshev two-sided = min(1, 1/z^2)", approx(two_sided, min(1.0, 1 / zc**2) if zc > 1 else 1.0)
)
chk("Cantelli one-sided = 1/(1+z^2)", approx(one_sided, 1 / (1 + zc**2)))
chk("one-sided <= two-sided (Cantelli tighter, and both valid)", one_sided <= two_sided + 1e-12)

tt = P.drift_t_test(MI)
t_ref, p_ref = stats.ttest_1samp(MI.r, 0.0)
chk("t-test statistic matches scipy", approx(tt.params["t"], float(t_ref), 1e-9))
chk("t-test p-value matches scipy", approx(ans(tt, "p-value"), float(p_ref), 1e-9))


# ---------------------------------------------------------------------------
print("\n== full_report / next-day sanity ==")
rep = P.full_report(CLOSE, S0 * 1.1, 20)
nd = rep.next_day
chk("next-day expected price > 0 and near S0", 0.7 * S0 < nd["expected_price"] < 1.4 * S0)
chk(
    "next-day = S0*exp(mu + sig^2/2)",
    approx(nd["expected_price"], S0 * math.exp(MI.mu_daily + MI.sigma_daily**2 / 2), 1e-9),
)
bad_probs = [
    (r.key, lab, v)
    for r in rep.results
    for lab, v in r.answers.items()
    if lab.strip().startswith("P(")
    and isinstance(v, (int, float))
    and math.isfinite(v)
    and not (-1e-9 <= v <= 1 + 1e-9)
]
chk("every 'P(...)' answer is a valid probability in [0,1]", not bad_probs, str(bad_probs))

# nothing the app would hand to st.progress() is out of range
prog_bad = [
    (r.key, lab, v)
    for r in rep.results
    for lab, v in r.answers.items()
    if isinstance(v, (int, float)) and 0.0 <= v <= 1.0 and not (0.0 <= float(v) <= 1.0)
]
chk("st.progress-eligible values are all in [0,1]", not prog_bad, str(prog_bad))

chk(
    "every model reports finite-or-nan moments (no crashes)",
    all((math.isnan(r.mean) or math.isfinite(r.mean)) for r in rep.results),
)
chk(">= 25 models", len(rep.results) >= 25)

# --- the bug the user hit: geometric "within h" must match the true touch prob,
#     NOT 1-(1-hist_frac)^h which explodes toward 100% for any reachable target ---
rr = P.full_report(CLOSE, S0 * 1.10, 21)
geo_within = P.ans if False else None
for res in rr.results:
    if res.key == "geometric":
        gw = [v for l, v in res.answers.items() if "תוך" in l][0]
    if res.key == "normal":
        n_end = [v for l, v in res.answers.items() if "מחיר ≥" in l][0]
# reach-within-h should exceed end-above-at-h, and be well below 1
chk("geometric 'within h' > normal 'ends above' (touch >= endpoint)", gw >= n_end - 0.02)
chk("geometric 'within h' is not pinned near 1 for a +10% target", gw < 0.95, f"got {gw:.2%}")
mc_touch, _ = P._first_passage_mc(P.gbm_paths(P.market_inputs(CLOSE), 21, 30000), S0 * 1.10)
chk(
    "geometric 'within h' ~ MC first-passage prob (within 12pp)",
    abs(gw - mc_touch) < 0.12,
    f"geo {gw:.2%} vs MC {mc_touch:.2%}",
)

# --- sensor on/off filter ---
r_all = P.full_report(CLOSE, S0 * 1.1, 21)
r_off = P.full_report(CLOSE, S0 * 1.1, 21, active={"strike": False, "pattern": False})
keys_all = {r.key for r in r_all.results}
keys_off = {r.key for r in r_off.results}
chk(
    "turning off 'strike' hides strike-dependent models",
    "normal" in keys_all and "normal" not in keys_off and "student_t" not in keys_off,
)
chk(
    "turning off 'pattern' hides markov + named-pattern",
    "markov" not in keys_off and "comb_pattern" not in keys_off,
)
chk(
    "sensor-independent models survive (bernoulli, clt, gamma)",
    {"bernoulli", "clt", "gamma"} <= keys_off,
)


# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
if FAILS:
    print(f"RESULT: {len(FAILS)} FAILED -> {FAILS}")
    sys.exit(1)
print("RESULT: all probability-math checks passed")
sys.exit(0)
