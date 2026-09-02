"""
CARN-X  –  Diagnosis & Characterization Layer (Layer 1)
=======================================================
שכבת אבחון ואפיון מקיפה לרשת הנוירונים.
מבצעת ניתוחים סטטיסטיים, הסתברותיים, זיהוי סדר/חזרות/סימטריה/קיצון,
מדידת תנודתיות, מחזוריות, קורלציות, התפלגויות ועוד.

מיועד לשמש כשכבה ראשונה בארכיטקטורת CARN-X.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pywt
from scipy import fft, signal, stats
from scipy.stats import (
    anderson,
    gaussian_kde,
    genextreme,
    jarque_bera,
    kendalltau,
    kstest,
    norm,
    normaltest,
    pearsonr,
    shapiro,
    skewnorm,
    spearmanr,
)
from scipy.stats import (
    entropy as scipy_entropy,
)
from scipy.stats import (
    t as student_t,
)
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, grangercausalitytests, kpss, pacf

# Optional advanced libraries (graceful fallback)
try:
    from arch import arch_model

    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

try:
    import ruptures as rpt

    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Data classes for structured output
# ---------------------------------------------------------------------------


@dataclass
class BasicStats:
    n_obs: int
    mean: float
    median: float
    mode_approx: float
    variance: float
    std: float
    skewness: float
    kurtosis: float
    min_val: float
    max_val: float
    range: float
    iqr: float
    mad: float  # Median Absolute Deviation
    cv: float  # Coefficient of Variation
    quantiles: dict[str, float] = field(default_factory=dict)


@dataclass
class DistributionFit:
    best_fit_name: str
    best_fit_params: dict[str, float]
    aic: float
    bic: float
    ks_statistic: float
    ks_pvalue: float
    is_normal_jb: bool
    is_normal_shapiro: bool
    jarque_bera_stat: float
    jarque_bera_pvalue: float
    anderson_stat: float
    anderson_critical: dict[str, float]
    entropy: float
    kde_bandwidth: float | None = None


@dataclass
class VolatilityProfile:
    historical_vol: float
    rolling_vol_mean: float
    rolling_vol_std: float
    parkinson_vol: float | None
    garman_klass_vol: float | None
    arch_lm_stat: float | None
    arch_lm_pvalue: float | None
    garch_volatility: np.ndarray | None = None
    garch_params: dict[str, float] | None = None


@dataclass
class CyclicityProfile:
    dominant_period: float | None
    dominant_frequency: float | None
    spectral_entropy: float
    acf_lags: np.ndarray
    acf_values: np.ndarray
    pacf_values: np.ndarray
    significant_acf_lags: list[int]
    stl_trend_strength: float | None
    stl_seasonal_strength: float | None
    wavelet_energy: dict[str, float] = field(default_factory=dict)


@dataclass
class ExtremesProfile:
    n_outliers_iqr: int
    n_outliers_zscore: int
    n_outliers_mad: int
    outlier_indices: list[int]
    max_drawdown: float
    max_drawup: float
    peak_count: int
    trough_count: int
    extreme_value_shape: float | None  # GEV shape
    extreme_value_loc: float | None
    extreme_value_scale: float | None
    change_points: list[int] = field(default_factory=list)


@dataclass
class OrderRepetitionSymmetry:
    is_strictly_increasing: bool
    is_strictly_decreasing: bool
    is_monotonic: bool
    trend_slope: float
    trend_pvalue: float
    unique_ratio: float  # 1.0 = no repetitions
    entropy_normalized: float  # 0..1
    repetition_score: float  # higher = more repetitions
    symmetry_score: float  # 0..1 (1 = perfect symmetry around mean)
    autocorrelation_lag1: float
    runs_test_stat: float | None
    runs_test_pvalue: float | None
    order_importance_proxy: float  # heuristic 0..1


@dataclass
class CorrelationProfile:
    pairwise_pearson: np.ndarray | None
    pairwise_spearman: np.ndarray | None
    pairwise_kendall: np.ndarray | None
    condition_number: float | None
    pca_explained_variance_ratio: np.ndarray | None
    macro_micro_proxy: dict[str, float] | None = None


@dataclass
class CharacterizationResult:
    """תוצאה מלאה של שכבת האבחון."""

    basic: BasicStats
    distribution: DistributionFit
    volatility: VolatilityProfile
    cyclicity: CyclicityProfile
    extremes: ExtremesProfile
    order_rep_sym: OrderRepetitionSymmetry
    correlation: CorrelationProfile
    raw_summary: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"CharacterizationResult(\n"
            f"  n_obs={self.basic.n_obs},\n"
            f"  mean={self.basic.mean:.4f}, std={self.basic.std:.4f},\n"
            f"  skew={self.basic.skewness:.3f}, kurt={self.basic.kurtosis:.3f},\n"
            f"  unique_ratio={self.order_rep_sym.unique_ratio:.3f},\n"
            f"  dominant_period={self.cyclicity.dominant_period},\n"
            f"  max_drawdown={self.extremes.max_drawdown:.4f},\n"
            f"  recommendations={len(self.recommendations)}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Main Diagnosis Layer
# ---------------------------------------------------------------------------


class DiagnosisLayer:
    """
    שכבת אבחון ואפיון מקיפה.
    מקבלת סדרת נתונים (1D או DataFrame) ומחזירה CharacterizationResult עשיר.
    """

    def __init__(
        self,
        rolling_window: int = 20,
        acf_nlags: int = 40,
        outlier_z: float = 3.0,
        random_state: int = 42,
    ):
        self.rolling_window = rolling_window
        self.acf_nlags = acf_nlags
        self.outlier_z = outlier_z
        self.random_state = random_state
        np.random.seed(random_state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def diagnose(
        self,
        data: np.ndarray | pd.Series | pd.DataFrame | list[float],
        high: np.ndarray | None = None,
        low: np.ndarray | None = None,
        volume: np.ndarray | None = None,
        macro_features: pd.DataFrame | None = None,
    ) -> CharacterizationResult:
        """
        מריץ אבחון מלא על הקלט.
        """
        series, multi = self._prepare_input(data)
        n = len(series)

        if n < 5:
            raise ValueError("נדרשות לפחות 5 תצפיות לאבחון משמעותי.")

        basic = self._compute_basic_stats(series)
        dist = self._fit_distribution(series)
        vol = self._compute_volatility(series, high=high, low=low)
        cyc = self._compute_cyclicity(series)
        ext = self._compute_extremes(series)
        ors = self._compute_order_repetition_symmetry(series)
        corr = self._compute_correlations(series, multi, macro_features)
        recommendations = self._generate_recommendations(basic, dist, vol, cyc, ext, ors, corr)

        result = CharacterizationResult(
            basic=basic,
            distribution=dist,
            volatility=vol,
            cyclicity=cyc,
            extremes=ext,
            order_rep_sym=ors,
            correlation=corr,
            recommendations=recommendations,
            raw_summary={
                "n_obs": n,
                "has_multi": multi is not None,
                "has_macro": macro_features is not None,
            },
        )
        return result

    def _prepare_input(self, data: Any) -> tuple[np.ndarray, np.ndarray | None]:
        if isinstance(data, pd.DataFrame):
            if data.shape[1] == 1:
                series = data.iloc[:, 0].astype(float).values
                multi = None
            else:
                series = data.iloc[:, 0].astype(float).values
                multi = data.astype(float).values
        elif isinstance(data, pd.Series):
            series = data.astype(float).values
            multi = None
        else:
            arr = np.asarray(data, dtype=float)
            if arr.ndim == 1:
                series = arr
                multi = None
            elif arr.ndim == 2:
                series = arr[:, 0]
                multi = arr
            else:
                raise ValueError("data חייב להיות 1D או 2D.")
        mask = np.isfinite(series)
        series = series[mask]
        if multi is not None:
            multi = multi[mask]
        return series, multi

    def _compute_basic_stats(self, x: np.ndarray) -> BasicStats:
        n = len(x)
        mean = float(np.mean(x))
        median = float(np.median(x))
        hist, bin_edges = np.histogram(x, bins="auto")
        mode_approx = float(bin_edges[np.argmax(hist)])
        var = float(np.var(x, ddof=1)) if n > 1 else 0.0
        std = float(np.std(x, ddof=1)) if n > 1 else 0.0
        skew = float(stats.skew(x, bias=False)) if n > 2 else 0.0
        kurt = float(stats.kurtosis(x, bias=False)) if n > 3 else 0.0
        q = np.quantile(x, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
        quantiles = {
            "q01": float(q[0]),
            "q05": float(q[1]),
            "q10": float(q[2]),
            "q25": float(q[3]),
            "q50": float(q[4]),
            "q75": float(q[5]),
            "q90": float(q[6]),
            "q95": float(q[7]),
            "q99": float(q[8]),
        }
        iqr = quantiles["q75"] - quantiles["q25"]
        mad = float(np.median(np.abs(x - median)))
        cv = std / mean if mean != 0 else np.nan
        return BasicStats(
            n_obs=n,
            mean=mean,
            median=median,
            mode_approx=mode_approx,
            variance=var,
            std=std,
            skewness=skew,
            kurtosis=kurt,
            min_val=float(np.min(x)),
            max_val=float(np.max(x)),
            range=float(np.ptp(x)),
            iqr=float(iqr),
            mad=mad,
            cv=float(cv) if np.isfinite(cv) else np.nan,
            quantiles=quantiles,
        )

    def _fit_distribution(self, x: np.ndarray) -> DistributionFit:
        jb_stat, jb_p = jarque_bera(x)
        try:
            _shapiro_stat, shapiro_p = shapiro(x[:5000] if len(x) > 5000 else x)
            is_normal_shapiro = shapiro_p > 0.05
        except Exception:
            _shapiro_stat, shapiro_p = np.nan, np.nan
            is_normal_shapiro = False
        is_normal_jb = jb_p > 0.05
        anderson_result = anderson(x, dist="norm")
        anderson_crit = {
            str(sl): float(cv)
            for sl, cv in zip(anderson_result.significance_level, anderson_result.critical_values)
        }
        hist, _ = np.histogram(x, bins="auto", density=True)
        hist = hist[hist > 0]
        ent = float(scipy_entropy(hist)) if len(hist) > 0 else 0.0
        candidates = {"norm": norm, "t": student_t, "skewnorm": skewnorm, "genextreme": genextreme}
        best_name = "norm"
        best_params: dict[str, float] = {}
        best_aic = np.inf
        best_bic = np.inf
        best_ks_stat = np.inf
        best_ks_p = 0.0
        for name, dist in candidates.items():
            try:
                params = dist.fit(x)
                loglik = np.sum(dist.logpdf(x, *params))
                k = len(params)
                n = len(x)
                aic = 2 * k - 2 * loglik
                bic = k * np.log(n) - 2 * loglik
                ks_stat, ks_p = kstest(x, lambda v: dist.cdf(v, *params))
                if aic < best_aic:
                    best_aic = aic
                    best_bic = bic
                    best_name = name
                    best_params = {f"p{i}": float(p) for i, p in enumerate(params)}
                    best_ks_stat = float(ks_stat)
                    best_ks_p = float(ks_p)
            except Exception:
                continue
        try:
            kde = gaussian_kde(x)
            bw = float(kde.factor * np.std(x, ddof=1))
        except Exception:
            bw = None
        return DistributionFit(
            best_fit_name=best_name,
            best_fit_params=best_params,
            aic=float(best_aic),
            bic=float(best_bic),
            ks_statistic=best_ks_stat,
            ks_pvalue=best_ks_p,
            is_normal_jb=bool(is_normal_jb),
            is_normal_shapiro=bool(is_normal_shapiro),
            jarque_bera_stat=float(jb_stat),
            jarque_bera_pvalue=float(jb_p),
            anderson_stat=float(anderson_result.statistic),
            anderson_critical=anderson_crit,
            entropy=ent,
            kde_bandwidth=bw,
        )

    def _compute_volatility(
        self, x: np.ndarray, high: np.ndarray | None = None, low: np.ndarray | None = None
    ) -> VolatilityProfile:
        rets = np.diff(np.log(np.abs(x) + 1e-12))
        hist_vol = float(np.std(rets) * np.sqrt(252)) if len(rets) > 1 else 0.0
        if len(x) >= self.rolling_window:
            s = pd.Series(x)
            roll_std = s.pct_change().rolling(self.rolling_window).std() * np.sqrt(252)
            roll_mean = float(roll_std.mean())
            roll_std_val = float(roll_std.std())
        else:
            roll_mean = hist_vol
            roll_std_val = 0.0
        park = None
        gk = None
        if high is not None and low is not None and len(high) == len(low) == len(x):
            log_hl = np.log(high / low)
            park = float(np.sqrt(np.mean(log_hl**2) / (4 * np.log(2))) * np.sqrt(252))
            log_co = np.log(x[1:] / x[:-1]) if len(x) > 1 else np.array([0.0])
            gk = float(
                np.sqrt(np.mean(0.5 * log_hl[1:] ** 2 - (2 * np.log(2) - 1) * log_co**2))
                * np.sqrt(252)
            )
        arch_stat, arch_p = None, None
        try:
            if len(rets) > 20:
                lm = het_arch(rets, nlags=5)
                arch_stat = float(lm[0])
                arch_p = float(lm[1])
        except Exception:
            pass
        garch_vol = None
        garch_params = None
        if HAS_ARCH and len(rets) > 50:
            try:
                am = arch_model(rets * 100, vol="Garch", p=1, q=1, rescale=False)
                res = am.fit(disp="off")
                garch_vol = res.conditional_volatility.values / 100
                garch_params = {
                    "omega": float(res.params.get("omega", np.nan)),
                    "alpha": float(res.params.get("alpha[1]", np.nan)),
                    "beta": float(res.params.get("beta[1]", np.nan)),
                }
            except Exception:
                pass
        return VolatilityProfile(
            historical_vol=hist_vol,
            rolling_vol_mean=roll_mean,
            rolling_vol_std=roll_std_val,
            parkinson_vol=park,
            garman_klass_vol=gk,
            arch_lm_stat=arch_stat,
            arch_lm_pvalue=arch_p,
            garch_volatility=garch_vol,
            garch_params=garch_params,
        )

    def _compute_cyclicity(self, x: np.ndarray) -> CyclicityProfile:
        n = len(x)
        x_det = signal.detrend(x)
        freqs = fft.fftfreq(n)
        spectrum = np.abs(fft.fft(x_det))
        pos_mask = freqs > 0
        if np.any(pos_mask):
            dominant_idx = np.argmax(spectrum[pos_mask])
            dominant_freq = float(freqs[pos_mask][dominant_idx])
            dominant_period = 1.0 / dominant_freq if dominant_freq > 0 else None
        else:
            dominant_freq = None
            dominant_period = None
        ps = spectrum[pos_mask] ** 2
        ps = ps / (ps.sum() + 1e-12)
        spectral_ent = float(-np.sum(ps * np.log2(ps + 1e-12)))
        nlags = min(self.acf_nlags, n // 2 - 1)
        acf_vals = acf(x, nlags=nlags, fft=True)
        try:
            pacf_vals = pacf(x, nlags=nlags, method="yw")
        except Exception:
            pacf_vals = np.zeros(nlags + 1)
        conf = 1.96 / np.sqrt(n)
        sig_lags = [i for i in range(1, len(acf_vals)) if abs(acf_vals[i]) > conf]
        trend_str, seas_str = None, None
        if n >= 50:
            try:
                stl = STL(x, period=min(13, n // 4), robust=True)
                res = stl.fit()
                var_resid = np.var(res.resid)
                trend_str = max(0.0, 1.0 - var_resid / (np.var(res.trend + res.resid) + 1e-12))
                seas_str = max(0.0, 1.0 - var_resid / (np.var(res.seasonal + res.resid) + 1e-12))
            except Exception:
                pass
        wavelet_energy = {}
        try:
            coeffs = pywt.wavedec(x_det, "db4", level=min(4, pywt.dwt_max_level(n, "db4")))
            total_e = sum(np.sum(c**2) for c in coeffs) + 1e-12
            for i, c in enumerate(coeffs):
                wavelet_energy[f"level_{i}"] = float(np.sum(c**2) / total_e)
        except Exception:
            pass
        return CyclicityProfile(
            dominant_period=dominant_period,
            dominant_frequency=dominant_freq,
            spectral_entropy=spectral_ent,
            acf_lags=np.arange(len(acf_vals)),
            acf_values=acf_vals,
            pacf_values=pacf_vals,
            significant_acf_lags=sig_lags,
            stl_trend_strength=float(trend_str) if trend_str is not None else None,
            stl_seasonal_strength=float(seas_str) if seas_str is not None else None,
            wavelet_energy=wavelet_energy,
        )

    def _compute_extremes(self, x: np.ndarray) -> ExtremesProfile:
        n = len(x)
        median = np.median(x)
        mad = np.median(np.abs(x - median)) + 1e-12
        q25, q75 = np.quantile(x, [0.25, 0.75])
        iqr = q75 - q25 + 1e-12
        z_scores = np.abs((x - np.mean(x)) / (np.std(x) + 1e-12))
        outliers_z = np.where(z_scores > self.outlier_z)[0].tolist()
        outliers_iqr = np.where((x < q25 - 1.5 * iqr) | (x > q75 + 1.5 * iqr))[0].tolist()
        outliers_mad = np.where(np.abs(x - median) / mad > 3.5)[0].tolist()
        all_outliers = sorted(set(outliers_z + outliers_iqr + outliers_mad))
        cummax = np.maximum.accumulate(x)
        drawdown = (x - cummax) / (cummax + 1e-12)
        max_dd = float(np.min(drawdown))
        cummin = np.minimum.accumulate(x)
        drawup = (x - cummin) / (np.abs(cummin) + 1e-12)
        max_du = float(np.max(drawup))
        peaks, _ = signal.find_peaks(x, distance=max(3, n // 50))
        troughs, _ = signal.find_peaks(-x, distance=max(3, n // 50))
        gev_shape = gev_loc = gev_scale = None
        try:
            block = max(10, n // 20)
            if n >= 30:
                blocks = [x[i : i + block].max() for i in range(0, n - block + 1, block)]
                if len(blocks) >= 5:
                    shape, loc, scale = genextreme.fit(blocks)
                    gev_shape, gev_loc, gev_scale = float(shape), float(loc), float(scale)
        except Exception:
            pass
        change_pts = []
        if HAS_RUPTURES and n >= 30:
            try:
                algo = rpt.Pelt(model="rbf", min_size=5).fit(x.reshape(-1, 1))
                change_pts = algo.predict(pen=3)[:-1]
            except Exception:
                pass
        return ExtremesProfile(
            n_outliers_iqr=len(outliers_iqr),
            n_outliers_zscore=len(outliers_z),
            n_outliers_mad=len(outliers_mad),
            outlier_indices=all_outliers[:50],
            max_drawdown=max_dd,
            max_drawup=max_du,
            peak_count=len(peaks),
            trough_count=len(troughs),
            extreme_value_shape=gev_shape,
            extreme_value_loc=gev_loc,
            extreme_value_scale=gev_scale,
            change_points=change_pts,
        )

    def _compute_order_repetition_symmetry(self, x: np.ndarray) -> OrderRepetitionSymmetry:
        n = len(x)
        diffs = np.diff(x)
        is_inc = bool(np.all(diffs > 0))
        is_dec = bool(np.all(diffs < 0))
        is_mono = is_inc or is_dec
        t = np.arange(n)
        slope, intercept, r_value, p_value, std_err = stats.linregress(t, x)
        unique_ratio = len(np.unique(x)) / n
        hist, _ = np.histogram(x, bins=min(50, n // 2))
        p = hist / hist.sum()
        p = p[p > 0]
        ent = -np.sum(p * np.log2(p))
        max_ent = np.log2(len(p)) if len(p) > 1 else 1.0
        ent_norm = float(ent / max_ent) if max_ent > 0 else 0.0
        repetition_score = 1.0 - unique_ratio
        mean = np.mean(x)
        left = x[x <= mean]
        right = x[x > mean]
        if len(left) > 5 and len(right) > 5:
            lq = np.quantile(mean - left, [0.25, 0.5, 0.75])
            rq = np.quantile(right - mean, [0.25, 0.5, 0.75])
            sym_score = 1.0 - np.mean(np.abs(lq - rq)) / (np.std(x) + 1e-12)
            sym_score = float(np.clip(sym_score, 0, 1))
        else:
            sym_score = 0.5
        if n > 2 and np.std(x) > 1e-12:
            ac1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
        else:
            ac1 = 0.0  # corrcoef is undefined (NaN) on a zero-variance series
        runs_stat = runs_p = None
        try:
            median = np.median(x)
            signs = (x > median).astype(int)
            runs = 1 + np.sum(signs[1:] != signs[:-1])
            n1 = np.sum(signs)
            n2 = n - n1
            if n1 > 0 and n2 > 0:
                exp_runs = 1 + 2 * n1 * n2 / n
                var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n)) / (n**2 * (n - 1))
                if var_runs > 0:
                    z = (runs - exp_runs) / np.sqrt(var_runs)
                    runs_stat = float(z)
                    runs_p = float(2 * (1 - norm.cdf(abs(z))))
        except Exception:
            pass
        order_proxy = float(
            0.4 * min(1.0, abs(slope) * n / (np.std(x) + 1e-12))
            + 0.4 * abs(ac1)
            + 0.2 * (1.0 if is_mono else 0.0)
        )
        order_proxy = float(np.clip(order_proxy, 0, 1))
        return OrderRepetitionSymmetry(
            is_strictly_increasing=is_inc,
            is_strictly_decreasing=is_dec,
            is_monotonic=is_mono,
            trend_slope=float(slope),
            trend_pvalue=float(p_value),
            unique_ratio=float(unique_ratio),
            entropy_normalized=ent_norm,
            repetition_score=float(repetition_score),
            symmetry_score=sym_score,
            autocorrelation_lag1=ac1,
            runs_test_stat=runs_stat,
            runs_test_pvalue=runs_p,
            order_importance_proxy=order_proxy,
        )

    def _compute_correlations(
        self, series: np.ndarray, multi: np.ndarray | None, macro: pd.DataFrame | None
    ) -> CorrelationProfile:
        pearson_mat = spearman_mat = kendall_mat = None
        cond_num = None
        pca_var = None
        macro_micro = None
        if multi is not None and multi.shape[1] > 1:
            try:
                pearson_mat = np.corrcoef(multi.T)
                spearman_mat = np.zeros_like(pearson_mat)
                kendall_mat = np.zeros_like(pearson_mat)
                for i in range(multi.shape[1]):
                    for j in range(multi.shape[1]):
                        spearman_mat[i, j] = spearmanr(multi[:, i], multi[:, j])[0]
                        kendall_mat[i, j] = kendalltau(multi[:, i], multi[:, j])[0]
            except Exception:
                pass
            try:
                cov = EmpiricalCovariance().fit(multi)
                cond_num = float(np.linalg.cond(cov.covariance_))
            except Exception:
                pass
            try:
                pca = PCA(n_components=min(5, multi.shape[1]))
                pca.fit(StandardScaler().fit_transform(multi))
                pca_var = pca.explained_variance_ratio_
            except Exception:
                pass
        if macro is not None and len(macro) == len(series):
            try:
                macro_micro = {}
                for col in macro.columns:
                    r, p = pearsonr(series, macro[col].values)
                    macro_micro[str(col)] = {"corr": float(r), "pvalue": float(p)}
            except Exception:
                pass
        return CorrelationProfile(
            pairwise_pearson=pearson_mat,
            pairwise_spearman=spearman_mat,
            pairwise_kendall=kendall_mat,
            condition_number=cond_num,
            pca_explained_variance_ratio=pca_var,
            macro_micro_proxy=macro_micro,
        )

    def _generate_recommendations(self, basic, dist, vol, cyc, ext, ors, corr) -> list[str]:
        recs = []
        if not dist.is_normal_jb and not dist.is_normal_shapiro:
            recs.append(
                "התפלגות אינה נורמלית – מומלץ להשתמש בפונקציות הפסד עמידות (Huber / quantile) או טרנספורמציה."
            )
        if abs(basic.skewness) > 1.0:
            recs.append(
                f"skewness גבוה ({basic.skewness:.2f}) – שקול log / Box-Cox או מודלים אסימטריים."
            )
        if basic.kurtosis > 3.0:
            recs.append(
                f"kurtosis גבוה ({basic.kurtosis:.2f}) – זנבות כבדים; EVT או Student-t עשויים להתאים."
            )
        if vol.arch_lm_pvalue is not None and vol.arch_lm_pvalue < 0.05:
            recs.append(
                "קיימת heteroskedasticity (ARCH) – GARCH או volatility clustering layers מומלצים."
            )
        if vol.historical_vol > 0.4:
            recs.append("תנודתיות היסטורית גבוהה – הפעל מודול Extreme Event מוקדם.")
        if cyc.dominant_period is not None and 5 < cyc.dominant_period < basic.n_obs * 2:
            recs.append(
                f"זוהה מחזור דומיננטי ~{cyc.dominant_period:.1f} – שקול positional encoding מחזורי / Fourier features."
            )
        if cyc.stl_seasonal_strength is not None and cyc.stl_seasonal_strength > 0.4:
            recs.append("עונתיות משמעותית – STL / seasonal decomposition לפני הליבה הנוירונית.")
        if ext.max_drawdown < -0.25:
            recs.append(
                f"Max Drawdown חריף ({ext.max_drawdown:.1%}) – הפעל Extreme Handler + Leverage constraints."
            )
        if ext.n_outliers_iqr > basic.n_obs * 0.05:
            recs.append("שיעור חריגים גבוה – robust scaling או winsorization.")
        if ors.order_importance_proxy > 0.6:
            recs.append(
                "סדר חשוב (order_importance גבוה) – העדף מסלולים שמורים סדר (P(n,k) / n^k)."
            )
        if ors.repetition_score > 0.4:
            recs.append("חזרות משמעותיות – מבנים עם חזרות (n^k או C(n+k-1,k)) עדיפים.")
        if ors.unique_ratio > 0.95 and ors.order_importance_proxy < 0.3:
            recs.append("מעט חזרות + סדר פחות חשוב – C(n,k) עשוי להתאים.")
        if corr.condition_number is not None and corr.condition_number > 30:
            recs.append("מטריצת הקורלציה ill-conditioned – PCA / ridge / feature selection.")
        if not recs:
            recs.append("הנתונים נראים סטנדרטיים יחסית – ניתן להמשיך עם קונפיגורציית בסיס.")
        return recs


def diagnose_series(
    data: np.ndarray | pd.Series | pd.DataFrame | list[float], **kwargs
) -> CharacterizationResult:
    layer = DiagnosisLayer(
        **{k: v for k, v in kwargs.items() if k in DiagnosisLayer.__init__.__code__.co_varnames}
    )
    return layer.diagnose(
        data,
        **{
            k: v for k, v in kwargs.items() if k not in DiagnosisLayer.__init__.__code__.co_varnames
        },
    )


if __name__ == "__main__":
    print("=" * 70)
    print("CARN-X Diagnosis Layer – Self Test")
    print("=" * 70)
    rng = np.random.default_rng(42)
    t = np.arange(300)
    series = (
        0.02 * t
        + 3 * np.sin(2 * np.pi * t / 20)
        + 1.5 * np.sin(2 * np.pi * t / 7)
        + rng.normal(0, 1.0, size=300)
    )
    series[150:160] += 8
    layer = DiagnosisLayer(rolling_window=15, acf_nlags=30)
    result = layer.diagnose(series)
    print(result)
    print("\n--- Recommendations ---")
    for r in result.recommendations:
        print(f" • {r}")
    print("\n" + "=" * 70)
    print("Diagnosis Layer ready.")
    print("=" * 70)
