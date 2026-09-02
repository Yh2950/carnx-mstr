"""
CARN-X  –  Combinatorial Gate  (Layer 2)  ·  Research-Grade
=============================================================
Advanced combinatorial filtering & routing engine.

Features:
- Differentiable soft routing (Concrete / temperature softmax)
- Dirichlet Bayesian posterior over the 4 classical cases
- Information-theoretic diagnostics (entropy, KL, MI proxy)
- Symbolic combinatorial expressions (SymPy)
- Dynamic routing graph (NetworkX)
- Attention-style continuous gates for every strategic module
- Precise complexity estimation + approximation strategy
- Formal integration of the strategic puzzles as operators
- Multi-resolution signal extraction from DiagnosisLayer
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import sympy as sp
from scipy.special import gammaln, softmax
from scipy.stats import beta as beta_dist
from scipy.stats import entropy as scipy_entropy

from diagnosis_layer import CharacterizationResult

warnings.filterwarnings("ignore")


class CombinatorialCase(Enum):
    ORDER_WITH_REP = auto()
    ORDER_NO_REP = auto()
    NO_ORDER_WITH_REP = auto()
    NO_ORDER_NO_REP = auto()


CASE_LABELS = {
    CombinatorialCase.ORDER_WITH_REP: "n^k (Ordered + Repetition)",
    CombinatorialCase.ORDER_NO_REP: "P(n,k) (Ordered + Injection)",
    CombinatorialCase.NO_ORDER_WITH_REP: "C(n+k-1,k) (Multiset)",
    CombinatorialCase.NO_ORDER_NO_REP: "C(n,k) (Combination)",
}

_n, _k = sp.symbols("n k", positive=True, integer=True)
SYMBOLIC_FORMS = {
    CombinatorialCase.ORDER_WITH_REP: _n**_k,
    CombinatorialCase.ORDER_NO_REP: sp.factorial(_n) / sp.factorial(_n - _k),
    CombinatorialCase.NO_ORDER_WITH_REP: sp.binomial(_n + _k - 1, _k),
    CombinatorialCase.NO_ORDER_NO_REP: sp.binomial(_n, _k),
}


@dataclass
class BayesianCasePosterior:
    alpha: np.ndarray
    mean_probs: np.ndarray
    mode_probs: np.ndarray
    variance: np.ndarray
    credible_intervals_90: list[tuple[float, float]]
    map_case: CombinatorialCase
    posterior_entropy: float
    evidence_strength: float


@dataclass
class InformationMetrics:
    decision_entropy: float
    decision_confidence: float
    kl_to_uniform: float
    mutual_info_proxy: float
    spectral_complexity: float


@dataclass
class CombinatorialComplexity:
    n_hat: float
    k_hat: float
    log_space_size: float
    log_space_std: float
    exact_feasible: bool
    recommended_approximation: str
    sampling_budget_hint: int
    symbolic_expression: str
    asymptotic_class: str


@dataclass
class AttentionGates:
    fibonacci_paths: float
    euclid_gcd_cycles: float
    balance_scale_search: float
    egg_drop_sequential: float
    poor_pigs_parallel_info: float
    monte_carlo_uncertainty: float
    graph_algorithms: float
    neural_ode_flow: float
    fourier_spectral: float
    extreme_event_handler: float
    game_theoretic_nash: float
    garch_volatility: float
    wavelet_multiresolution: float
    change_point_adaptive: float
    leverage_optimizer: float


@dataclass
class RoutingGraph:
    nodes: list[str]
    edges: list[tuple[str, str, float]]
    critical_path: list[str]
    modularity_hint: float


@dataclass
class LossSchedule:
    w_task: float
    w_combinatorial: float
    w_structure: float
    w_extreme: float
    w_leverage: float
    w_consistency: float
    w_information: float
    annealing_start: float
    annealing_end: float
    risk_aversion: float
    exploration_temperature: float


@dataclass
class AdvancedRoutingMap:
    dominant_case: CombinatorialCase
    soft_probs: np.ndarray
    bayesian: BayesianCasePosterior
    info_metrics: InformationMetrics
    complexity: CombinatorialComplexity
    preserve_order_strength: float
    allow_repetition_strength: float
    symmetry_preference: float
    injection_strength: float
    multiset_strength: float
    positional_encoding_strength: float
    embed_dim: int
    num_heads: int
    num_pathways: int
    depth: int
    dropout_hint: float
    activation_hint: str
    gates: AttentionGates
    loss: LossSchedule
    routing_graph: RoutingGraph
    summary: list[str]
    warnings: list[str]
    neuro_symbolic_hints: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GateResult:
    routing: AdvancedRoutingMap
    raw_signals: dict[str, float]
    debug: dict[str, Any] = field(default_factory=dict)


class CombinatorialGate:
    def __init__(
        self,
        temperature: float = 0.55,
        prior_strength: float = 1.5,
        random_state: int = 42,
    ):
        self.temperature = float(temperature)
        self.prior_strength = float(prior_strength)
        self.rng = np.random.default_rng(random_state)

    def route(self, diagnosis: CharacterizationResult) -> GateResult:
        signals = self._extract_rich_signals(diagnosis)
        logits = self._compute_advanced_logits(signals)
        soft_probs = softmax(logits / max(self.temperature, 1e-4))

        bayesian = self._bayesian_posterior(soft_probs, signals, diagnosis)
        info = self._information_metrics(soft_probs, signals)
        complexity = self._analyze_complexity(diagnosis, soft_probs, signals)
        gates = self._compute_attention_gates(signals, soft_probs)
        loss = self._build_loss_schedule(signals, soft_probs, gates)
        rgraph = self._build_routing_graph(soft_probs, gates)

        dominant = list(CombinatorialCase)[int(np.argmax(soft_probs))]
        preserve_order = float(soft_probs[0] + soft_probs[1])
        allow_rep = float(soft_probs[0] + soft_probs[2])
        symmetry = float(soft_probs[2] + soft_probs[3])
        injection = float(soft_probs[1])
        multiset = float(soft_probs[0] + soft_probs[2])
        positional = preserve_order

        embed_dim, num_heads, num_pathways, depth, dropout, act = self._recommend_capacity(
            complexity, signals, soft_probs, diagnosis
        )

        summary, warnings, ns_hints = self._generate_explanations(
            dominant, soft_probs, bayesian, info, complexity, gates, signals
        )

        routing = AdvancedRoutingMap(
            dominant_case=dominant,
            soft_probs=soft_probs,
            bayesian=bayesian,
            info_metrics=info,
            complexity=complexity,
            preserve_order_strength=preserve_order,
            allow_repetition_strength=allow_rep,
            symmetry_preference=symmetry,
            injection_strength=injection,
            multiset_strength=multiset,
            positional_encoding_strength=positional,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_pathways=num_pathways,
            depth=depth,
            dropout_hint=dropout,
            activation_hint=act,
            gates=gates,
            loss=loss,
            routing_graph=rgraph,
            summary=summary,
            warnings=warnings,
            neuro_symbolic_hints=ns_hints,
        )
        return GateResult(routing=routing, raw_signals=signals, debug={"logits": logits.tolist()})

    def _extract_rich_signals(self, d: CharacterizationResult) -> dict[str, float]:
        ors = d.order_rep_sym
        cyc = d.cyclicity
        ext = d.extremes
        vol = d.volatility
        dist = d.distribution
        basic = d.basic
        n = max(basic.n_obs, 1)
        s: dict[str, float] = {}

        s["order_proxy"] = float(np.clip(ors.order_importance_proxy, 0, 1))
        s["monotonic"] = 1.0 if ors.is_monotonic else 0.0
        s["lag1_ac"] = float(np.clip(abs(ors.autocorrelation_lag1), 0, 1))
        s["trend_strength"] = float(
            np.clip(abs(ors.trend_slope) * n / (basic.std + 1e-9) / 4.0, 0, 1)
        )
        s["runs_regularity"] = float(np.clip(1.0 - (ors.runs_test_pvalue or 1.0), 0, 1))

        s["repetition"] = float(np.clip(ors.repetition_score, 0, 1))
        s["unique_ratio"] = float(np.clip(ors.unique_ratio, 0, 1))
        s["entropy_norm"] = float(np.clip(ors.entropy_normalized, 0, 1))
        s["diversity"] = float(np.clip(ors.unique_ratio * ors.entropy_normalized, 0, 1))

        s["symmetry"] = float(np.clip(ors.symmetry_score, 0, 1))
        s["skew_abs"] = float(np.clip(abs(basic.skewness) / 2.5, 0, 1))
        s["kurtosis_excess"] = float(np.clip(basic.kurtosis / 6.0, 0, 1))
        s["heavy_tails"] = 1.0 if basic.kurtosis > 1.5 else 0.0
        s["non_normal"] = 0.0 if dist.is_normal_jb else 1.0

        s["dd_severity"] = float(np.clip(-ext.max_drawdown / 0.45, 0, 1))
        s["outlier_frac"] = float(np.clip(ext.n_outliers_iqr / n / 0.08, 0, 1))
        s["change_point_density"] = float(np.clip(len(ext.change_points) / max(n / 30, 1), 0, 1))
        s["peak_density"] = float(np.clip(ext.peak_count / max(n / 15, 1), 0, 1))

        s["vol_level"] = float(np.clip(vol.historical_vol / 0.75, 0, 1))
        s["arch"] = 1.0 if (vol.arch_lm_pvalue is not None and vol.arch_lm_pvalue < 0.05) else 0.0
        s["vol_of_vol"] = float(
            np.clip(vol.rolling_vol_std / (vol.rolling_vol_mean + 1e-9) / 2, 0, 1)
        )

        s["has_cycle"] = 1.0 if (cyc.dominant_period and 4 < cyc.dominant_period < n * 0.4) else 0.0
        s["seasonal"] = float(cyc.stl_seasonal_strength or 0.0)
        s["trend_stl"] = float(cyc.stl_trend_strength or 0.0)
        s["spectral_ent"] = float(np.clip(cyc.spectral_entropy / 5.5, 0, 1))
        s["wavelet_concentration"] = 0.0
        if cyc.wavelet_energy:
            vals = np.array(list(cyc.wavelet_energy.values()))
            s["wavelet_concentration"] = float(
                1.0 - scipy_entropy(vals + 1e-12) / np.log(len(vals) + 1e-12)
            )

        s["log_n"] = float(np.log1p(n))
        s["n_scale"] = float(np.clip(n / 800, 0, 1))
        s["order_x_cycle"] = s["order_proxy"] * s["has_cycle"]
        s["rep_x_heavy"] = s["repetition"] * s["heavy_tails"]
        s["extreme_x_vol"] = s["dd_severity"] * s["vol_level"]
        s["symmetry_x_diversity"] = s["symmetry"] * s["diversity"]
        return s

    def _compute_advanced_logits(self, s: dict[str, float]) -> np.ndarray:
        order_axis = (
            1.4 * s["order_proxy"]
            + 0.9 * s["lag1_ac"]
            + 0.7 * s["trend_strength"]
            + 0.6 * s["monotonic"]
            + 0.5 * s["runs_regularity"]
            + 0.8 * s["order_x_cycle"]
        )
        non_order_axis = (
            1.2 * (1.0 - s["order_proxy"])
            + 1.0 * s["symmetry"]
            + 0.7 * s["diversity"]
            + 0.5 * s["symmetry_x_diversity"]
        )
        rep_axis = (
            1.3 * s["repetition"]
            + 0.9 * (1.0 - s["unique_ratio"])
            + 0.6 * (1.0 - s["entropy_norm"])
            + 0.5 * s["has_cycle"]
            + 0.7 * s["rep_x_heavy"]
        )
        uniq_axis = 1.2 * s["unique_ratio"] + 0.9 * s["entropy_norm"] + 0.6 * s["diversity"]
        flex = (
            0.9 * s["heavy_tails"]
            + 0.8 * s["dd_severity"]
            + 0.7 * s["outlier_frac"]
            + 0.6 * s["vol_level"]
            + 0.5 * s["arch"]
            + 1.0 * s["extreme_x_vol"]
        )
        injection_pref = (
            1.1 * s["lag1_ac"] * s["monotonic"]
            + 0.8 * s["trend_strength"] * (1.0 - s["repetition"])
            + 0.6 * s["order_proxy"] * s["unique_ratio"]
        )

        logits = np.zeros(4)
        logits[0] = (
            0.55 * order_axis
            + 0.50 * rep_axis
            + 0.35 * flex
            - 0.25 * non_order_axis
            - 0.15 * injection_pref
        )
        logits[1] = (
            0.60 * order_axis
            + 0.45 * uniq_axis
            + 0.55 * injection_pref
            - 0.20 * flex
            - 0.15 * non_order_axis
        )
        logits[2] = (
            0.50 * non_order_axis
            + 0.55 * rep_axis
            + 0.30 * flex
            + 0.25 * s["symmetry"]
            - 0.20 * order_axis
        )
        logits[3] = (
            0.55 * non_order_axis
            + 0.50 * uniq_axis
            + 0.40 * s["symmetry"]
            + 0.25 * s["diversity"]
            - 0.25 * flex
            - 0.15 * order_axis
        )
        logits += 0.15 * s["n_scale"]
        return logits

    def _bayesian_posterior(
        self, soft_probs: np.ndarray, signals: dict[str, float], d: CharacterizationResult
    ) -> BayesianCasePosterior:
        n = d.basic.n_obs
        ess = max(4.0, np.sqrt(n) * (0.6 + 0.4 * signals["n_scale"]))
        ess *= 1.0 + 0.4 * signals["dd_severity"] + 0.3 * signals["order_proxy"]
        alpha0 = np.full(4, self.prior_strength)
        alpha = alpha0 + soft_probs * ess
        mean = alpha / alpha.sum()
        mode = np.maximum(alpha - 1, 0)
        mode = mode / mode.sum() if mode.sum() > 0 else mean.copy()
        a0 = alpha.sum()
        var = alpha * (a0 - alpha) / (a0**2 * (a0 + 1))
        intervals = []
        for i in range(4):
            a_i, b_i = alpha[i], a0 - alpha[i]
            lo = float(beta_dist.ppf(0.05, a_i, b_i))
            hi = float(beta_dist.ppf(0.95, a_i, b_i))
            intervals.append((lo, hi))
        map_idx = int(np.argmax(mode))
        post_ent = float(scipy_entropy(mean + 1e-12))
        prior_mean = alpha0 / alpha0.sum()
        evidence = float(np.sum(mean * np.log((mean + 1e-12) / (prior_mean + 1e-12))))
        return BayesianCasePosterior(
            alpha=alpha,
            mean_probs=mean,
            mode_probs=mode,
            variance=var,
            credible_intervals_90=intervals,
            map_case=list(CombinatorialCase)[map_idx],
            posterior_entropy=post_ent,
            evidence_strength=evidence,
        )

    def _information_metrics(
        self, soft_probs: np.ndarray, signals: dict[str, float]
    ) -> InformationMetrics:
        ent = float(scipy_entropy(soft_probs + 1e-12))
        conf = float(1.0 - ent / np.log(4))
        kl_uniform = float(np.sum(soft_probs * np.log((soft_probs + 1e-12) * 4)))
        p_order = soft_probs[0] + soft_probs[1]
        p_rep = soft_probs[0] + soft_probs[2]
        h_order = scipy_entropy([p_order, 1 - p_order])
        h_rep = scipy_entropy([p_rep, 1 - p_rep])
        h_joint = scipy_entropy(soft_probs + 1e-12)
        mi = float(max(0.0, h_order + h_rep - h_joint))
        return InformationMetrics(
            decision_entropy=ent,
            decision_confidence=conf,
            kl_to_uniform=kl_uniform,
            mutual_info_proxy=mi,
            spectral_complexity=float(signals.get("spectral_ent", 0.5)),
        )

    def _log_size(self, case: CombinatorialCase, n: float, k: float) -> float:
        n = max(n, 2.0)
        k = float(np.clip(k, 1.0, n))
        if case == CombinatorialCase.ORDER_WITH_REP:
            return k * np.log(n)
        if case == CombinatorialCase.ORDER_NO_REP:
            return float(gammaln(n + 1) - gammaln(n - k + 1))
        if case == CombinatorialCase.NO_ORDER_WITH_REP:
            return float(gammaln(n + k) - gammaln(k + 1) - gammaln(n))
        return float(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1))

    def _analyze_complexity(
        self, d: CharacterizationResult, soft_probs: np.ndarray, signals: dict[str, float]
    ) -> CombinatorialComplexity:
        n_obs = d.basic.n_obs
        n_hat = float(np.clip(n_obs * (0.35 + 0.65 * signals["unique_ratio"]), 8, 20000))
        k_ratio = 0.12 + 0.28 * signals["order_proxy"] + 0.18 * signals["has_cycle"]
        k_hat = float(np.clip(n_hat * k_ratio, 2, n_hat * 0.85))
        log_sizes = np.array([self._log_size(c, n_hat, k_hat) for c in CombinatorialCase])
        log_space = float(np.dot(soft_probs, log_sizes))
        log_space_std = float(np.sqrt(np.dot(soft_probs, (log_sizes - log_space) ** 2)))
        exact_feasible = log_space < 18
        if exact_feasible:
            approx, budget = "exact", 0
        elif log_space < 28:
            approx, budget = "gumbel", 4096
        elif log_space < 40:
            approx, budget = "importance", 16384
        else:
            approx, budget = "mcmc_or_greedy", 65536
        dom = list(CombinatorialCase)[int(np.argmax(soft_probs))]
        sym = str(SYMBOLIC_FORMS[dom])
        asym = {
            CombinatorialCase.ORDER_WITH_REP: "O(n^k)",
            CombinatorialCase.ORDER_NO_REP: "O(n!/(n-k)!)",
            CombinatorialCase.NO_ORDER_WITH_REP: "O((n+k)^k / k!)",
            CombinatorialCase.NO_ORDER_NO_REP: "O(n choose k)",
        }[dom]
        return CombinatorialComplexity(
            n_hat=n_hat,
            k_hat=k_hat,
            log_space_size=log_space,
            log_space_std=log_space_std,
            exact_feasible=exact_feasible,
            recommended_approximation=approx,
            sampling_budget_hint=budget,
            symbolic_expression=sym,
            asymptotic_class=asym,
        )

    def _gate(self, x: float) -> float:
        return float(np.clip(x, 0.0, 1.0))

    def _conf(self, p: np.ndarray) -> float:
        return float(1.0 - scipy_entropy(p + 1e-12) / np.log(len(p)))

    def _compute_attention_gates(
        self, s: dict[str, float], soft_probs: np.ndarray
    ) -> AttentionGates:
        conf = self._conf(soft_probs)
        return AttentionGates(
            fibonacci_paths=self._gate(
                0.40 * s["order_proxy"]
                + 0.25 * s["has_cycle"]
                + 0.20 * soft_probs[0]
                + 0.15 * s["lag1_ac"]
            ),
            euclid_gcd_cycles=self._gate(
                0.55 * s["has_cycle"] + 0.25 * s["seasonal"] + 0.20 * (1.0 - s["spectral_ent"])
            ),
            balance_scale_search=self._gate(
                0.45 * s["outlier_frac"]
                + 0.30 * s["dd_severity"]
                + 0.25 * s["change_point_density"]
            ),
            egg_drop_sequential=self._gate(
                0.35 * s["order_proxy"] + 0.30 * s["vol_level"] + 0.35 * s["dd_severity"]
            ),
            poor_pigs_parallel_info=self._gate(
                0.35 * s["n_scale"] + 0.30 * (1.0 - conf) + 0.35 * s["outlier_frac"]
            ),
            monte_carlo_uncertainty=self._gate(
                0.30 * s["heavy_tails"]
                + 0.25 * s["non_normal"]
                + 0.25 * s["vol_level"]
                + 0.20 * (1.0 - conf)
            ),
            graph_algorithms=self._gate(
                0.40 * s["order_proxy"]
                + 0.30 * s["has_cycle"]
                + 0.30 * (soft_probs[0] + soft_probs[1])
            ),
            neural_ode_flow=self._gate(
                0.40 * s["trend_strength"]
                + 0.30 * s["lag1_ac"]
                + 0.30 * (1.0 - s["change_point_density"])
            ),
            fourier_spectral=self._gate(
                0.50 * s["has_cycle"] + 0.30 * s["seasonal"] + 0.20 * (1.0 - s["spectral_ent"])
            ),
            extreme_event_handler=self._gate(
                0.40 * s["dd_severity"]
                + 0.25 * s["vol_level"]
                + 0.20 * s["outlier_frac"]
                + 0.15 * s["arch"]
            ),
            game_theoretic_nash=self._gate(
                0.30 * s["vol_level"]
                + 0.25 * (1.0 - conf)
                + 0.25 * s["heavy_tails"]
                + 0.20 * s["n_scale"]
            ),
            garch_volatility=self._gate(
                0.55 * s["arch"] + 0.30 * s["vol_level"] + 0.15 * s["vol_of_vol"]
            ),
            wavelet_multiresolution=self._gate(
                0.45 * s["wavelet_concentration"]
                + 0.30 * s["has_cycle"]
                + 0.25 * s["change_point_density"]
            ),
            change_point_adaptive=self._gate(
                0.60 * s["change_point_density"] + 0.25 * s["dd_severity"] + 0.15 * s["vol_of_vol"]
            ),
            leverage_optimizer=self._gate(
                0.50 * s["dd_severity"] + 0.30 * s["extreme_x_vol"] + 0.20 * s["vol_level"]
            ),
        )

    def _build_loss_schedule(
        self, s: dict[str, float], soft_probs: np.ndarray, gates: AttentionGates
    ) -> LossSchedule:
        conf = self._conf(soft_probs)
        w_task = 1.0
        w_comb = 0.12 + 0.28 * (1.0 - conf)
        w_struct = 0.08 + 0.22 * conf
        w_ext = 0.04 + 0.40 * s["dd_severity"] + 0.18 * s["vol_level"]
        w_lev = 0.06 + 0.30 * gates.leverage_optimizer
        w_cons = 0.05 + 0.10 * (1.0 if soft_probs.max() < 0.55 else 0.0)
        w_info = 0.07 + 0.12 * (1.0 - conf)
        total = w_task + w_comb + w_struct + w_ext + w_lev + w_cons + w_info
        ws = [x / total for x in (w_task, w_comb, w_struct, w_ext, w_lev, w_cons, w_info)]
        risk = float(
            np.clip(
                0.35 * s["dd_severity"]
                + 0.30 * s["vol_level"]
                + 0.20 * s["heavy_tails"]
                + 0.15 * s["arch"],
                0,
                1,
            )
        )
        explor = float(np.clip(0.4 + 0.5 * (1.0 - conf) + 0.2 * s["n_scale"], 0.3, 1.4))
        return LossSchedule(
            w_task=ws[0],
            w_combinatorial=ws[1],
            w_structure=ws[2],
            w_extreme=ws[3],
            w_leverage=ws[4],
            w_consistency=ws[5],
            w_information=ws[6],
            annealing_start=1.0,
            annealing_end=0.3,
            risk_aversion=risk,
            exploration_temperature=explor,
        )

    def _recommend_capacity(
        self,
        complexity: CombinatorialComplexity,
        s: dict[str, float],
        soft_probs: np.ndarray,
        d: CharacterizationResult,
    ) -> tuple[int, int, int, int, float, str]:
        log_omega = complexity.log_space_size
        n = d.basic.n_obs
        dim = 48 + 20 * np.log2(1 + log_omega) + 12 * np.log2(1 + n)
        dim = float(
            np.nan_to_num(dim, nan=192.0)
        )  # degenerate (e.g. zero-variance) input can NaN this
        dim = int(np.clip(dim, 64, 768))
        dim = (dim // 16) * 16
        heads = max(4, min(16, dim // 48))
        conf = self._conf(soft_probs)
        pathways = 3 if conf < 0.55 else (2 if conf < 0.72 else 1)
        depth = 4
        if s["n_scale"] > 0.4:
            depth = 6
        if s["n_scale"] > 0.7 or log_omega > 25:
            depth = 8
        if s["dd_severity"] > 0.6:
            depth = min(depth + 1, 10)
        dropout = float(np.clip(0.08 + 0.12 * (1.0 - conf) + 0.05 * s["vol_level"], 0.05, 0.35))
        act = (
            "mish"
            if (s["heavy_tails"] > 0.5 or s["dd_severity"] > 0.5)
            else ("swish" if s["has_cycle"] > 0.5 else "gelu")
        )
        return dim, heads, pathways, depth, dropout, act

    def _build_routing_graph(self, soft_probs: np.ndarray, gates: AttentionGates) -> RoutingGraph:
        nodes = [
            "input",
            "diagnosis",
            "comb_gate",
            "fibonacci",
            "euclid",
            "balance",
            "eggdrop",
            "pigs",
            "fourier",
            "wavelet",
            "garch",
            "extreme",
            "leverage",
            "graph_alg",
            "neural_ode",
            "mc",
            "core_transformer",
            "output",
        ]
        edges = [
            ("input", "diagnosis", 1.0),
            ("diagnosis", "comb_gate", 1.0),
            ("comb_gate", "core_transformer", 0.9),
            ("core_transformer", "output", 1.0),
        ]
        module_map = {
            "fibonacci": gates.fibonacci_paths,
            "euclid": gates.euclid_gcd_cycles,
            "balance": gates.balance_scale_search,
            "eggdrop": gates.egg_drop_sequential,
            "pigs": gates.poor_pigs_parallel_info,
            "fourier": gates.fourier_spectral,
            "wavelet": gates.wavelet_multiresolution,
            "garch": gates.garch_volatility,
            "extreme": gates.extreme_event_handler,
            "leverage": gates.leverage_optimizer,
            "graph_alg": gates.graph_algorithms,
            "neural_ode": gates.neural_ode_flow,
            "mc": gates.monte_carlo_uncertainty,
        }
        for mod, w in module_map.items():
            if w > 0.15:
                edges.append(("comb_gate", mod, float(w)))
                edges.append((mod, "core_transformer", float(w * 0.85)))
        if gates.extreme_event_handler > 0.4 and gates.leverage_optimizer > 0.4:
            edges.append(("extreme", "leverage", 0.7))
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_weighted_edges_from(edges)
        try:
            critical = nx.dag_longest_path(G, weight="weight")
        except Exception:
            critical = ["input", "diagnosis", "comb_gate", "core_transformer", "output"]
        mod_hint = float(np.mean([w for _, _, w in edges if w < 0.99]))
        return RoutingGraph(
            nodes=nodes, edges=edges, critical_path=critical, modularity_hint=mod_hint
        )

    def _generate_explanations(self, dominant, soft_probs, bayes, info, complexity, gates, signals):
        summary = [
            f"Dominant case: {CASE_LABELS[dominant]}  (soft mass={soft_probs.max():.1%})",
            f"Bayesian MAP: {CASE_LABELS[bayes.map_case]}  | evidence={bayes.evidence_strength:.3f}",
            f"Decision confidence={info.decision_confidence:.1%}  | H={info.decision_entropy:.3f}  | MI(order;rep)≈{info.mutual_info_proxy:.3f}",
            f"Complexity: n̂={complexity.n_hat:.1f}, k̂={complexity.k_hat:.1f}, log|Ω|={complexity.log_space_size:.2f}±{complexity.log_space_std:.2f}",
            f"Approximation strategy → {complexity.recommended_approximation}  (budget≈{complexity.sampling_budget_hint})",
            f"Symbolic: {complexity.symbolic_expression}   [{complexity.asymptotic_class}]",
        ]
        if gates.extreme_event_handler > 0.45:
            summary.append(
                f"⚠ Extreme-Event gate HIGH ({gates.extreme_event_handler:.2f}) – crash regime active"
            )
        if gates.leverage_optimizer > 0.4:
            summary.append(f"Leverage optimizer engaged ({gates.leverage_optimizer:.2f})")
        if gates.fourier_spectral > 0.5:
            summary.append("Fourier/Spectral features strongly indicated")
        if gates.fibonacci_paths > 0.5:
            summary.append("Fibonacci path-counting relevant (sequential recurrence)")
        warnings = []
        if info.decision_confidence < 0.40:
            warnings.append(
                "Low decision confidence → multi-pathway + higher exploration temperature"
            )
        if complexity.log_space_size > 35:
            warnings.append(
                "Huge combinatorial space → mandatory approximation (MCMC / Gumbel / greedy)"
            )
        if signals["dd_severity"] > 0.55:
            warnings.append("Severe drawdown → elevated extreme & leverage loss weights")
        ns_hints = [
            f"Neuro-symbolic: keep symbolic form '{complexity.symbolic_expression}' as regularizer target",
            "Prefer Gumbel-Softmax for discrete pathway selection during training",
            "Inject Fibonacci recurrence as inductive bias when gate_fib high",
            "Balance-scale / egg-drop gates suggest adaptive computation time / early-exit",
        ]
        return summary, warnings, ns_hints


def run_full_pipeline(data, **gate_kwargs):
    from diagnosis_layer import DiagnosisLayer

    diagnosis = DiagnosisLayer().diagnose(data)
    gate = CombinatorialGate(**gate_kwargs)
    return diagnosis, gate.route(diagnosis)


if __name__ == "__main__":
    print("=" * 78)
    print("CARN-X  ·  Research-Grade Combinatorial Gate  ·  Self-Test")
    print("=" * 78)

    rng = np.random.default_rng(42)
    t = np.arange(320)
    series = (
        0.012 * t
        + 3.1 * np.sin(2 * np.pi * t / 17)
        + 1.4 * np.sin(2 * np.pi * t / 5)
        + rng.normal(0, 0.85, size=320)
    )
    series[140:160] -= 9.0

    from diagnosis_layer import DiagnosisLayer

    diag = DiagnosisLayer().diagnose(series)
    gate = CombinatorialGate(temperature=0.50, prior_strength=1.8)
    result = gate.route(diag)
    r = result.routing

    print("\n▸ Soft probabilities")
    for case, p in zip(CombinatorialCase, r.soft_probs):
        print(f"   {CASE_LABELS[case]:<42} {p:.1%}")

    print(f"\n▸ Dominant: {CASE_LABELS[r.dominant_case]}")
    print(
        f"▸ Bayesian MAP: {CASE_LABELS[r.bayesian.map_case]}  (evidence={r.bayesian.evidence_strength:.3f})"
    )
    print(
        f"▸ Confidence: {r.info_metrics.decision_confidence:.1%}   MI(order;rep)≈{r.info_metrics.mutual_info_proxy:.3f}"
    )

    print(
        f"\n▸ Complexity: log|Ω|={r.complexity.log_space_size:.2f} ± {r.complexity.log_space_std:.2f}"
    )
    print(
        f"  Approximation: {r.complexity.recommended_approximation}  |  {r.complexity.asymptotic_class}"
    )
    print(f"  Symbolic: {r.complexity.symbolic_expression}")

    print(
        f"\n▸ Capacity: dim={r.embed_dim}, heads={r.num_heads}, pathways={r.num_pathways}, depth={r.depth}"
    )
    print(f"  activation={r.activation_hint}, dropout={r.dropout_hint:.2f}")

    print("\n▸ Key Attention Gates")
    g = r.gates
    print(
        f"  extreme={g.extreme_event_handler:.3f}  leverage={g.leverage_optimizer:.3f}  "
        f"fourier={g.fourier_spectral:.3f}  fib={g.fibonacci_paths:.3f}"
    )
    print(
        f"  garch={g.garch_volatility:.3f}  eggdrop={g.egg_drop_sequential:.3f}  "
        f"balance={g.balance_scale_search:.3f}  euclid={g.euclid_gcd_cycles:.3f}"
    )

    print("\n▸ Loss weights")
    ls = r.loss
    print(
        f"  task={ls.w_task:.3f} comb={ls.w_combinatorial:.3f} struct={ls.w_structure:.3f} "
        f"extreme={ls.w_extreme:.3f} lev={ls.w_leverage:.3f} info={ls.w_information:.3f}"
    )
    print(f"  risk_aversion={ls.risk_aversion:.3f}  explor_temp={ls.exploration_temperature:.3f}")

    print("\n▸ Routing critical path")
    print("  " + " → ".join(r.routing_graph.critical_path))

    print("\n▸ Summary")
    for line in r.summary:
        print(f"  • {line}")
    if r.warnings:
        print("\n▸ Warnings")
        for w in r.warnings:
            print(f"  ⚠ {w}")

    print("\n" + "=" * 78)
    print("Research-Grade Combinatorial Gate ready.")
    print("=" * 78)
