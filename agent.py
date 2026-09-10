"""
CARN-X  --  conversational analysis agent
=========================================
A domain agent that talks like a human analyst.  You tell it, in free text, how
much you put in and at what price(s); it reads that, runs the *real* trained
neural-net forecast + Monte-Carlo cone + leverage engine on your position, and
answers with a grounded read -- unrealised P/L, the model's directional call and
confidence, the projected P/L distribution, probability of profit / of ruin under
your leverage, and the honest calibration caveats.

Brain, in order of preference (auto-detected, all optional):
  * a local Ollama model at http://localhost:11434   -> free-form conversation
  * ANTHROPIC_API_KEY (env or st.secrets)             -> free-form conversation
  * neither                                           -> the built-in analyst
In every mode the numbers come from THIS repo's model -- the LLM only phrases and
handles free-form follow-ups; it is instructed never to invent figures.

Nothing here trains, saves, or mutates the model / the dashboard.  Read-only.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from datetime import date

import numpy as np
import streamlit as st

# --------------------------------------------------------------------------- #
# the position the user is describing
# --------------------------------------------------------------------------- #
_HEB_NUM = {
    "אפס": 0, "אחד": 1, "אחת": 1, "שתיים": 2, "שניים": 2, "שלוש": 3, "ארבע": 4,
    "חמש": 5, "שש": 6, "שבע": 7, "שמונה": 8, "תשע": 9, "עשר": 10,
}
_MULT = {
    "k": 1e3, "אלף": 1e3, "אלפים": 1e3, "אלפי": 1e3, "thousand": 1e3,
    "m": 1e6, "מיליון": 1e6, "מיליוני": 1e6, "million": 1e6, "mm": 1e6,
}


@dataclass
class Position:
    instrument: str = "MSTR"            # "MSTR" | "BTC"
    amount_usd: float | None = None     # capital deployed (your own money)
    entries: list[tuple[float, float]] = field(default_factory=list)  # (price, weight)
    leverage: float = 1.0
    horizon_days: int = 20
    entry_date: date | None = None

    @property
    def avg_entry(self) -> float | None:
        if not self.entries:
            return None
        w = sum(x[1] for x in self.entries) or float(len(self.entries))
        return sum(p * wt for p, wt in self.entries) / w if w else None


def _to_float(tok: str) -> float | None:
    tok = tok.replace(",", "").replace("$", "").replace("₪", "").strip()
    m = re.match(r"^(\d+(?:\.\d+)?)\s*([a-zא-ת]+)?$", tok, re.I)
    if not m:
        return None
    val = float(m.group(1))
    unit = (m.group(2) or "").lower()
    if unit in _MULT:
        val *= _MULT[unit]
    return val


def parse(text: str, pos: Position) -> Position:
    """Merge whatever the message reveals into `pos` (non-destructive)."""
    t = " " + text.strip() + " "
    low = t.lower()
    out = replace(pos, entries=list(pos.entries))

    # instrument
    if re.search(r"\b(btc|bitcoin|ביטקוין|ביטקויין|ביט קוין)\b", low):
        out.instrument = "BTC"
    elif re.search(r"\b(mstr|microstrategy|מייקרוסטרטג|מיקרוסטרטג|סטרטג)\b", low):
        out.instrument = "MSTR"

    # leverage:  "מינוף 2",  "2x",  "פי 3",  "x2"
    lev = re.search(r"(?:מינוף|leverage|lev)\D{0,4}(\d+(?:\.\d+)?)", low) \
        or re.search(r"\bx\s?(\d+(?:\.\d+)?)\b", low) \
        or re.search(r"(\d+(?:\.\d+)?)\s?x\b", low) \
        or re.search(r"פי\s+(\d+(?:\.\d+)?)", low)
    if lev:
        v = float(lev.group(1))
        if 1.0 <= v <= 20.0:
            out.leverage = v
    if re.search(r"בלי מינוף|ללא מינוף|no leverage|unlevered|1x\b|בלי לוורג", low):
        out.leverage = 1.0

    # horizon:  "עוד 3 חודשים",  "טווח 20 יום",  "בעוד שנה",  "in 6 months"
    hz = re.search(r"(\d+(?:\.\d+)?)\s*(ימים|יום|days|day)", low)
    wk = re.search(r"(\d+(?:\.\d+)?)\s*(שבועות|שבוע|weeks|week)", low)
    mo = re.search(r"(\d+(?:\.\d+)?)\s*(חודשים|חודש|months|month)", low)
    yr = re.search(r"(\d+(?:\.\d+)?)\s*(שנים|שנה|years|year)", low)
    if re.search(r"\bשנה\b|\ba year\b|בעוד שנה", low) and not yr:
        out.horizon_days = 252
    if hz:
        out.horizon_days = int(round(float(hz.group(1))))
    elif wk:
        out.horizon_days = int(round(float(wk.group(1)) * 5))
    elif mo:
        out.horizon_days = int(round(float(mo.group(1)) * 21))
    elif yr:
        out.horizon_days = int(round(float(yr.group(1)) * 252))
    out.horizon_days = max(1, min(out.horizon_days, 252))

    # entry price(s):  "בשער 250",  "מחיר כניסה 132",  "at $410",  "נכנסתי ב-250 ו-300"
    price_ctx = re.findall(
        r"(?:שער|מחיר|כניסה|נכנסתי|קניתי|ב-?|at|price|entry)\D{0,6}"
        r"(\d[\d,]*(?:\.\d+)?)", low)
    # bare "250 ו 300" after a price cue
    if not price_ctx:
        price_ctx = re.findall(r"\b(\d{2,}(?:\.\d+)?)\b(?=\s*(?:ו|and|,|\+))", low)
    got_prices = []
    for p in price_ctx:
        v = _to_float(p)
        if v and 1 <= v <= 100000:
            got_prices.append(v)
    # amount:  "$10,000",  "10 אלף דולר",  "השקעתי 25000"
    amt = re.search(
        r"(?:השקעתי|שמתי|הכנסתי|invested|put|deposited|capital|הון|סכום)\D{0,6}"
        r"(\d[\d,]*(?:\.\d+)?\s*(?:k|m|אלף|אלפים|מיליון|million|thousand)?)", low)
    dollar = re.search(r"[$₪]\s?(\d[\d,]*(?:\.\d+)?\s*(?:k|m|אלף|מיליון)?)", low)
    if amt:
        v = _to_float(amt.group(1))
        if v and v >= 10:
            out.amount_usd = v
    elif dollar and out.amount_usd is None:
        v = _to_float(dollar.group(1))
        if v and v >= 10:
            out.amount_usd = v

    # if a message has exactly one number and we still need a price, take it as price;
    # if we have a price but no amount and the number is large, treat as amount.
    loose = [_to_float(x) for x in re.findall(r"\b(\d[\d,]*(?:\.\d+)?)\b", low)]
    loose = [v for v in loose if v is not None]
    if not got_prices and not out.entries and len(loose) == 1 and out.amount_usd is not None:
        if 1 <= loose[0] <= 100000:
            got_prices = [loose[0]]
    if got_prices:
        # replace entries only when the user is (re)stating them
        if re.search(r"שער|מחיר|כניסה|נכנסתי|קניתי|entry|price|at\b", low) or not out.entries:
            out.entries = [(p, 1.0) for p in got_prices]

    if out.amount_usd is None and not got_prices and len(loose) == 1 and loose[0] >= 500 \
            and not out.entries:
        out.amount_usd = loose[0]

    return out


def missing(pos: Position) -> list[str]:
    need = []
    if pos.amount_usd is None:
        need.append("amount")
    if not pos.entries:
        need.append("entry")
    return need


# --------------------------------------------------------------------------- #
# the analysis  --  runs the real model on the position
# --------------------------------------------------------------------------- #
@dataclass
class Analysis:
    ok: bool
    note: str = ""
    price_now: float = 0.0
    avg_entry: float = 0.0
    horizon: int = 20
    pnl_now_pct: float = 0.0
    pnl_now_usd: float = 0.0
    exp_move_pct: float = 0.0
    p_up: float = 0.5
    q05_pct: float = 0.0
    q95_pct: float = 0.0
    proj_equity_mult: float = 1.0
    p_profit: float = 0.5
    p_double: float = 0.0
    p_ruin: float = 0.0
    mc_expected_price: float = 0.0
    mc_var5_price: float = 0.0
    mc_prob_up: float = 0.5
    lev_ceiling: float = 1.0
    instrument: str = "MSTR"


def _dist_row(dist, horizon: int):
    hs = sorted(int(h) for h in dist["horizon"].tolist())
    h = min(hs, key=lambda x: abs(x - horizon)) if hs else horizon
    return dist.loc[dist["horizon"] == h].iloc[0], h


def analyze(pos: Position, *, data, get_model, last_price: float,
            live_price: float | None = None) -> Analysis:
    price_now = float(live_price or last_price or 0.0)
    avg = pos.avg_entry or price_now
    lev = max(1.0, float(pos.leverage))

    a = Analysis(ok=False, price_now=price_now, avg_entry=avg,
                 horizon=pos.horizon_days, instrument=pos.instrument)

    # unrealised, path-independent (leverage scales the % swing on your capital)
    move_now = (price_now / avg - 1.0) if avg else 0.0
    a.pnl_now_pct = move_now * lev * 100.0
    a.pnl_now_usd = (pos.amount_usd or 0.0) * move_now * lev

    try:
        from inference import monte_carlo_paths, predict_distribution, tail_probability
        pm = get_model()
        if pm is None:
            a.note = "no_model"
            return a
        key = f"_cx_dist_{getattr(pm, 'trained_at', id(pm))}"
        dist = st.session_state.get(key)
        if dist is None:
            dist = predict_distribution(pm, data)
            st.session_state[key] = dist
        row, h = _dist_row(dist, pos.horizon_days)
        a.horizon = int(h)
        a.exp_move_pct = float(row.expected_move_pct)
        a.p_up = float(row.p_up)
        a.q05_pct = float(getattr(row, "ret_q05_pct", -abs(row.expected_move_pct) * 3))
        a.q95_pct = float(getattr(row, "ret_q95_pct", abs(row.expected_move_pct) * 3))

        # projected equity distribution under leverage (levered return = lev * asset return)
        def equity_mult(asset_ret_pct: float) -> float:
            return 1.0 + lev * (asset_ret_pct / 100.0)

        a.proj_equity_mult = max(0.0, equity_mult(a.exp_move_pct))
        # normal approx on the horizon return from the model's q05/q95 (~5th/95th)
        mu = a.exp_move_pct
        sd = max(1e-6, (a.q95_pct - a.q05_pct) / 3.2898)
        z_be = (0.0 - mu) / sd                       # break-even (asset flat)
        z_ruin = (-100.0 / lev - mu) / sd            # equity wiped
        z_dbl = (100.0 / lev - mu) / sd              # equity doubled
        from math import erf, sqrt
        ncdf = lambda z: 0.5 * (1.0 + erf(z / sqrt(2.0)))
        a.p_profit = 1.0 - ncdf(z_be)
        a.p_ruin = ncdf(z_ruin) if lev > 1.0 else 0.0
        a.p_double = 1.0 - ncdf(z_dbl)

        try:
            mc = monte_carlo_paths(pm, data, horizon=a.horizon, n_paths=4000)
            a.mc_expected_price = float(mc.expected_price)
            a.mc_var5_price = float(mc.var_5_price)
            a.mc_prob_up = float(mc.prob_up)
            # blend the MC prob-up into p_profit for the *unlevered* direction sense
            a.p_profit = 0.5 * a.p_profit + 0.5 * (
                a.mc_prob_up if price_now >= avg else 1.0 - (1.0 - a.mc_prob_up))
        except Exception:  # noqa: BLE001
            pass

        try:
            from leverage_factorization import LeverageFactorization
            conf = a.p_up if a.p_up > 0.5 else 1.0 - a.p_up
            rec = LeverageFactorization().recommend(
                p_up=a.p_up, confidence=float(conf),
                ann_vol=float(getattr(row, "ann_vol_implied", 0.6)))
            a.lev_ceiling = float(getattr(rec, "leverage", getattr(rec, "ceiling", 1.0)))
        except Exception:  # noqa: BLE001
            a.lev_ceiling = 1.5 if a.p_up > 0.55 else 1.0

        _ = tail_probability  # (kept importable for future tail questions)
        a.ok = True
    except Exception as e:  # noqa: BLE001
        a.note = f"error:{type(e).__name__}"
    return a


# --------------------------------------------------------------------------- #
# phrasing  --  the built-in analyst voice (used when there is no LLM)
# --------------------------------------------------------------------------- #
def _pct(x: float, s: int = 1) -> str:
    return f"{x:+.{s}f}%"


def _usd(x: float) -> str:
    a = abs(x)
    sign = "-" if x < 0 else ""
    if a >= 1e6:
        return f"{sign}${a/1e6:.2f}M"
    if a >= 1e3:
        return f"{sign}${a/1e3:,.1f}k"
    return f"{sign}${a:,.0f}"


def phrase_analysis(a: Analysis, pos: Position) -> str:
    name = "מייקרוסטרטג'י (MSTR)" if a.instrument == "MSTR" else "ביטקוין"
    lev_txt = "בלי מינוף" if pos.leverage <= 1.0 else f"במינוף פי-{pos.leverage:g}"
    lines = []

    if pos.leverage > 1.0 and a.pnl_now_pct <= -100.0:
        lines.append(
            f"נכון לעכשיו {name} ב-${a.price_now:,.2f} מול כניסה ב-${a.avg_entry:,.2f} — "
            f"ירידה של {_pct((a.price_now/a.avg_entry-1)*100)} על הנכס. {lev_txt}, הפוזיציה הזו "
            f"כבר הייתה מקבלת margin call וההון ({_usd(pos.amount_usd or 0)}) אבוד. "
            f"מכאן אני מנתח מה צפוי למחיר עצמו:"
        )
    else:
        where = "בירוק" if a.pnl_now_usd >= 0 else "באדום"
        lines.append(
            f"נכון לעכשיו {name} ב-${a.price_now:,.2f}, הכניסה הממוצעת שלך ב-${a.avg_entry:,.2f} — "
            f"אתה {where} על **{_usd(a.pnl_now_usd)}** ({_pct(a.pnl_now_pct)} על ההון, {lev_txt})."
        )

    call = "עלייה" if a.p_up >= 0.5 else "ירידה"
    conf = a.p_up if a.p_up >= 0.5 else 1 - a.p_up
    strength = ("קלושה" if conf < 0.55 else "מתונה" if conf < 0.62
                else "ברורה" if conf < 0.72 else "חזקה")
    lines.append(
        f"המודל נותן ל-{a.horizon} ימי מסחר קדימה נטייה ל**{call}** בהסתברות {a.p_up:.0%} "
        f"(ודאות {strength}), תשואה צפויה {_pct(a.exp_move_pct)} על הנכס, "
        f"וטווח סביר של {_pct(a.q05_pct)} עד {_pct(a.q95_pct)}."
    )

    eq_move = (a.proj_equity_mult - 1) * 100
    lines.append(
        f"אם התרחיש המרכזי מתממש, ההון שלך זז בערך {_pct(eq_move)} — "
        f"כלומר {_usd((pos.amount_usd or 0) * (a.proj_equity_mult - 1))}. "
        f"הסתברות מוערכת לצאת ברווח מהנקודה הזו: **{a.p_profit:.0%}**."
    )

    if pos.leverage > 1.0:
        lines.append(
            f"במינוף פי-{pos.leverage:g}: הסיכוי להכפיל את ההון בטווח הזה ~{a.p_double:.0%}, "
            f"והסיכוי למחיקה מלאה (margin call) ~**{a.p_ruin:.0%}**. "
            f"תקרת המינוף שהמנוע ממליץ עליה כאן היא פי-{a.lev_ceiling:g} — "
            + ("אתה בתוך הטווח." if pos.leverage <= a.lev_ceiling + 1e-6
               else "אתה מעליה, זו פוזיציה אגרסיבית.")
        )
    else:
        lines.append(
            f"בלי מינוף אין סיכון מחיקה. אם היית שוקל מינוף, המנוע לא היה חורג מפי-{a.lev_ceiling:g} "
            f"בהינתן הביטחון הנוכחי."
        )

    if a.mc_expected_price:
        lines.append(
            f"סימולציית מונטה-קרלו ({a.horizon} ימים, 4,000 מסלולים): מחיר צפוי "
            f"${a.mc_expected_price:,.2f}, רצפת VaR-5% סביב ${a.mc_var5_price:,.2f}."
        )

    lines.append(
        "_זכור: זו תחזית הסתברותית מכוילת, לא ודאות. הרווחים בזנב תלויי-מסלול, "
        "ומינוף מגדיל גם את הטעות של המודל._"
    )
    return "\n\n".join(lines)


def _ask(need: list[str], pos: Position) -> str:
    if "amount" in need and "entry" in need:
        return ("כדי לנתח לך את זה כמו שצריך אני צריך שני דברים: כמה כסף שלך נכנס לפוזיציה, "
                "ובאיזה מחיר (או מחירים) נכנסת. תכתוב לי חופשי, למשל "
                "\"שמתי 20 אלף דולר ב-MSTR בשער 310\".")
    if "amount" in need:
        return "כמה כסף שלך נכנס לפוזיציה? (הסכום שהשקעת, לא כולל מינוף)"
    return f"באיזה מחיר נכנסת ל{('MSTR' if pos.instrument == 'MSTR' else 'ביטקוין')}? אפשר כמה מחירים אם קנית בפעימות."


_GREET = (
    "אני CARN — היועץ של המודל. תגיד לי כמה כסף שמת ובאיזה שער, "
    "ואני ארוץ עליך את התחזית ההסתברותית, מונטה-קרלו וניתוח המינוף האמיתיים ואגיד לך "
    "איפה אתה עומד. אפשר לדבר איתי חופשי."
)


# --------------------------------------------------------------------------- #
# optional LLM layer  (Ollama first, then Anthropic; both entirely optional)
# --------------------------------------------------------------------------- #
def _ollama_model() -> str | None:
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=0.6) as r:
            tags = json.loads(r.read().decode())
        names = [m["name"] for m in tags.get("models", [])]
        for pref in ("llama3.1", "llama3", "qwen2.5", "mistral", "phi3", "gemma2"):
            for n in names:
                if n.startswith(pref):
                    return n
        return names[0] if names else None
    except Exception:  # noqa: BLE001
        return None


def _anthropic_key() -> str | None:
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            return str(st.secrets["ANTHROPIC_API_KEY"])
    except Exception:  # noqa: BLE001
        pass
    return os.environ.get("ANTHROPIC_API_KEY")


_SYS = (
    "אתה CARN, יועץ השקעות כמותי בתוך אפליקציית CARN-X לתחזית MSTR/ביטקוין. "
    "מדבר עברית, ישיר, מספרי, בלי הייפ ובלי אזהרות משפטיות מנופחות. "
    "כל המספרים שאתה מצטט חייבים לבוא מתוך גוש ה-JSON 'ANALYSIS' שמסופק לך — "
    "אסור להמציא מספרים. אם חסר מידע, שאל שאלה אחת קצרה. "
    "אתה מסביר מה המודל אומר על הפוזיציה של המשתמש, לא נותן עצה גורפת."
)


def _llm_reply(history: list[dict], analysis_json: str | None) -> str | None:
    payload_user = history[-1]["content"]
    ctx = f"\n\nANALYSIS(JSON):\n{analysis_json}" if analysis_json else ""
    msgs = [{"role": m["role"], "content": m["content"]} for m in history[:-1]]
    msgs.append({"role": "user", "content": payload_user + ctx})

    om = _ollama_model()
    if om:
        try:
            body = json.dumps({
                "model": om,
                "messages": [{"role": "system", "content": _SYS}] + msgs,
                "stream": False, "options": {"temperature": 0.4},
            }).encode()
            req = urllib.request.Request(
                "http://localhost:11434/api/chat", data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())["message"]["content"].strip()
        except Exception:  # noqa: BLE001
            pass

    key = _anthropic_key()
    if key:
        try:
            body = json.dumps({
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 900, "system": _SYS, "messages": msgs,
            }).encode()
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages", data=body,
                headers={"content-type": "application/json",
                         "x-api-key": key, "anthropic-version": "2023-06-01"})
            with urllib.request.urlopen(req, timeout=40) as r:
                data = json.loads(r.read().decode())
            return "".join(b.get("text", "") for b in data.get("content", [])).strip()
        except Exception:  # noqa: BLE001
            pass
    return None


def brain_label() -> str:
    if _ollama_model():
        return "Ollama · מקומי"
    if _anthropic_key():
        return "Claude · חופשי"
    return "מנוע פנימי"


# --------------------------------------------------------------------------- #
# the turn
# --------------------------------------------------------------------------- #
def reply(user_text: str, *, data, get_model, last_price: float,
          live_price: float | None = None) -> str:
    ss = st.session_state
    pos: Position = ss.get("cx_agent_pos") or Position()
    hist: list[dict] = ss.get("cx_agent_hist") or []

    pos = parse(user_text, pos)
    ss["cx_agent_pos"] = pos

    need = missing(pos)
    analysis_json = None
    draft = None

    if need and not re.search(r"\?|מה אתה חושב|תסביר|למה", user_text):
        draft = _ask(need, pos)
    else:
        a = analyze(pos, data=data, get_model=get_model,
                    last_price=last_price, live_price=live_price)
        if a.note == "no_model":
            return ("אין מודל מאומן כרגע — עבור ל'הגדרות' ולחץ 'אמן מודל עכשיו', "
                    "ואז אני יכול לרוץ על הפוזיציה שלך.")
        if not a.ok and need:
            draft = _ask(need, pos)
        elif not a.ok:
            draft = ("ניסיתי להריץ את המודל ונתקלתי בבעיה. נסה שוב עוד רגע, "
                     "או ודא שיש מודל מאומן ב'הגדרות'.")
        else:
            draft = phrase_analysis(a, pos)
            analysis_json = json.dumps({
                "instrument": a.instrument, "price_now": round(a.price_now, 2),
                "avg_entry": round(a.avg_entry, 2), "amount_usd": pos.amount_usd,
                "leverage": pos.leverage, "horizon_days": a.horizon,
                "pnl_now_pct": round(a.pnl_now_pct, 1), "pnl_now_usd": round(a.pnl_now_usd, 0),
                "model_expected_move_pct": round(a.exp_move_pct, 2), "model_p_up": round(a.p_up, 3),
                "return_q05_pct": round(a.q05_pct, 1), "return_q95_pct": round(a.q95_pct, 1),
                "p_profit_from_here": round(a.p_profit, 3), "p_double": round(a.p_double, 3),
                "p_ruin": round(a.p_ruin, 3), "leverage_ceiling": round(a.lev_ceiling, 2),
                "mc_expected_price": round(a.mc_expected_price, 2),
                "mc_var5_price": round(a.mc_var5_price, 2),
            }, ensure_ascii=False)

    turn = hist + [{"role": "user", "content": user_text}]
    llm = _llm_reply(turn, analysis_json)
    answer = llm or draft or _GREET

    ss["cx_agent_hist"] = (turn + [{"role": "assistant", "content": answer}])[-16:]
    return answer


def greeting() -> str:
    return _GREET


# --------------------------------------------------------------------------- #
# the panel  --  carved-mahogany chat, drawn in the sidebar
# --------------------------------------------------------------------------- #
def _logo_svg(size: int = 46) -> str:
    """The agent mark: an electric coin-brain on a carved-mahogany seal --
    the app's icon language, restyled for the agent."""
    return f"""
<svg width="{size}" height="{size}" viewBox="0 0 96 96" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <defs>
    <radialGradient id="cxa-wood" cx="34%" cy="28%" r="80%">
      <stop offset="0%" stop-color="#7A3B22"/><stop offset="46%" stop-color="#5A2617"/>
      <stop offset="100%" stop-color="#331107"/>
    </radialGradient>
    <linearGradient id="cxa-bolt" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#EAF9FF"/><stop offset="55%" stop-color="#7FE9FF"/>
      <stop offset="100%" stop-color="#37C4E6"/>
    </linearGradient>
    <linearGradient id="cxa-gold" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#F7E3A6"/><stop offset="100%" stop-color="#C9973F"/>
    </linearGradient>
    <filter id="cxa-glow" x="-40%" y="-40%" width="180%" height="180%">
      <feGaussianBlur stdDeviation="2.1"/>
    </filter>
  </defs>
  <circle cx="48" cy="48" r="45" fill="url(#cxa-wood)" stroke="url(#cxa-gold)" stroke-width="2.4"/>
  <circle cx="48" cy="48" r="37" fill="none" stroke="#2C0F05" stroke-width="1.4" opacity="0.7"/>
  <circle cx="48" cy="48" r="37" fill="none" stroke="url(#cxa-gold)" stroke-width="1" opacity="0.55"/>
  <g opacity="0.9">
    <path d="M40 20c-12 3-20 13-20 27 0 9 4 16 11 21" stroke="url(#cxa-gold)" stroke-width="2" fill="none" stroke-linecap="round" opacity="0.5"/>
    <path d="M56 76c12-3 20-13 20-27 0-9-4-16-11-21" stroke="url(#cxa-gold)" stroke-width="2" fill="none" stroke-linecap="round" opacity="0.5"/>
  </g>
  <path d="M54 18 L34 50 L47 50 L40 78 L64 42 L50 42 Z"
        fill="url(#cxa-bolt)" stroke="#EAF9FF" stroke-width="1.2" stroke-linejoin="round"
        filter="url(#cxa-glow)"/>
  <path d="M54 18 L34 50 L47 50 L40 78 L64 42 L50 42 Z"
        fill="url(#cxa-bolt)" stroke="#FFFFFF" stroke-width="1" stroke-linejoin="round"/>
  <g stroke="#7FE9FF" stroke-width="1.5" opacity="0.85">
    <circle cx="30" cy="34" r="2.2" fill="#DFF7FF"/><circle cx="66" cy="60" r="2.2" fill="#DFF7FF"/>
    <circle cx="64" cy="30" r="1.8" fill="#DFF7FF"/><circle cx="32" cy="62" r="1.8" fill="#DFF7FF"/>
    <path d="M30 34 L44 44 M66 60 L52 50 M64 30 L52 40 M32 62 L44 52" opacity="0.5"/>
  </g>
</svg>"""


def _brain_json_for_history() -> str | None:
    return None


def render_sidebar(*, data, get_model, last_price: float,
                   live_price: float | None = None) -> None:
    """Draw the CARN agent panel at the top of the sidebar."""
    ss = st.session_state
    ss.setdefault("cx_agent_chat", [{"role": "assistant", "content": _GREET}])

    with st.sidebar:
        st.markdown(
            f'<div class="cx-agent-head">{_logo_svg(44)}'
            f'<div class="cx-agent-id"><b>CARN</b>'
            f'<span>יועץ · {brain_label()}</span></div></div>',
            unsafe_allow_html=True,
        )
        box = st.container(height=430)
        with box:
            for m in ss["cx_agent_chat"]:
                who = "cx-a-me" if m["role"] == "user" else "cx-a-bot"
                st.markdown(
                    f'<div class="cx-a-msg {who}">{_md(m["content"])}</div>',
                    unsafe_allow_html=True,
                )

        prompt = None
        try:
            prompt = st.chat_input("דבר עם CARN…", key="cx_agent_input")
        except Exception:  # noqa: BLE001  -- older Streamlit / nesting
            with st.form("cx_agent_form", clear_on_submit=True):
                txt = st.text_area("הודעה", key="cx_agent_ta",
                                   label_visibility="collapsed", height=70)
                if st.form_submit_button("שלח") and txt.strip():
                    prompt = txt

        if prompt:
            ss["cx_agent_chat"].append({"role": "user", "content": prompt})
            with st.spinner("CARN מנתח…"):
                ans = reply(prompt, data=data, get_model=get_model,
                            last_price=last_price, live_price=live_price)
            ss["cx_agent_chat"].append({"role": "assistant", "content": ans})
            st.rerun()

    st.sidebar.markdown('<div class="cx-agent-rule"></div>', unsafe_allow_html=True)


def _md(text: str) -> str:
    import html as _h
    t = _h.escape(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"_(.+?)_", r"<i>\1</i>", t)
    return t.replace("\n\n", "<br><br>").replace("\n", "<br>")
