# prop_finder_streamlit.py
# -------------------------------------------------------------
# "Prop Finder" - EV/Edge ranker for player props and other markets
# Sports: Basketball, Hockey, NFL, Football (with Top-20 leagues filter)
# Streamlit app: no scraping; you provide legal data via CSV or an API you may use.
# Includes: EV/Edge, vig adjustment, Kelly stake, CLV, ROI tracker, Discord alerts,
# optional password via secrets, export/import logs (CSV).
# ASCII-only text to avoid encoding issues.
# -------------------------------------------------------------

import json
import hashlib
from typing import Optional, Dict, Tuple, List

import pandas as pd
import numpy as np
import streamlit as st
import requests

# ----------------------- Config ------------------------------
DEFAULT_TOP20_FOOTBALL_LEAGUES = [
    "Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1",
    "Eredivisie", "Primeira Liga", "Belgian Pro League", "Scottish Premiership",
    "Super Lig", "MLS", "Brasileirao", "Argentine Primera", "Liga MX",
    "Austrian Bundesliga", "Swiss Super League", "Danish Superliga", "Russian Premier League",
    "Ukrainian Premier League", "Greek Super League"
]

DEFAULT_SPORTS = ["Basket", "Hockey", "NFL", "Fotboll"]

# ----------------------- Helpers -----------------------------
def send_discord(webhook_url: str, message: str) -> None:
    try:
        r = requests.post(webhook_url, json={"content": message}, timeout=10)
        r.raise_for_status()
    except Exception as e:
        st.warning(f"Discord webhook failed: {e}")

def american_to_decimal(american_odds: float) -> float:
    try:
        american_odds = float(american_odds)
    except Exception:
        return np.nan
    if american_odds >= 100:
        return 1.0 + (american_odds / 100.0)
    if american_odds <= -100:
        return 1.0 + (100.0 / abs(american_odds))
    return np.nan

def implied_prob_from_decimal(o: float) -> float:
    return 1.0 / o if o and o > 1.0 else np.nan

def remove_vig_two_sided(p_over: float, p_under: float) -> Tuple[float, float]:
    if any(pd.isna([p_over, p_under])):
        return p_over, p_under
    s = p_over + p_under
    if s <= 0 or np.isnan(s):
        return p_over, p_under
    return p_over / s, p_under / s

def ev_decimal(p: float, o: float) -> float:
    if pd.isna(p) or pd.isna(o):
        return np.nan
    return p * (o - 1.0) - (1.0 - p)

def edge_percent(p: float, o: float) -> float:
    if pd.isna(p) or pd.isna(o) or p <= 0:
        return np.nan
    fair = 1.0 / p
    return (o - fair) / fair * 100.0

def stable_hash(row: Dict) -> str:
    m = hashlib.sha256()
    m.update(json.dumps(row, sort_keys=True, default=str).encode("utf-8"))
    return m.hexdigest()[:12]

def kelly_stake(bankroll: float, p: float, o: float, kelly_cap: float = 0.25, min_unit: float = 1.0) -> float:
    if bankroll <= 0 or pd.isna(p) or pd.isna(o):
        return 0.0
    b = max(o - 1.0, 0.0)
    q = 1.0 - p
    if b <= 0:
        return 0.0
    f = (b * p - q) / b
    f = max(0.0, min(f, kelly_cap))
    stake = round(max(min_unit, bankroll * f), 2)
    return stake

# ----------------------- CSV schema --------------------------
REQUIRED_ODDS_COLS = [
    "book", "sport", "league", "player", "market", "selection",
    "line", "odds_decimal", "odds_american", "timestamp", "event_id", "opponent"
]

REQUIRED_PROJ_COLS = [
    "sport", "league", "player", "market", "line", "prob_over", "prob_under"
]

# ----------------------- UI setup ----------------------------
st.set_page_config(page_title="Prop Finder", page_icon=":dart:", layout="wide")

# Optional password protection via Secrets
password_required = bool(st.secrets.get("APP_PASSWORD"))
if password_required:
    pw = st.text_input("Password", type="password")
    if pw != st.secrets["APP_PASSWORD"]:
        st.stop()

st.title("Prop Finder - Player Props (Basket, Hockey, NFL, Fotboll)")

with st.expander("How it works", expanded=True):
    st.markdown(
        """
Upload two CSV files (see required columns below). The app computes EV and Edge%, suggests Kelly stake, logs bets, and tracks CLV and ROI.

CSV formats
odds_props.csv: book,sport,league,player,market,selection,line,odds_decimal,odds_american,timestamp,event_id,opponent
projections.csv: sport,league,player,market,line,prob_over,prob_under
"""
    )

# ----------------------- State -------------------------------
ss = st.session_state
ss.setdefault("bankroll", 10000.0)
ss.setdefault("kelly_cap", 0.25)
ss.setdefault("min_unit", 50.0)
ss.setdefault(
    "bets_log",
    pd.DataFrame(
        columns=[
            "time", "pick_id", "sport", "league", "player", "market", "selection", "line",
            "odds_open", "odds_now", "model_p", "stake", "book", "event_id", "timestamp", "opponent",
            "status", "profit", "clv_pct"
        ]
    )
)
ss.setdefault("seen_pick_ids", set())

# ----------------------- Sidebar -----------------------------
with st.sidebar:
    st.header("Settings")
    sports_selected = st.multiselect("Sports", DEFAULT_SPORTS, default=DEFAULT_SPORTS)
    top20_default = st.toggle("Filter football to Top-20 leagues", value=False)
    top20_list_text = st.text_area(
        "Top-20 list (one per line)",
        value="\n".join(DEFAULT_TOP20_FOOTBALL_LEAGUES)
    )
    min_edge = st.slider("Min Edge %", -10.0, 20.0, 3.0, 0.5)
    min_ev = st.slider("Min EV", -1.0, 1.0, -0.05, 0.01)

    st.divider()
    st.subheader("Bankroll")
    ss["bankroll"] = st.number_input("Bankroll", min_value=0.0, value=float(ss["bankroll"]))
    ss["kelly_cap"] = st.slider("Kelly cap (max fraction)", 0.0, 1.0, float(ss["kelly_cap"]), 0.05)
    ss["min_unit"] = st.number_input("Minimum stake (unit)", min_value=0.0, value=float(ss["min_unit"]))

    st.divider()
    st.subheader("Alerts")
    discord_webhook = st.text_input("Discord Webhook (optional)")
    auto_alert = st.toggle("Send alerts for new picks", value=False)

# ----------------------- Upload ------------------------------
col_upl1, col_upl2 = st.columns(2)
with col_upl1:
    odds_file = st.file_uploader("Upload odds_props.csv", type=["csv"], key="odds")
with col_upl2:
    proj_file = st.file_uploader("Upload projections.csv", type=["csv"], key="proj")

# ----------------------- Read CSV ----------------------------
def read_csv(file, required_cols: List[str]) -> Optional[pd.DataFrame]:
    if not file:
        return None
    try:
        df = pd.read_csv(file)
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            return None
        return df
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return None

odds_df = read_csv(odds_file, REQUIRED_ODDS_COLS)
proj_df = read_csv(proj_file, REQUIRED_PROJ_COLS)

# ----------------------- Process -----------------------------
results_df = None
if odds_df is not None and proj_df is not None:
    # decimal odds if missing
    mask_na_dec = odds_df["odds_decimal"].isna() | (odds_df["odds_decimal"] <= 1)
    odds_df.loc[mask_na_dec, "odds_decimal"] = odds_df.loc[mask_na_dec, "odds_american"].apply(american_to_decimal)

    # filter sports
    odds_df = odds_df[odds_df["sport"].isin(sports_selected)].copy()
    proj_df = proj_df[proj_df["sport"].isin(sports_selected)].copy()

    # Football: Top-20 filter
    if top20_default and "Fotboll" in sports_selected:
        top20_set = set([s.strip() for s in top20_list_text.splitlines() if s.strip()])

        is_football_odds = odds_df["sport"] == "Fotboll"
        odds_df = pd.concat(
            [
                odds_df[~is_football_odds],
                odds_df[is_football_odds & odds_df["league"].isin(top20_set)]
            ]
        )

        is_football_proj = proj_df["sport"] == "Fotboll"
        proj_df = pd.concat(
            [
                proj_df[~is_football_proj],
                proj_df[is_football_proj & proj_df["league"].isin(top20_set)]
            ]
        )

    key_cols = ["sport", "league", "player", "market", "line"]
    merged = odds_df.merge(proj_df, on=key_cols, how="left", suffixes=("", "_proj"))

    merged["imp_prob"] = merged["odds_decimal"].apply(implied_prob_from_decimal)

    grp_cols_wo_sel = key_cols + ["book", "event_id"]

    def vig_adjust(group: pd.DataFrame) -> pd.DataFrame:
        sels = set(group["selection"].astype(str).str.lower())
        if {"over", "under"}.issubset(sels) and group["imp_prob"].notna().all():
            over_prob = group.loc[group["selection"].str.lower() == "over", "imp_prob"].iloc[0]
            under_prob = group.loc[group["selection"].str.lower() == "under", "imp_prob"].iloc[0]
            p_over_fair, p_under_fair = remove_vig_two_sided(over_prob, under_prob)
            group.loc[group["selection"].str.lower() == "over", "imp_prob_fair"] = p_over_fair
            group.loc[group["selection"].str.lower() == "under", "imp_prob_fair"] = p_under_fair
        else:
            group["imp_prob_fair"] = group["imp_prob"]
        return group

    merged = merged.groupby(grp_cols_wo_sel, group_keys=False).apply(vig_adjust)

    def pick_model_p(row) -> float:
        sel = str(row.get("selection", "")).lower()
        if sel == "over":
            return row.get("prob_over", np.nan)
        if sel == "under":
            return row.get("prob_under", np.nan)
        if sel == "yes":
            return row.get("prob_over", np.nan)
        if sel == "no":
            return row.get("prob_under", np.nan)
        return np.nan

    merged["model_p"] = merged.apply(pick_model_p, axis=1)

    merged["EV"] = merged.apply(lambda r: ev_decimal(r["model_p"], r["odds_decimal"]), axis=1)
    merged["EdgePct"] = merged.apply(lambda r: edge_percent(r["model_p"], r["odds_decimal"]), axis=1)

    merged["KellyStake"] = merged.apply(
        lambda r: kelly_stake(ss["bankroll"], r["model_p"], r["odds_decimal"], ss["kelly_cap"], ss["min_unit"]),
        axis=1
    )

    filt = (merged["EdgePct"] >= min_edge) & (merged["EV"] >= min_ev)
    results_df = merged.loc[filt].copy()

    results_df.sort_values(["EdgePct", "EV"], ascending=False, inplace=True)

    id_cols = ["book", "sport", "league", "player", "market", "selection", "line", "event_id"]
    results_df["pick_id"] = results_df[id_cols].astype(str).apply(lambda r: stable_hash(dict(zip(id_cols, r))), axis=1)

# ----------------------- Finder view -------------------------
if results_df is not None and not results_df.empty:
    st.subheader("Best picks right now")

    leagues = sorted(results_df["league"].dropna().unique().tolist())
    markets = sorted(results_df["market"].dropna().unique().tolist())
    books = sorted(results_df["book"].dropna().unique().tolist())

    c1, c2, c3 = st.columns(3)
    with c1:
        sel_leagues = st.multiselect("Filter league", leagues, default=[])
    with c2:
        sel_markets = st.multiselect("Filter market", markets, default=[])
    with c3:
        sel_books = st.multiselect("Filter book", books, default=[])

    view_df = results_df.copy()
    if sel_leagues:
        view_df = view_df[view_df["league"].isin(sel_leagues)]
    if sel_markets:
        view_df = view_df[view_df["market"].isin(sel_markets)]
    if sel_books:
        view_df = view_df[view_df["book"].isin(sel_books)]

    display_cols = [
        "sport", "league", "player", "market", "selection", "line",
        "odds_decimal", "model_p", "EV", "EdgePct", "KellyStake", "book", "timestamp", "opponent"
    ]
    st.dataframe(view_df[display_cols], use_container_width=True, height=520)

    # export picks
    csv = view_df[display_cols + ["pick_id", "event_id"]].to_csv(index=False).encode("utf-8")
    st.download_button("Download picks (CSV)", data=csv, file_name="prop_values.csv", mime="text/csv")

    # log picks
    if st.button("Log visible picks as placed"):
        now = pd.Timestamp.utcnow().isoformat()
        to_log = view_df.copy()
        to_log = to_log.assign(
            time=now,
            odds_open=view_df["odds_decimal"],
            odds_now=view_df["odds_decimal"],
            model_p=view_df["model_p"],
            stake=view_df["KellyStake"],
            status="open",
            profit=0.0,
            clv_pct=0.0
        )
        log_cols = ss["bets_log"].columns
        ss["bets_log"] = pd.concat([ss["bets_log"], to_log[log_cols]], ignore_index=True).drop_duplicates(
            subset=["pick_id"], keep="last"
        )
        st.success(f"Logged {len(to_log)} picks.")

    # discord alerts
    if auto_alert and discord_webhook:
        new_rows = view_df[~view_df["pick_id"].isin(ss["seen_pick_ids"])]
        for _, r in new_rows.iterrows():
            msg = (
                f"**{r['sport']} - {r['league']}**\n"
                f"{r['player']} | {r['market']} {str(r['selection']).upper()} {r['line']} @ {r['odds_decimal']} ({r['book']})\n"
                f"Edge: {r['EdgePct']:.2f}% | EV: {r['EV']:.3f} | p(model): {r['model_p']:.3f}\n"
                f"Event: {r.get('event_id','-')} | {r.get('timestamp','')}"
            )
            send_discord(discord_webhook, msg)
        ss["seen_pick_ids"].update(set(view_df["pick_id"].tolist()))
        st.info(f"Sent {len(new_rows)} new alerts.")

else:
    st.info("Upload CSV files above to see results.")

# ----------------------- Tracker & CLV -----------------------
st.subheader("Tracker - ROI and CLV")

if results_df is not None and not ss["bets_log"].empty:
    latest_odds = results_df.set_index("pick_id")["odds_decimal"].to_dict()
    log = ss["bets_log"].copy()
    log["odds_now"] = log.apply(lambda r: latest_odds.get(r["pick_id"], r["odds_now"]), axis=1)
    log["clv_pct"] = (log["odds_now"] - log["odds_open"]) / log["odds_open"] * 100.0
    ss["bets_log"] = log

edit_log = st.data_editor(ss["bets_log"], use_container_width=True, height=360, key="log_editor")
ss["bets_log"] = edit_log

if not ss["bets_log"].empty:
    log = ss["bets_log"].copy()

    def settle_profit(row):
        if row["status"] == "won":
            return round(row["stake"] * (row["odds_open"] - 1.0), 2)
        if row["status"] == "lost":
            return -round(row["stake"], 2)
        if row["status"] == "push":
            return 0.0
        return row.get("profit", 0.0)

    log["profit"] = log.apply(settle_profit, axis=1)
    total_profit = log["profit"].sum()
    st.metric("Total result", f"{total_profit:.2f}")

    cexp, cimp = st.columns(2)
    with cexp:
        dl = log.to_csv(index=False).encode("utf-8")
        st.download_button("Download log (CSV)", data=dl, file_name="bets_log.csv", mime="text/csv")
    with cimp:
        up = st.file_uploader("Import log (CSV)", type=["csv"], key="logimp")
        if up is not None:
            try:
                imp = pd.read_csv(up)
                ss["bets_log"] = pd.concat([ss["bets_log"], imp], ignore_index=True).drop_duplicates(
                    subset=["pick_id"], keep="last"
                )
                st.success("Imported log.")
            except Exception as e:
                st.error(f"Import failed: {e}")

# ----------------------- Deploy guide ------------------------
st.divider()
st.subheader("Deploy - Streamlit Cloud")
st.markdown(
    """
1. Create a GitHub repo with prop_finder_streamlit.py and requirements.txt.
2. Go to share.streamlit.io and connect your repo.
3. (Optional) Secrets: APP_PASSWORD and DISCORD_WEBHOOK.
4. Start the app - you get a public URL.

requirements.txt
streamlit==1.36.0
pandas==2.2.2
numpy==1.26.4
requests==2.32.3
"""
)

