#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

# ------------------------------------------------------------
# Argparse + polkujen varmistus
# ------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Auto Decider – muodostaa action-planin, trade- & sell-listat sekä portfolio_after_sim.csv:n."
    )
    ap.add_argument("--project_root", required=True, help="Projektin juuripolku (C:\\...\\seasonality_project)")
    ap.add_argument("--universe_csv", required=True, help="Universumi CSV (esim. seasonality_reports\\constituents_raw.csv)")
    ap.add_argument("--run_root", required=True, help="Käytettävä RUN-kansio (älä muuta jos käytät vakiota)")
    ap.add_argument("--price_cache_dir", required=True, help="Käytettävä price_cache-kansio")
    ap.add_argument("--today", required=True, help="Päivä YYYY-MM-DD")
    # >>> MUUTOS: ei enää required=True
    ap.add_argument(
        "--portfolio_state",
        default=None,
        help="Salkun tilan JSON (oletus: <project_root>\\seasonality_reports\\portfolio_state.json)"
    )
    ap.add_argument("--commit", type=int, default=0, help="1 = kirjoita portfolio_state.json; 0 = dry-run")
    ap.add_argument("--config", default=None, help="(valinn.) polku asetustiedostolle")
    ap.add_argument("--no_new_positions", action="store_true", help="Estä uudet positiot (vain myyntejä/kevennyksiä)")
    ap.add_argument("--vintage_cutoff", default=None, help="(valinn.) aikaraja datalle")

    args = ap.parse_args()

    # >>> MUUTOS: oletuspolku portfolio_state:lle jos sitä ei annettu
    if not args.portfolio_state:
        args.portfolio_state = os.path.join(
            os.path.abspath(args.project_root),
            "seasonality_reports",
            "portfolio_state.json",
        )

    # Peruspolut ja validoinnit
    args.project_root = os.path.abspath(args.project_root)
    args.run_root = os.path.abspath(args.run_root)
    args.price_cache_dir = os.path.abspath(args.price_cache_dir)
    args.universe_csv = os.path.abspath(args.universe_csv)
    args.portfolio_state = os.path.abspath(args.portfolio_state)

    if not os.path.isdir(args.run_root):
        raise FileNotFoundError(f"run_root puuttuu: {args.run_root}")
    if not os.path.isdir(args.price_cache_dir):
        raise FileNotFoundError(f"price_cache_dir puuttuu: {args.price_cache_dir}")
    if not os.path.isfile(args.universe_csv):
        raise FileNotFoundError(f"universe_csv puuttuu: {args.universe_csv}")

    # actions/YYYMMDD
    ymd = args.today.replace("-", "")
    args.actions_dir = os.path.join(args.run_root, "actions", ymd)
    os.makedirs(args.actions_dir, exist_ok=True)

    print(f"[INFO] Using RUN_ROOT : {args.run_root}")
    print(f"[INFO] Using PRICES  : {args.price_cache_dir}")
    print(f"[INFO] Using UNIVERSE: {args.universe_csv}")
    print(f"[INFO] PortfolioState: {args.portfolio_state}")
    print(f"[INFO] Actions dir   : {args.actions_dir}")

    return args


# ------------------------------------------------------------
# Alla on yksinkertaiset apurutiinit. Varsinainen valintalogiikka
# on jätetty ennalleen; jos sinulla on aiemmassa versiossa omat
# funktiot (esim. build_trade_candidates), ne voidaan käyttää tässä.
# ------------------------------------------------------------

def _safe_read_csv(path: str) -> pd.DataFrame:
    """Lukee CSV:n yrittäen automaattisesti erotinta (,/;)."""
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=";")

def _write_csv(df: pd.DataFrame, path: str):
    if df is None:
        return
    df.to_csv(path, index=False)

def _load_portfolio_state(path_json: str) -> dict:
    if os.path.isfile(path_json):
        try:
            with open(path_json, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"positions": []}

def _save_portfolio_state(state: dict, path_json: str):
    Path(os.path.dirname(path_json)).mkdir(parents=True, exist_ok=True)
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

# ------------------------------------------------------------
# Paikanpitäjälogiikka: säilytetään nykyinen käyttäytyminen.
# Jos sinulla on jo projektissa omat funktiot, ne voivat korvata
# nämä helposti – tämä runko vain varmistaa tiedostojen syntymisen.
# ------------------------------------------------------------

def build_trade_candidates(args) -> pd.DataFrame:
    """
    Tässä kohtaa kutsutaan yleensä ML/overlay -putkea ja filtteröintiä.
    Säilytetään sarakenimet aiempaan tapaan ('Ticker', 'Side', ...).
    Jos sinulla on jo valmis toteutus, korvaa tämä funktio sillä.
    """
    # Esimerkkirunko: tyhjä df jos data puuttuu
    cols = ["Ticker", "Side", "Score", "Note"]
    return pd.DataFrame([], columns=cols)

def build_sell_candidates(args, portfolio_state) -> pd.DataFrame:
    cols = ["Ticker", "Reason"]
    return pd.DataFrame([], columns=cols)

def simulate_portfolio_after_trades(portfolio_state, buys_df, sells_df) -> pd.DataFrame:
    """
    Muodosta taulukko, joka tallennetaan portfolio_after_sim.csv:ksi.
    Oikeassa toteutuksessa tämä päivittää positioita, kappalemääriä jne.
    """
    cols = ["Ticker", "Side", "Qty", "Comment"]
    return pd.DataFrame([], columns=cols)

def write_action_plan(args, buys_df, sells_df):
    plan = []
    plan.append(f"Date: {args.today}")
    plan.append(f"Universe: {os.path.basename(args.universe_csv)}")
    plan.append(f"Run root: {args.run_root}")
    plan.append("")
    plan.append("=== Buys ===")
    if buys_df is not None and len(buys_df):
        for _, r in buys_df.iterrows():
            plan.append(f"- {r.get('Ticker','?')} (Score={r.get('Score','')})")
    else:
        plan.append("(none)")
    plan.append("")
    plan.append("=== Sells ===")
    if sells_df is not None and len(sells_df):
        for _, r in sells_df.iterrows():
            plan.append(f"- {r.get('Ticker','?')} ({r.get('Reason','')})")
    else:
        plan.append("(none)")

    out = os.path.join(args.actions_dir, "action_plan.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(plan))
    print(f"[OK] Action Plan : {out}")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    args = parse_args()

    # Lataa/valmistele salkun tila
    state = _load_portfolio_state(args.portfolio_state)

    # Rakenna listat (tässä pidetään kiinni aiemmista tiedostonimistä)
    buys_df  = build_trade_candidates(args)
    sells_df = build_sell_candidates(args, state)
    port_df  = simulate_portfolio_after_trades(state, buys_df, sells_df)

    # Kirjoitukset
    trade_path = os.path.join(args.actions_dir, "trade_candidates.csv")
    sell_path  = os.path.join(args.actions_dir, "sell_candidates.csv")
    port_path  = os.path.join(args.actions_dir, "portfolio_after_sim.csv")

    _write_csv(buys_df, trade_path)
    print(f"[OK] Candidates : {trade_path}")

    _write_csv(sells_df, sell_path)
    print(f"[OK] Sells      : {sell_path}")

    _write_csv(port_df, port_path)
    print(f"[OK] Portfolio  : {port_path}")

    # Action plan
    write_action_plan(args, buys_df, sells_df)

    # Tallennetaanko portfolio_state.json?
    if int(args.commit) == 1:
        _save_portfolio_state(state, args.portfolio_state)
        print(f"[OK] Portfolio state COMMITTED: {args.portfolio_state}")
    else:
        print("[INFO] Dry-run: portfolio_state.json ei päivitetty (commit=0).")


if __name__ == "__main__":
    main()
