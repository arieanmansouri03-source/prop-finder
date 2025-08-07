# -*- coding: utf-8 -*-
# prop_finder_streamlit.py
# -------------------------------------------------------------
# "Prop Finder" - EV/Edge-ranker for player props and other markets
# Sports: Basketball, Hockey, NFL, Football (with Top-20 leagues filter)
# Streamlit app: no scraping; you provide legal data via CSV or an API you are allowed to use.
# Includes: EV/Edge, vig adjustment, Kelly stake, CLV, ROI tracker, Discord alerts,
# simple password gate via secrets, export/import logs (CSV).
# -------------------------------------------------------------

import json
import hashlib
from typing import Optional, Dict, Tuple, List

import pandas as pd
import numpy as np
import streamlit as st
import requests

# ----------------------- Configuration -----------------------
DEFAULT_TOP20_FOOTBALL_LEAGUES = [
    "Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1",
    "Eredivisie", "Primeira Liga", "Belgian Pro League", "Scottish Premiership",
    "Super Lig", "MLS", "Brasileirao", "Argentine Primera", "Liga MX",
    "Austrian Bundesliga", "Swiss Super League", "Danish Superliga", "Russian Premier League",
    "Ukrainian Premier League", "Greek Super League"
]

DEFAULT_SPORTS = ["Basket", "Hockey", "NFL", "Fotboll"]

# ----------------------- Helpers -----------------------------

def send_discord(webhook_url: str, message: str):
    try:

numpy==1.26.4
requests==2.32.3
"""
)
