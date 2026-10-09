"""
fpl_data.py — Lapisan data FPL (tanpa dependensi Streamlit, mudah diuji).

Berisi:
  * fetch bootstrap / fixtures / element-summary (paralel, dengan retry)
  * import skuad dari Entry ID: picks, bank, FREE TRANSFER (disimulasikan dari
    histori), HARGA JUAL (aturan profit dibagi dua), dan chip yang sudah dipakai
    beserta GW-nya (penting karena 2026/27 punya 2 set chip: GW1-19 & GW20-38).
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import requests

API = "https://fantasy.premierleague.com/api"
HEADERS = {"User-Agent": "Mozilla/5.0 (FPL-Optimizer-v3)"}
CHIP_NAMES = {"wildcard": "Wildcard", "freehit": "Free Hit",
              "bboost": "Bench Boost", "3xc": "Triple Captain"}
FIRST_HALF_LAST_GW = 19          # set chip pertama harus dipakai sebelum deadline GW19


def _get(url: str, timeout: int = 30, retries: int = 3):
    for k in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (403, 404):
                return None
        except requests.RequestException:
            pass
        time.sleep(0.8 * (k + 1))
    return None


def fetch_bootstrap():
    return _get(f"{API}/bootstrap-static/")


def fetch_fixtures():
    return _get(f"{API}/fixtures/")


def fetch_summaries(player_ids, workers: int = 16) -> dict:
    """element-summary (histori per-GW + history_past) untuk banyak pemain."""
    def one(pid):
        return pid, _get(f"{API}/element-summary/{pid}/", timeout=20, retries=2)
    out = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for pid, data in ex.map(one, list(player_ids)):
            out[int(pid)] = data
    return out


# --------------------------------------------------------------------------
# Status GW
# --------------------------------------------------------------------------
def gameweek_status(bootstrap) -> dict:
    ev = bootstrap["events"]
    nxt = next((e["id"] for e in ev if e.get("is_next")), None)
    cur = next((e["id"] for e in ev if e.get("is_current")), None)
    last_gw = max(e["id"] for e in ev)
    if nxt is None:
        nxt = (cur + 1) if cur else 1
    return {"next_gw": nxt, "current_gw": cur, "last_gw": last_gw}


# --------------------------------------------------------------------------
# Free transfer, harga jual, chip
# --------------------------------------------------------------------------
def compute_free_transfers(history_current: list, chips: list, next_gw: int) -> int:
    """Simulasi aturan FT: +1 per GW, simpan maks 5, WC/FH tidak memakai FT.

    FT tersedia untuk GW2 = 1. Setelah GW k:
      * WC/FH  -> ft = min(5, ft + 1)
      * lainnya -> ft = min(5, max(ft - transfers, 0) + 1)
    """
    chip_by_event = {c["event"]: c["name"] for c in chips}
    per_event = {h["event"]: h for h in history_current}
    ft = 1
    for gw in range(2, next_gw):
        h = per_event.get(gw)
        n = h.get("event_transfers", 0) if h else 0
        if chip_by_event.get(gw) in ("wildcard", "freehit"):
            ft = min(5, ft + 1)
        else:
            ft = min(5, max(ft - n, 0) + 1)
    return int(ft)


def sell_price(buy: int, now: int) -> int:
    """Aturan FPL: harga jual = beli + floor(profit / 2); jika turun = harga kini."""
    return buy + (now - buy) // 2 if now > buy else now


def fetch_entry(entry_id, bootstrap) -> tuple[dict | None, str]:
    st = gameweek_status(bootstrap)
    nxt, cur = st["next_gw"], st["current_gw"]
    els = {e["id"]: e for e in bootstrap["elements"]}

    picks = None
    used_gw = None
    for gw in dict.fromkeys([nxt, cur, nxt - 1]):
        if gw and gw >= 1:
            picks = _get(f"{API}/entry/{entry_id}/event/{gw}/picks/", timeout=20, retries=2)
            if picks:
                used_gw = gw
                break
    if not picks:
        return None, "Entry ID tidak ditemukan / picks belum tersedia."

    hist = _get(f"{API}/entry/{entry_id}/history/", timeout=20, retries=2) or {}
    trf = _get(f"{API}/entry/{entry_id}/transfers/", timeout=20, retries=2) or []

    chips_used = [{"name": c["name"], "event": c["event"]} for c in hist.get("chips", [])
                  if c.get("name") in CHIP_NAMES]
    ac = picks.get("active_chip")
    if ac in CHIP_NAMES and not any(c["event"] == used_gw and c["name"] == ac for c in chips_used):
        chips_used.append({"name": ac, "event": used_gw})

    ids = [p["element"] for p in picks["picks"]]
    last_in = {}
    for t in sorted(trf, key=lambda x: x.get("time", "")):
        last_in[t["element_in"]] = t["element_in_cost"]
    sells = {}
    for pid in ids:
        now = els[pid]["now_cost"]
        buy = last_in.get(pid, now - els[pid].get("cost_change_start", 0))
        sells[pid] = sell_price(int(buy), int(now))

    eh = picks.get("entry_history", {})
    ft = compute_free_transfers(hist.get("current", []), hist.get("chips", []), nxt)
    return {
        "player_ids": ids,
        "bank": eh.get("bank", 0) / 10.0,
        "ft": ft,
        "chips_used": chips_used,
        "sell_prices": sells,
        "picks_gw": used_gw,
        "squad_value": eh.get("value", 0) / 10.0,
    }, f"OK — skuad GW{used_gw} diimpor (FT dihitung dari histori transfer)."


def chips_available(chips_used: list, gws: list[int]) -> dict:
    """{(chip, half): bool} — tersedia atau tidak untuk tiap paruh musim."""
    avail = {}
    for name in CHIP_NAMES.values():
        for half in (1, 2):
            avail[(name, half)] = True
    for c in chips_used:
        nm = CHIP_NAMES.get(c["name"])
        half = 1 if c["event"] <= FIRST_HALF_LAST_GW else 2
        if nm:
            avail[(nm, half)] = False
    return avail
