import time

import numpy as np
import pandas as pd
import streamlit as st

import fpl_data as D
import fpl_model as M
import fpl_optimizer as O

# ============================================================
# FPL OPTIMIZER v3 — xP Predictor (struktural + GBM, di-backtest)
#                    + MILP multi-GW (transfer, hit, WC/FH/BB/TC)
# ============================================================
st.set_page_config(page_title="FPL Optimizer v3", layout="wide", page_icon="⚽")


def show(df, **kw):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def primary_button(label, **kw):
    try:
        return st.button(label, type="primary", width="stretch", **kw)
    except Exception:
        return st.button(label, type="primary", use_container_width=True, **kw)


# ------------------------------------------------------------------ data
@st.cache_data(ttl=1800, show_spinner=False)
def load_core():
    return D.fetch_bootstrap(), D.fetch_fixtures()


@st.cache_resource(ttl=1800, show_spinner=False)
def load_context(stamp):
    b, fx = load_core()
    ids = [e["id"] for e in b["elements"] if e["status"] != "u"]
    summaries = D.fetch_summaries(ids)
    return M.build_context(b, fx, summaries)


bootstrap, fixtures = load_core()
if not bootstrap or fixtures is None:
    st.error("Gagal terhubung ke API Fantasy Premier League. Coba lagi beberapa saat.")
    st.stop()

gwst = D.gameweek_status(bootstrap)
next_gw, last_gw = gwst["next_gw"], gwst["last_gw"]
stamp = (next_gw, sum(1 for f in fixtures if f.get("finished")))

st.title("⚽ FPL Optimizer v3")
st.caption("Stage 1: model xP (struktural + GBM, divalidasi walk-forward)  •  "
           "Stage 2: MILP multi-GW (transfer, hit, Wildcard / Free Hit / Bench Boost / Triple Captain)")

with st.spinner("Mengunduh histori semua pemain & melatih model (pertama kali ±1 menit, lalu di-cache)…"):
    try:
        ctx = load_context(stamp)
    except Exception as e:
        st.error(f"Gagal membangun model: {e}")
        st.stop()
el = ctx.el
TEAM_SHORT = dict(zip(el["team"], el["team_short"]))
lab = lambda pid: f"{el.at[pid, 'web_name']} ({el.at[pid, 'team_short']})"
fmt_opt = lambda pid: f"{el.at[pid, 'web_name']} · {el.at[pid, 'team_short']} · {el.at[pid, 'pos_name']} · £{el.at[pid, 'price']:.1f}"

# ------------------------------------------------------------------ sidebar
st.sidebar.header("📥 Skuad Anda")
for k, v in {"ids": [], "sell": {}, "chips_used": [], "bank_in": 0.5, "ft_in": 1}.items():
    st.session_state.setdefault(k, v)

entry = st.sidebar.text_input("Entry ID FPL", placeholder="mis. 123456")
if st.sidebar.button("🔽 Import skuad, bank, FT & chip") and entry.strip():
    with st.spinner("Mengambil data entry…"):
        data, msg = D.fetch_entry(entry.strip(), bootstrap)
    if data:
        st.session_state.ids = data["player_ids"]
        st.session_state.sell = data["sell_prices"]
        st.session_state.chips_used = data["chips_used"]
        st.session_state.bank_in = float(data["bank"])
        st.session_state.ft_in = int(max(1, min(5, data["ft"])))
        st.sidebar.success(msg)
        st.sidebar.caption("Transfer yang sudah Anda buat untuk GW depan (pending) tidak terlihat di API publik — "
                           "sesuaikan skuad manual bila ada.")
    else:
        st.sidebar.error(msg)

bank = st.sidebar.number_input("Bank (£m)", 0.0, 30.0, key="bank_in", step=0.1)
ft = st.sidebar.number_input("Free transfer tersedia", 0, 5, key="ft_in")

avail_def = D.chips_available(st.session_state.chips_used, [])
st.sidebar.markdown("**Chip yang masih tersedia**")
s1 = st.sidebar.multiselect("Set 1 (wajib dipakai sebelum deadline GW19)", O.CHIPS,
                            default=[c for c in O.CHIPS if avail_def[(c, 1)]])
s2 = st.sidebar.multiselect("Set 2 (GW20–38)", O.CHIPS,
                            default=[c for c in O.CHIPS if avail_def[(c, 2)]])
chips_avail = {(c, 1): c in s1 for c in O.CHIPS} | {(c, 2): c in s2 for c in O.CHIPS}

st.sidebar.header("⚙️ Perencanaan")
max_h = last_gw - next_gw + 1
horizon = st.sidebar.slider("Horizon (GW ke depan)", 1, min(8, max_h), min(5, max_h))
chip_mode = st.sidebar.radio("Chip", ["Cari penempatan terbaik otomatis", "Jangan pakai chip"], index=0)
budget = st.sidebar.slider("Batas waktu pencarian chip (detik)", 30, 300, 120, step=15)
allow_hits = st.sidebar.checkbox("Izinkan transfer berbayar (−4)", True)
compare = st.sidebar.checkbox("Bandingkan skenario (tanpa transfer / tanpa hit)", True)

with st.sidebar.expander("Lanjutan"):
    max_tr = st.slider("Maks transfer per GW (non-Wildcard)", 1, 6, 5)
    gamma = st.slider("Diskon GW jauh (kepastian prediksi)", 0.80, 1.00, 0.93, 0.01)
    bench_w = st.slider("Bobot nilai bench (autosub)", 0.0, 0.3, 0.10, 0.01)
    ep_blend = st.slider("Campur dengan ep_next resmi FPL (GW depan)", 0.0, 0.4, 0.0, 0.05)
    st.caption("Biaya peluang chip (poin) — chip hanya dipakai jika keuntungannya melebihi ini; "
               "otomatis mengecil mendekati batas set (GW19 / GW38):")
    cc = {c: st.number_input(c, 0.0, 40.0, v, 1.0) for c, v in
          {"Wildcard": 8.0, "Free Hit": 10.0, "Bench Boost": 8.0, "Triple Captain": 5.0}.items()}
    force_out = st.multiselect("Anggap OUT (cedera/absen) — override", el.index.tolist(),
                               format_func=fmt_opt)
    force_fit = st.multiselect("Anggap FIT 100% — override", el.index.tolist(), format_func=fmt_opt)

gws = list(range(next_gw, next_gw + horizon))
overrides = {**{p: "fit" for p in force_fit}, **{p: "out" for p in force_out}}

# ------------------------------------------------------------------ skuad
st.subheader(f"📋 Skuad saat ini  (rencana mulai GW{next_gw})")
squad = st.multiselect("Pilih 15 pemain (atau import via Entry ID):", el.index.tolist(),
                       default=[i for i in st.session_state.ids if i in el.index],
                       format_func=fmt_opt)


def squad_problem(ids):
    if len(ids) != 15:
        return f"Pilih tepat 15 pemain (sekarang {len(ids)})."
    pc = el.loc[ids, "element_type"].value_counts().to_dict()
    if pc != {1: 2, 2: 5, 3: 5, 4: 3} and any(pc.get(p, 0) != n for p, n in {1: 2, 2: 5, 3: 5, 4: 3}.items()):
        return "Komposisi harus 2 GKP, 5 DEF, 5 MID, 3 FWD."
    if el.loc[ids, "team"].value_counts().max() > 3:
        return "Maksimal 3 pemain dari satu klub."
    return None


prob = squad_problem(squad)
if prob:
    st.info(prob)
    st.stop()

sell = {i: int(st.session_state.sell.get(i, el.at[i, "now_cost"])) for i in squad}
val = sum(sell.values()) / 10 + bank
st.caption(f"Nilai jual skuad + bank = £{val:.1f}m"
           + ("" if st.session_state.sell else "  •  (harga jual diasumsikan = harga kini; import Entry ID untuk harga jual akurat)"))

if not primary_button("🎯 JALANKAN OPTIMASI"):
    if "res" not in st.session_state:
        st.stop()
else:
    xp, det = M.predict_xp(ctx, gws, overrides, ep_blend)
    kw = dict(gamma=gamma, bench_w=bench_w, max_transfers=max_tr, chip_costs=cc)
    t0 = time.time()
    bar = st.progress(0.0, text="Memulai…")
    with st.spinner("Solver MILP bekerja…"):
        no_chip_only = chip_mode.startswith("Jangan")
        if no_chip_only:
            main = O.optimize(el, xp, squad, bank, ft, chips_avail, sell, allowed_chips=[],
                              allow_hits=allow_hits, time_limit=60, **kw)
        else:
            main = O.optimize_with_chips(el, xp, squad, bank, ft, chips_avail, sell,
                                         allow_hits=allow_hits, total_time=budget, final_time=60,
                                         progress=lambda f, t: bar.progress(f, text=t), **kw)
        scen = {}
        if compare:
            bar.progress(0.96, text="Skenario pembanding…")
            scen["Tidak transfer & tanpa chip"] = O.optimize(
                el, xp, squad, bank, ft, chips_avail, sell, allow_transfers=False,
                allowed_chips=[], time_limit=30, **kw)
            scen["Transfer TANPA hit, tanpa chip"] = O.optimize(
                el, xp, squad, bank, ft, chips_avail, sell, allowed_chips=[], allow_hits=False,
                time_limit=45, **kw)
            scen["Transfer + hit boleh, tanpa chip"] = O.optimize(
                el, xp, squad, bank, ft, chips_avail, sell, allowed_chips=[], allow_hits=True,
                time_limit=45, **kw)
    bar.empty()
    st.session_state["res"] = dict(main=main, scen=scen, xp=xp, det=det, gws=gws, squad=squad,
                                   ft=ft, bank=bank, sell=sell, secs=time.time() - t0)

R = st.session_state["res"]
main, scen, xp, det, gws, squad = R["main"], R["scen"], R["xp"], R["det"], R["gws"], R["squad"]
xp_sum = xp.sum(axis=1)
fxmap = {}
if len(det):
    for (p, g), grp in det.groupby(["element", "gw"]):
        fxmap[(p, g)] = ", ".join(f"{TEAM_SHORT[o]}({'H' if h else 'A'})"
                                  for o, h in zip(grp["opp"], grp["home"]))
fx_str = lambda p, g: fxmap.get((p, g), "—  (BGW)")
g0 = main["gws"][0]

tab_act, tab_plan, tab_pred, tab_diag, tab_info = st.tabs(
    [f"🎯 Aksi GW{gws[0]}", "🗓️ Rencana & Chip", "📈 Prediksi pemain", "🔬 Akurasi model", "ℹ️ Asumsi"])

# ------------------------------------------------------------------ TAB AKSI
with tab_act:
    base_total = scen["Tidak transfer & tanpa chip"]["total_xp"] if scen else None
    c = st.columns(4)
    c[0].metric("Chip untuk GW ini", g0["chip"] or "Tidak ada")
    c[1].metric("Transfer", f"{g0['n_transfers']}" + (" (gratis)" if g0["chip"] in ("Wildcard", "Free Hit") else ""),
                f"−{g0['hit']} poin hit" if g0["hit"] else "tanpa hit", delta_color="inverse")
    c[2].metric(f"xP GW{g0['gw']} (bersih)", f"{g0['xp_net']:.1f}")
    c[3].metric(f"Total xP {len(gws)} GW", f"{main['total_xp']:.1f}",
                f"{main['total_xp'] - base_total:+.1f} vs diam" if base_total is not None else None)

    if g0["chip"] == "Free Hit":
        st.warning("**Free Hit** — skuad di bawah hanya untuk GW ini; skuad permanen Anda kembali setelahnya.")
    elif g0["chip"]:
        st.success(f"**{g0['chip']}** direkomendasikan untuk dipakai SEKARANG.")

    st.markdown("#### 🔄 Transfer")
    if g0["chip"] == "Free Hit":
        out_ids, in_ids = sorted(set(squad) - set(g0["play_squad"])), sorted(set(g0["play_squad"]) - set(squad))
    else:
        out_ids, in_ids = g0["transfers_out"], g0["transfers_in"]
    if not in_ids:
        st.success("✅ **Tidak perlu transfer** — simpan free transfer (maks 5).")
    else:
        chip0 = g0["chip"]
        n_in = len(in_ids)
        paid = 0 if chip0 else max(0, n_in - g0["ft_available"])
        txt = f"{n_in} transfer"
        if chip0:
            txt += f" ({chip0}: tanpa hit)"
        else:
            txt += f": {n_in - paid} gratis" + (f", {paid} berbayar (−{4 * paid} poin)" if paid else "")
        sell_out = sum(R["sell"].get(i, int(el.at[i, "now_cost"])) for i in out_ids)
        buy_in = sum(int(el.at[i, "now_cost"]) for i in in_ids)
        st.write(f"{txt}.  Selisih biaya: £{(buy_in - sell_out) / 10:+.1f}m.")
        c1, c2 = st.columns(2)
        c1.markdown("🔴 **Jual**")
        with c1:
            show(pd.DataFrame([{"Pemain": lab(i), "Pos": el.at[i, "pos_name"],
                                "Harga jual": R["sell"].get(i, el.at[i, "now_cost"]) / 10,
                                f"xP {len(gws)} GW": round(xp_sum[i], 1),
                                f"xP GW{gws[0]}": round(xp.at[i, gws[0]], 1)} for i in out_ids]))
        c2.markdown("🟢 **Beli**")
        with c2:
            show(pd.DataFrame([{"Pemain": lab(i), "Pos": el.at[i, "pos_name"],
                                "Harga": el.at[i, "price"], f"xP {len(gws)} GW": round(xp_sum[i], 1),
                                f"xP GW{gws[0]}": round(xp.at[i, gws[0]], 1),
                                "Lawan": fx_str(i, gws[0]),
                                "Own%": el.at[i, "selected_by_percent"],
                                "Net transfer": int(el.at[i, "transfers_in_event"] - el.at[i, "transfers_out_event"])}
                               for i in in_ids]))
        st.caption("Net transfer = transfers in − out pekan ini (indikasi tekanan harga, bukan prediksi resmi).")

    st.markdown("#### 👑 Kapten & susunan")
    xi = g0["xi"]
    k1, k2 = st.columns(2)
    mult = 3 if g0["chip"] == "Triple Captain" else 2
    k1.metric("Kapten" + (" (3×)" if mult == 3 else ""), lab(g0["captain"]),
              f"{xp.at[g0['captain'], g0['gw']] * mult:.1f} xP")
    k2.metric("Vice-kapten", lab(g0["vice"]), f"{xp.at[g0['vice'], g0['gw']]:.1f} xP")
    cand = sorted(g0["play_squad"], key=lambda i: -xp.at[i, g0["gw"]])[:5]
    show(pd.DataFrame([{"Kandidat kapten": lab(i), "xP": round(xp.at[i, g0["gw"]], 2),
                        "Lawan": fx_str(i, g0["gw"]), "Own%": el.at[i, "selected_by_percent"]} for i in cand]))
    order = {1: 0, 2: 1, 3: 2, 4: 3}
    xi_sorted = sorted(xi, key=lambda i: (order[el.at[i, "element_type"]], -xp.at[i, g0["gw"]]))
    show(pd.DataFrame([{"Starting XI": lab(i) + (" (C)" if i == g0["captain"] else " (V)" if i == g0["vice"] else ""),
                        "Pos": el.at[i, "pos_name"], "Lawan": fx_str(i, g0["gw"]),
                        "xP": round(xp.at[i, g0["gw"]], 2),
                        "Status": el.at[i, "status"]} for i in xi_sorted]))
    st.markdown("**Bench (urutan autosub):** " + " → ".join(
        f"{lab(i)} {xp.at[i, g0['gw']]:.1f}" for i in g0["bench"]))

# ------------------------------------------------------------------ TAB RENCANA
with tab_plan:
    st.markdown("#### Rencana per GW")
    st.caption("Hanya baris pertama yang perlu dieksekusi sekarang; GW berikutnya adalah rencana awal "
               "dan akan diperbarui saat Anda menjalankan ulang tiap pekan.")
    rows = []
    for g in main["gws"]:
        ins = g["transfers_in"] if g["chip"] != "Free Hit" else sorted(set(g["play_squad"]) - set(squad))
        outs = g["transfers_out"] if g["chip"] != "Free Hit" else sorted(set(squad) - set(g["play_squad"]))
        rows.append({"GW": g["gw"], "Chip": g["chip"] or "—",
                     "Transfer": f"{len(ins)}" if g["chip"] != "Free Hit" else "FH",
                     "FT tersedia": g["ft_available"], "Hit": -g["hit"] if g["hit"] else 0,
                     "Keluar": ", ".join(lab(i) for i in outs) or "—",
                     "Masuk": ", ".join(lab(i) for i in ins) or "—",
                     "Kapten": lab(g["captain"]), "xP bersih": round(g["xp_net"], 1)})
    show(pd.DataFrame(rows))

    st.markdown("#### 🃏 Keputusan chip")
    plan = main.get("chip_plan", {g["chip"]: g["gw"] for g in main["gws"] if g["chip"]})
    gain = main.get("chip_gain_table")
    chip_rows = []
    for c_ in O.CHIPS:
        if not (chips_avail.get((c_, 1)) or chips_avail.get((c_, 2))):
            chip_rows.append({"Chip": c_, "Rekomendasi": "Sudah terpakai", "Keterangan": ""})
        elif c_ in plan:
            chip_rows.append({"Chip": c_, "Rekomendasi": f"Pakai di GW{plan[c_]}", "Keterangan": ""})
        else:
            chip_rows.append({"Chip": c_, "Rekomendasi": "TAHAN",
                              "Keterangan": "Tidak ada GW di horizon yang keuntungannya melebihi biaya peluang."})
    show(pd.DataFrame(chip_rows))
    if gain:
        st.markdown("**Keuntungan bersih tiap penempatan chip tunggal** (poin xP, setelah biaya peluang; "
                    "hijau = layak):")
        gt = pd.Series(gain).unstack(0).reindex(columns=[c_ for c_ in O.CHIPS if c_ in {k[0] for k in gain}])
        gt.index.name = "GW"
        try:
            st.dataframe(gt.style.format("{:+.1f}").highlight_max(axis=0, color="#2e7d32"), width="stretch")
        except Exception:
            st.dataframe(gt.round(1), use_container_width=True)
        st.caption(f"{main.get('n_solves', '?')} MILP dievaluasi dalam {main.get('search_seconds', 0):.0f} dtk. "
                   "Chip yang dikombinasikan (mis. Wildcard lalu Bench Boost) dievaluasi bersama, "
                   "jadi angka di atas bukan penjumlahan sederhana.")

    if scen:
        st.markdown("#### 📊 Nilai tiap keputusan (total xP bersih seluruh horizon)")
        base = scen["Tidak transfer & tanpa chip"]["total_xp"]
        comp = [("Tidak transfer & tanpa chip", base)] + [(k, v["total_xp"]) for k, v in scen.items()
                                                          if k != "Tidak transfer & tanpa chip"]
        comp.append(("**Rencana penuh (chip + transfer)**", main["total_xp"]))
        cdf = pd.DataFrame(comp, columns=["Skenario", "Total xP"])
        cdf["Δ vs diam"] = cdf["Total xP"] - base
        cdf["Total xP"], cdf["Δ vs diam"] = cdf["Total xP"].round(1), cdf["Δ vs diam"].round(1)
        show(cdf)
        nh = scen["Transfer TANPA hit, tanpa chip"]["total_xp"]
        wh = scen["Transfer + hit boleh, tanpa chip"]["total_xp"]
        if wh - nh > 0.5:
            st.info(f"💡 Transfer berbayar **layak**: mengizinkan hit menaikkan total xP bersih **{wh - nh:+.1f}** "
                    "(sudah dikurangi −4 per hit).")
        else:
            st.info("💡 Transfer berbayar **tidak layak** pada horizon ini — hit tidak menambah xP bersih.")

    with st.expander("Detail skuad & XI tiap GW"):
        for g in main["gws"]:
            st.markdown(f"**GW{g['gw']}** — {g['chip'] or 'tanpa chip'} • xP bersih {g['xp_net']:.1f}")
            show(pd.DataFrame([{"Pemain": lab(i) + (" (C)" if i == g["captain"] else ""),
                                "Pos": el.at[i, "pos_name"], "Lawan": fx_str(i, g["gw"]),
                                "xP": round(xp.at[i, g["gw"]], 2),
                                "Peran": "XI" if i in g["xi"] else "Bench"} for i in g["play_squad"]]))

# ------------------------------------------------------------------ TAB PREDIKSI
with tab_pred:
    f1, f2, f3 = st.columns([1, 1, 2])
    posf = f1.multiselect("Posisi", ["GKP", "DEF", "MID", "FWD"], default=["GKP", "DEF", "MID", "FWD"])
    maxp = f2.slider("Harga maks (£m)", 4.0, 15.0, 15.0, 0.5)
    q = f3.text_input("Cari nama")
    T = el[["web_name", "team_short", "pos_name", "price", "status", "selected_by_percent"]].copy()
    for g in gws:
        T[f"GW{g}"] = xp[g].round(2)
    T[f"Σ{len(gws)}GW"] = xp_sum.round(1)
    T["xP/£m"] = (xp_sum / el["price"]).round(2)
    T = T[T["pos_name"].isin(posf) & (T["price"] <= maxp)]
    if q:
        T = T[T["web_name"].str.contains(q, case=False, na=False)]
    T = T.sort_values(f"Σ{len(gws)}GW", ascending=False).head(150).rename(columns={
        "web_name": "Pemain", "team_short": "Klub", "pos_name": "Pos", "price": "Harga",
        "status": "Status", "selected_by_percent": "Own%"})
    show(T.reset_index(drop=True))

    st.markdown("**Bedah xP satu pemain (GW depan):**")
    pick = st.selectbox("Pemain", T.index.tolist() if len(T) else [], format_func=lambda i: fmt_opt(i)) \
        if len(T) else None
    if pick is not None and len(det):
        d = det[(det["element"] == pick) & (det["gw"] == gws[0])]
        cols = ["main", "gol", "assist", "cs", "kebobolan", "saves", "bonus", "defcon", "kartu",
                "struct", "gbm", "xP"]
        if len(d):
            out = d[cols].round(2).rename(columns={
                "main": "Main", "gol": "Gol", "assist": "Assist", "cs": "Clean sheet",
                "kebobolan": "Kebobolan", "saves": "Saves", "bonus": "Bonus", "defcon": "DefCon",
                "kartu": "Kartu", "struct": "xP struktural", "gbm": "xP GBM", "xP": "xP final"})
            out.insert(0, "Lawan", [f"{TEAM_SHORT[o]}({'H' if h else 'A'})" for o, h in zip(d["opp"], d["home"])])
            out.insert(1, "xMenit", d["exp_min"].round(0).values)
            show(out)
        else:
            st.write("Tidak ada pertandingan di GW ini (BGW).")

# ------------------------------------------------------------------ TAB DIAGNOSTIK
with tab_diag:
    bt = ctx.backtest
    st.markdown(f"Model dilatih dari **{len(ctx.hist):,} baris pemain-pertandingan**. "
                f"Bobot GBM pada blend final (hasil backtest): **α = {ctx.alpha:.1f}** "
                f"(sisanya model struktural).")
    if bt.get("ok"):
        st.markdown(f"**Walk-forward backtest** — tiap GW diprediksi hanya dengan data sebelumnya "
                    f"(GW {bt['rounds'][0]}–{bt['rounds'][-1]}, {bt['n_rows']:,} baris pemain dengan xMenit ≥ 30). "
                    "Lebih rendah = lebih baik untuk RMSE/MAE.")
        t = bt["table"].round(3)
        t.index.name = "Model"
        st.dataframe(t, width="stretch")
        st.markdown("**Kalibrasi** — rata-rata prediksi vs aktual per desil prediksi "
                    "(garis ideal: kedua kolom sama):")
        st.line_chart(bt["calibration"][["prediksi", "aktual"]])
        st.caption("Catatan jujur: poin FPL per-pertandingan sangat acak (satu gol = 4–6 poin). Korelasi per-baris "
                   "~0.2–0.35 sudah tergolong baik; kekuatan model ada pada *ranking* dan nilai rata-rata "
                   "multi-GW, bukan menebak hasil satu laga.")
    else:
        st.info(bt.get("note", "Backtest belum tersedia."))

# ------------------------------------------------------------------ TAB ASUMSI
with tab_info:
    st.markdown(f"""
**Aturan yang dimodelkan (FPL 2026/27):** skuad 15 / £100m / maks 3 per klub • free transfer bergulir maks 5,
transfer berlebih −4 • 2 set chip (WC, FH, BB, TC): set 1 wajib dipakai sebelum deadline GW19, set 2 GW20–38 •
1 chip per GW • tidak ada tambahan FT bulan Desember • harga jual = beli + ½ profit (dibulatkan ke bawah).

**Model xP** — dibangun dari komponen poin FPL: poin main (P(main), P(≥60′)), gol & assist (xG/xA per-90 dengan
shrinkage ke musim lalu & rata-rata posisi, diskalakan oleh kekuatan serang tim vs lawan), clean sheet
(`exp(−λ kebobolan)` dari model kekuatan tim berbasis xG), poin kebobolan (distribusi Poisson), saves, bonus,
DefCon, kartu. DGW = penjumlahan dua laga, BGW = 0. Lapisan GBM mengoreksi bias sisa dan bobotnya ditentukan
backtest.

**Ketersediaan** dibaca dari status/`chance_of_playing`/teks *news* ("Expected back 12 Dec"). Pemain cedera tanpa
tanggal kembali diasumsikan pulih bertahap. Gunakan override di sidebar bila Anda tahu kabar terbaru.

**Optimizer** memecahkan satu MILP multi-GW (HiGHS) untuk transfer, XI, kapten, dan tiap chip. Penempatan chip
dicari lewat evaluasi MILP berkunci + coordinate-ascent + pair-move, lalu solve akhir presisi.

**Batasan yang perlu Anda tahu**
- Perubahan harga ke depan **tidak** dimodelkan (harga jual/beli dianggap tetap sepanjang horizon).
- Risiko rotasi kompetisi Eropa & berita konferensi pers tidak ada di API → cek manual / pakai override.
- Prediksi makin tidak pasti untuk GW jauh; karena itu ada diskon (`Diskon GW jauh`) dan hanya aksi GW pertama
  yang perlu dieksekusi.
- Transfer pending untuk GW depan tidak terlihat di API publik.
- Biaya peluang chip adalah *heuristik* (nilai chip di luar horizon tidak diketahui); atur di panel Lanjutan.
""")