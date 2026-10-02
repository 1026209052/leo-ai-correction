# -*- coding: utf-8 -*-
"""τ_max（零危险漏报边界）的聚簇 bootstrap 区间 + 有效样本量。

输入（本机已有）：
  figures/data/pareto_samples.csv   列: sample_idx, peak, sgp4_rmse, eta, is_dangerous_if_not_intervened
  figures/data/sample_metadata.csv  列: sample_idx, norad_id, is_non_maneuver

先按表8 的口径自检（危险漏报个数 32/288、介入率 62.6/58.2/37.3/15.9%），
再做以卫星为簇的 bootstrap（重采样卫星，取回该星的样本，重算 τ_max 与漏报率）。
"""
import csv
import os
import random
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D_WARN = 15.0          # 碰撞预警门限 (km)
N_EVAL = 1404          # 全体验证样本


def load():
    meta = {}
    with open(BASE + r"\figures\data\sample_metadata.csv", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            meta[int(r["sample_idx"])] = (int(r["norad_id"]), r["is_non_maneuver"].strip() == "True")
    rows = []
    with open(BASE + r"\figures\data\pareto_samples.csv", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            i = int(r["sample_idx"])
            rows.append({
                "idx": i,
                "peak": float(r["peak"]),
                "rmse": float(r["sgp4_rmse"]),
                "danger": int(r["is_dangerous_if_not_intervened"]) == 1,
                "norad": meta.get(i, (None, None))[0],
                "nonman": meta.get(i, (None, None))[1],
            })
    return rows


def tau_max(samples):
    """零危险漏报边界：第一个“危险却未介入”样本的峰值。"""
    cand = [s["peak"] for s in samples if s["danger"]]
    return min(cand) if cand else float("inf")


def miss_count(samples, tau):
    return sum(1 for s in samples if s["danger"] and s["peak"] < tau)


def main():
    rows = load()
    print(f"载入样本 {len(rows)} 条；含 norad 的 {sum(1 for r in rows if r['norad'])} 条")
    man = [r for r in rows if not r["nonman"]]
    print(f"机动样本（!is_non_maneuver） = {len(man)}  （论文口径 912）")
    print(f"危险样本（is_dangerous=1）   = {sum(1 for r in rows if r['danger'])}")
    print(f"峰值≥20 km 的样本            = {sum(1 for r in rows if r['peak'] >= 20)}  （论文 817）")

    print("\n=== 表8 口径自检 ===")
    for tau in (15, 20, 30, 50):
        inter = sum(1 for r in rows if r["peak"] >= tau) / len(rows) * 100
        m = miss_count(man, tau)
        print(f"  τ={tau:>2} km: 介入率={inter:5.1f}%   危险漏报={m:3d} 个 ({m/max(1,len(man))*100:5.2f}% of 机动样本)")
    tmax = tau_max(rows)
    print(f"\n  τ_max = min{{peak | 危险}} = {tmax:.4f} km    （论文 24.89）")
    dng = sorted([(r["peak"], r["idx"], r["norad"]) for r in rows if r["danger"]])
    print(f"  最小的 6 个“危险”样本峰值: " + ", ".join(f"{p:.2f}(#{i}/sat{n})" for p, i, n in dng[:6]))
    print(f"  峰值 < 30 km 的危险样本 = {sum(1 for p, _, _ in dng if p < 30)} 个")
    owner = dng[0][2]
    print(f"  定义 τ_max 的那个样本属于卫星 {owner}"
          f"（该卫星共 {sum(1 for r in rows if r['norad'] == owner)} 个验证样本）")

    # ---- 留一卫星（LOO）：边界由几颗卫星支撑 ----
    # ---- 聚簇 bootstrap（以卫星为单位）----
    bysat = defaultdict(list)
    for r in rows:
        bysat[r["norad"]].append(r)
    loo = []
    for c in bysat:
        sub = [s for s in rows if s["norad"] != c]
        loo.append((tau_max(sub), c))
    loo.sort()
    print(f"\n=== 留一颗卫星（LOO, 共 {len(loo)} 次）===")
    print(f"  τ_max 全部取值 ≥ {loo[0][0]:.2f} km（去掉卫星 {loo[0][1]} 时最小）")
    print(f"  去掉定义它的那颗（{owner}）后 τ_max 升至 {[v for v, c in loo if c == owner][0]:.2f} km")
    print("  → 以 24.89 km 为界的危险样本只有 1 个：该经验边界由**单个样本**决定")

    clusters = [k for k in bysat if k is not None]
    print(f"\n  簇（卫星）数 = {len(clusters)}")
    rnd = random.Random(20260925)
    B = 10000
    taus, miss20, miss15 = [], [], []
    for _ in range(B):
        pick = [rnd.choice(clusters) for _ in clusters]
        sample = [s for c in pick for s in bysat[c]]
        taus.append(tau_max(sample))
        manb = [s for s in sample if not s["nonman"]]
        miss20.append(miss_count(manb, 20) / max(1, len(manb)))
        miss15.append(miss_count(manb, 15) / max(1, len(manb)))
    taus.sort()

    def ci(v, lo=2.5, hi=97.5):
        v = sorted(v)
        return v[int(len(v) * lo / 100)], v[int(len(v) * hi / 100)]

    print(f"\n=== 聚簇 bootstrap (B={B}, 以卫星为簇) ===")
    print(f"  τ_max: 点估计 {tmax:.2f} km, 95% CI [{ci(taus)[0]:.2f}, {ci(taus)[1]:.2f}] km, "
          f"中位 {taus[B//2]:.2f}")
    print(f"  P(τ_max < 20 km) = {sum(1 for x in taus if x < 20)/B*100:.1f}%")
    print(f"  危险漏率@τ=20: {sum(miss20)/B*100:.3f}%  95% CI [{ci(miss20)[0]*100:.3f}, {ci(miss20)[1]*100:.3f}]%")
    print(f"  危险漏率@τ=15: {sum(miss15)/B*100:.3f}%  95% CI [{ci(miss15)[0]*100:.3f}, {ci(miss15)[1]*100:.3f}]%")

    # ---- 簇内相关 → 设计效应（以大机动组的 P(η<-20%) 为例）----
    print("\n=== 设计效应（以严重退化比例为例）===")
    try:
        big = {}
        with open(BASE + r"\figures\data\per_sample_eta_lstm.csv", encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                big[int(r["sample_idx"])] = int(r["norad_id"])
        print(f"  per_sample_eta_lstm.csv 提供 {len(big)} 条 (sample_idx→norad)")
        eta = {}
        with open(BASE + r"\figures\data\fig2_cdf_all.csv", encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                eta[int(r["sample_idx"])] = float(r["eta_mse"])
        pairs = [(v, eta[k]) for k, v in big.items() if k in eta]
        print(f"  可用 (norad, eta_mse) 对 = {len(pairs)}")
        if pairs:
            n = len(pairs)
            pbar = sum(1 for _, e in pairs if e < -20) / n
            g = defaultdict(list)
            for c, e in pairs:
                g[c].append(1 if e < -20 else 0)
            m = len(g)
            mbar = n / m
            # ICC（one-way ANOVA 估计）
            grand = pbar
            ssb = sum(len(v) * (sum(v) / len(v) - grand) ** 2 for v in g.values())
            ssw = sum(sum((x - sum(v) / len(v)) ** 2 for x in v) for v in g.values())
            msb, msw = ssb / (m - 1), ssw / (n - m)
            icc = (msb - msw) / (msb + (mbar - 1) * msw) if (msb + (mbar - 1) * msw) else 0
            deff = 1 + (mbar - 1) * max(0.0, icc)
            print(f"  簇数 m={m}, 平均簇大小={mbar:.1f}, P(η<-20%)={pbar*100:.2f}%")
            print(f"  ICC={icc:.3f}  设计效应 deff={deff:.2f}  有效样本量 n_eff = {n/deff:.0f} (名义 {n})")
    except Exception as e:  # noqa: BLE001
        print("  ERR", e)


if __name__ == "__main__":
    main()
