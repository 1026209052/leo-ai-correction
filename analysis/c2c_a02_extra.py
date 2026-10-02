# -*- coding: utf-8 -*-
"""α=0.2 配置下另外两组论文需要的数字（纯 CSV，本机可跑）：
   (1) LSTM vs LSTM+软掩码 的 P(η<-20%) / P(η<-10%) 聚簇配对 bootstrap（供附录B）
   (2) “两种方法都退化的交集样本”上退化幅度中位数（供 §5.1）
"""
import csv
import os
import random
import statistics
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
D = r"E:\paper-tianjinUnivercity\figures\data"
B = 10000
SEED = 20260919
T = 20.0


def load_eta(path, cols):
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows.append({c: (int(r[c]) if c in ("sample_idx", "norad_id") else float(r[c])) for c in cols})
    return rows


def cluster_boot(pairs, fn, B=B, seed=SEED):
    """pairs: list of (norad, a, b) —— 配对、以卫星为簇。返回 (diff, lo, hi, p)。"""
    by = {}
    for g, a, b in pairs:
        by.setdefault(g, []).append((a, b))
    keys = list(by)
    rng = random.Random(seed)
    obs = fn([b for _, (_, b) in [(k, v) for k in keys for v in by[k]]]) - \
        fn([a for _, (a, _) in [(k, v) for k in keys for v in by[k]]])
    d = []
    for _ in range(B):
        A, Bv = [], []
        for _ in range(len(keys)):
            for a, b in by[keys[rng.randrange(len(keys))]]:
                A.append(a)
                Bv.append(b)
        d.append(fn(Bv) - fn(A))
    d.sort()
    lo, hi = d[int(0.025 * B)], d[int(0.975 * B)]
    # 双侧 p（按 0 的位置）
    neg = sum(1 for x in d if x <= 0) / B
    pos = sum(1 for x in d if x >= 0) / B
    p = 2 * min(neg, pos)
    return obs, lo, hi, min(1.0, p)


def main():
    # ---------- (1) LSTM 侧的严重退化/中轻度退化差异 ----------
    rows = load_eta(os.path.join(D, "per_sample_eta_lstm.csv"),
                    ["sample_idx", "norad_id", "eta_lstm", "eta_lstm_softmask"])
    pairs = [(str(r["norad_id"]), r["eta_lstm"], r["eta_lstm_softmask"]) for r in rows]
    print(f"LSTM 配对样本 n={len(pairs)}, 簇数={len(set(p[0] for p in pairs))}")
    for name, thr in (("P(η<-20%)", -20.0), ("P(η<-10%)", -10.0), ("P(η<-30%)", -30.0)):
        fn = lambda v, t=thr: 100.0 * sum(1 for x in v if x < t) / len(v)
        obs, lo, hi, p = cluster_boot(pairs, fn)
        print(f"  {name:12s} Δ={obs:+7.2f} pp  95%CI [{lo:7.2f}, {hi:7.2f}]  p={p:.3f}")
    fn = lambda v: 100.0 * sum(1 for x in v if x > 0) / len(v)
    obs, lo, hi, p = cluster_boot(pairs, fn)
    print(f"  {'优于SGP4比例':12s} Δ={obs:+7.2f} pp  95%CI [{lo:7.2f}, {hi:7.2f}]  p={p:.3f}")

    # ---------- (2) 交集样本上的退化幅度中位数 ----------
    print()
    allrows = load_eta(os.path.join(D, "fig2_cdf_all.csv"),
                       ["sample_idx", "peak", "sgp4_rmse", "eta_mse", "eta_soft"])
    big = [r for r in allrows if r["peak"] >= T]
    both = [r for r in big if r["eta_mse"] < 0 and r["eta_soft"] < 0]
    print(f"大机动组 n={len(big)}；两者都退化(交集) n={len(both)}")
    if both:
        m1 = statistics.median([r["eta_mse"] for r in both])
        m2 = statistics.median([r["eta_soft"] for r in both])
        print(f"  交集上退化幅度中位数: 标准MSE {m1:+.2f}%  →  软掩码 {m2:+.2f}%")
    only_mse = [r for r in big if r["eta_mse"] < 0 and r["eta_soft"] >= 0]
    only_soft = [r for r in big if r["eta_soft"] < 0 and r["eta_mse"] >= 0]
    print(f"  仅MSE退化 {len(only_mse)} 个；仅软掩码退化 {len(only_soft)} 个")


if __name__ == "__main__":
    main()
