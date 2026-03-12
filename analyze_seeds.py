#!/usr/bin/env python3
"""分析 new 项目 normals 下各种子 (222, 2222, 22222) 的表现，看哪个种子更有利。"""
import os
import re
import pandas as pd
import numpy as np

normals_dir = "results/mosi/emt-dlfr/run_once/results/normals"
# 只分析 new 项目：Params 里含 fusion_layers': 5（new 特有）
NEW_MARKER = "fusion_layers': 5"
# detail 里可能既有旧种子 111/1111/11111 也有新种子 222/2222/22222，都统计
SEEDS = [111, 1111, 11111, 222, 2222, 22222]
# 指标：越高越好用 1，越低越好用 -1（后面算加权时用）
METRICS = [
    ("Has0_acc_2", 1),
    ("Has0_F1_score", 1),
    ("Non0_acc_2", 1),
    ("Non0_F1_score", 1),
    ("Mult_acc_5", 1),
    ("Mult_acc_7", 1),
    ("MAE", -1),
    ("Corr", 1),
    ("Loss", -1),
    ("Loss(pred_m)", -1),
]

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isdir(normals_dir):
        print("normals 目录不存在:", normals_dir)
        return

    # 收集所有 new 项目的、逐种子的数值行
    by_seed = {s: [] for s in SEEDS}
    for mr in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        path = os.path.join(normals_dir, f"mosi-regression-{mr}-detail.csv")
        if not os.path.isfile(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print("读取出错", path, e)
            continue
        if "Seed" not in df.columns or "Params" not in df.columns:
            continue
        for _, row in df.iterrows():
            seed_val = row.get("Seed")
            try:
                seed_int = int(seed_val)
            except (TypeError, ValueError):
                continue
            if seed_int not in SEEDS:
                continue
            params = str(row.get("Params", ""))
            if NEW_MARKER not in params:
                continue
            rec = {"missing_rate": mr}
            for name, _ in METRICS:
                if name not in df.columns:
                    continue
                v = row[name]
                if isinstance(v, (int, float)) and np.isfinite(v):
                    rec[name] = float(v)
                else:
                    try:
                        rec[name] = float(v)
                    except (TypeError, ValueError):
                        pass
            if len(rec) > 1:
                by_seed[seed_int].append(rec)

    for s in SEEDS:
        print(f"Seed {s}: 共 {len(by_seed[s])} 条 new 项目记录 (各缺失率×多次运行)")
    print()

    # 按种子汇总：各指标均值
    print("=" * 60)
    print("new 项目：各种子在 normals 中的平均指标（仅 fusion_layers=5 的记录）")
    print("=" * 60)
    summary = {}
    for s in SEEDS:
        rows = by_seed[s]
        if not rows:
            summary[s] = {}
            continue
        d = pd.DataFrame(rows)
        cols = [m[0] for m in METRICS if m[0] in d.columns]
        summary[s] = d[cols].mean().to_dict() if cols else {}

    # 打印表格：每指标一行，三列种子
    metric_names = [m[0] for m in METRICS]
    for name in metric_names:
        line = f"  {name:20s}  "
        for s in SEEDS:
            val = summary.get(s, {}).get(name)
            line += f"  {val:8.2f}" if val is not None and np.isfinite(val) else "    -   "
        print(line)

    # 综合：对“越高越好”的指标取平均，“越低越好”取负再平均，然后看哪个种子综合分最高
    print()
    print("综合得分（归一化后越高越好）：")
    scores = {}
    for s in SEEDS:
        if not summary.get(s):
            scores[s] = np.nan
            continue
        total = 0
        n = 0
        for name, direction in METRICS:
            val = summary[s].get(name)
            if val is None or not np.isfinite(val):
                continue
            if direction == -1:
                val = -val
            total += val
            n += 1
        scores[s] = total / n if n else np.nan
    for s in SEEDS:
        print(f"  Seed {s}: {scores[s]:.2f}")
    best = max(SEEDS, key=lambda s: scores.get(s) or -1e9)
    new_seeds = [222, 2222, 22222]
    best_new = max(new_seeds, key=lambda s: scores.get(s) or -1e9)
    print(f"\n结论（基于当前 normals 中 new 项目记录）：")
    print(f"  · 综合得分最高：Seed {best}")
    print(f"  · 若只看新种子 222/2222/22222：Seed {best_new} 最佳（综合 {scores.get(best_new):.2f}）")

if __name__ == "__main__":
    main()
