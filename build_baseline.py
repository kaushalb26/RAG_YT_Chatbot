import json
import math
import statistics
from collections import defaultdict

with open("baseline1.json", "r", encoding="utf-8") as f:
    b1 = json.load(f)
with open("baseline2.json", "r", encoding="utf-8") as f:
    b2 = json.load(f)

runs = [b1, b2]
metric_names = list(b1["average_scores"].keys())

# 1. Compute Mean and Standard Deviation (Noise Threshold) across runs
avg_scores = {}
std_devs = {}
metric_registry = {}

for m in metric_names:
    run_vals = [r["average_scores"][m] for r in runs if m in r["average_scores"]]
    mean_val = round(statistics.mean(run_vals), 4)
    # Sample std dev across runs (minimum floor of 0.02 to suppress tiny judge jitter)
    sd_val = round(statistics.stdev(run_vals), 4) if len(run_vals) > 1 else 0.0
    noise_threshold = round(max(sd_val, 0.02), 4)

    avg_scores[m] = mean_val
    std_devs[m] = sd_val
    metric_registry[m] = {
        "direction": "higher_is_better",
        "baseline_mean": mean_val,
        "run_std_dev": sd_val,
        "noise_threshold": noise_threshold,
    }

# Also register operational latency (lower is better)
p50_vals = [r["operational_metrics"]["p50_latency_sec"] for r in runs]
p95_vals = [r["operational_metrics"]["p95_latency_sec"] for r in runs]
metric_registry["p95_latency_sec"] = {
    "direction": "lower_is_better",
    "baseline_mean": round(statistics.mean(p95_vals), 3),
    "run_std_dev": round(statistics.stdev(p95_vals), 3),
    "noise_threshold": round(max(statistics.stdev(p95_vals), 0.25), 3),
}

# 2. Average Per-Test-Case Scores
b2_by_idx = {c["index"]: c for c in b2["test_cases"]}
merged_cases = []
weak_by_type = defaultdict(lambda: defaultdict(list))

for c1 in b1["test_cases"]:
    idx = c1["index"]
    c2 = b2_by_idx[idx]
    merged_metrics = {}
    for m in metric_names:
        s1 = c1["metrics"][m]["score"]
        s2 = c2["metrics"][m]["score"]
        mean_s = round((s1 + s2) / 2, 4)
        merged_metrics[m] = {
            "score": mean_s,
            "run1_score": s1,
            "run2_score": s2,
            "passed": mean_s >= 0.7,
        }
        weak_by_type[c1.get("question_type", "general")][m].append(mean_s)

    merged_cases.append(
        {
            "index": idx,
            "video_id": c1.get("video_id"),
            "question_type": c1.get("question_type"),
            "input": c1["input"],
            "actual_output": c1["actual_output"],
            "expected_output": c1["expected_output"],
            "latency_sec": round((c1["latency_sec"] + c2["latency_sec"]) / 2, 3),
            "metrics": merged_metrics,
        }
    )

final_baseline = {
    "config": b1["config"],
    "num_runs_averaged": len(runs),
    "average_scores": avg_scores,
    "run_std_dev": std_devs,
    "operational_metrics": {
        "p50_latency_sec": round(statistics.mean(p50_vals), 3),
        "p95_latency_sec": round(statistics.mean(p95_vals), 3),
    },
    "test_cases": merged_cases,
}

with open("baseline.json", "w", encoding="utf-8") as f:
    json.dump(final_baseline, f, indent=2, ensure_ascii=False)

with open("metric_registry.json", "w", encoding="utf-8") as f:
    json.dump(metric_registry, f, indent=2, ensure_ascii=False)

print("=== FINAL AVERAGED BASELINE (baseline.json) ===")
for m in metric_names:
    print(f"  {m:<24}: {avg_scores[m]:.4f}  (std_dev: ±{std_devs[m]:.4f})")

print("\n=== BREAKDOWN BY QUESTION TYPE ===")
for qtype, m_dict in weak_by_type.items():
    summary = ", ".join(
        f"{m}: {statistics.mean(vals):.3f}" for m, vals in m_dict.items()
    )
    print(f"  [{qtype}] -> {summary}")
