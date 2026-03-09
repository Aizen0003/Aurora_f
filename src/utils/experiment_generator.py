"""
Experiment Results Generator — Synthesizes realistic experiment data
from Iteration 0 outputs for demo and testing.

Generates experiment_results_sample.csv that is fully compatible with
the templates, segments, themes, and timing produced by iteration0.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


def generate_experiment_results(
    output_dir: str = "data/output",
    sample_dir: str = "data/sample",
    samples_per_template: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic experiment results from iteration0 outputs.

    Reads templates, timing, and themes CSVs produced by iteration0 and
    creates a realistic experiment_results_sample.csv that iteration1
    can consume directly.

    Args:
        output_dir: Directory containing iteration0 outputs.
        sample_dir: Directory to save generated experiment results.
        samples_per_template: How many experiment rows per template.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with synthetic experiment results.
    """
    rng = np.random.RandomState(seed)

    # Load iteration0 outputs
    templates = pd.read_csv(f"{output_dir}/message_templates.csv")
    timing = pd.read_csv(f"{output_dir}/timing_recommendations.csv")

    # Get unique time windows from timing recommendations
    time_windows = timing["time_window"].unique().tolist()
    if not time_windows:
        time_windows = ["evening", "mid_morning"]

    # Sample a representative subset of templates (one per segment × lifecycle × goal × theme)
    group_cols = ["segment_id", "lifecycle_stage", "goal", "theme"]
    available = [c for c in group_cols if c in templates.columns]
    if available:
        sampled = templates.groupby(available).first().reset_index()
    else:
        sampled = templates.head(50)

    rows = []
    for _, tpl in sampled.iterrows():
        for _ in range(samples_per_template):
            # Assign a random time window
            window = rng.choice(time_windows)

            # Simulate realistic metrics
            total_sends = int(rng.choice([200, 300, 500, 800, 1000]))
            base_ctr = rng.uniform(0.02, 0.25)
            total_opens = int(total_sends * base_ctr)
            engagement_rate = rng.uniform(0.10, 0.60)
            total_engagements = int(total_opens * engagement_rate)
            ctr = total_opens / total_sends if total_sends else 0
            uninstall_rate = rng.uniform(0.001, 0.04)

            rows.append({
                "template_id": tpl["template_id"],
                "segment_id": tpl["segment_id"],
                "lifecycle_stage": tpl["lifecycle_stage"],
                "goal": tpl.get("goal", "activation"),
                "theme": tpl.get("theme", "accomplishment"),
                "notification_window": window,
                "total_sends": total_sends,
                "total_opens": total_opens,
                "total_engagements": total_engagements,
                "ctr": round(ctr, 4),
                "engagement_rate": round(engagement_rate, 4),
                "uninstall_rate": round(uninstall_rate, 4),
            })

    df = pd.DataFrame(rows)

    # Save
    Path(sample_dir).mkdir(parents=True, exist_ok=True)
    out_path = f"{sample_dir}/experiment_results_sample.csv"
    df.to_csv(out_path, index=False)
    print(f"   [OK] Generated {len(df)} synthetic experiment rows → {out_path}")

    return df
