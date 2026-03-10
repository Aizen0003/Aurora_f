"""
Build submission zip with exact PS-required folder structure:

SpeakX_AURORA_<TeamName>.zip
├── iteration_0_before_learning/
│   ├── company_north_star.json
│   ├── feature_goal_map.json
│   ├── allowed_tone_hook_matrix.json
│   ├── user_segments.csv
│   ├── segment_goals.csv
│   ├── communication_themes.csv
│   ├── message_templates.csv
│   ├── timing_recommendations.csv
│   └── user_notification_schedule.csv
├── iteration_1_after_learning/
│   ├── user_segments.csv (if changed)
│   ├── message_templates.csv (updated)
│   ├── timing_recommendations.csv (updated)
│   └── user_notification_schedule.csv (updated)
├── experiment_results.csv
├── learning_delta_report.csv
├── codebase/ (complete runnable code)
└── README.txt (<=500 words)
"""

import zipfile
import os
import sys

TEAM_NAME = ""
ZIP_NAME = f"SpeakX_AURORA_{TEAM_NAME}.zip" if TEAM_NAME else "SpeakX_AURORA.zip"
BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(BASE, "data", "output")

# Iteration 0 files (from data/output/)
ITER0_FILES = {
    "company_north_star.json": os.path.join(OUTPUT, "company_north_star.json"),
    "feature_goal_map.json": os.path.join(OUTPUT, "feature_goal_map.json"),
    "allowed_tone_hook_matrix.json": os.path.join(OUTPUT, "allowed_tone_hook_matrix.json"),
    "user_segments.csv": os.path.join(OUTPUT, "user_segments.csv"),
    "segment_goals.csv": os.path.join(OUTPUT, "segment_goals.csv"),
    "communication_themes.csv": os.path.join(OUTPUT, "communication_themes.csv"),
    "message_templates.csv": os.path.join(OUTPUT, "message_templates.csv"),
    "timing_recommendations.csv": os.path.join(OUTPUT, "timing_recommendations.csv"),
    "user_notification_schedule.csv": os.path.join(OUTPUT, "user_notification_schedule.csv"),
}

# Iteration 1 files (improved versions)
ITER1_FILES = {
    "user_segments.csv": os.path.join(OUTPUT, "user_segments.csv"),  # same (not changed in iter1)
    "message_templates.csv": os.path.join(OUTPUT, "message_templates_improved.csv"),
    "timing_recommendations.csv": os.path.join(OUTPUT, "timing_recommendations_improved.csv"),
    "user_notification_schedule.csv": os.path.join(OUTPUT, "user_notification_schedule_improved.csv"),
}

# Root-level files
ROOT_FILES = {
    "experiment_results.csv": os.path.join(BASE, "data", "sample", "experiment_results_sample.csv"),
    "learning_delta_report.csv": os.path.join(OUTPUT, "learning_delta_report.csv"),
    "README.txt": os.path.join(BASE, "README.txt"),
}

# Codebase files to include (relative paths from BASE)
CODEBASE_INCLUDE = [
    "main.py",
    "requirements.txt",
    "config/config.yaml",
    "src/__init__.py",
    "src/llm_utils.py",
    "src/communication/__init__.py",
    "src/communication/nlp_template_optimizer.py",
    "src/communication/schedule_generator.py",
    "src/communication/template_generator.py",
    "src/communication/theme_engine.py",
    "src/communication/timing_optimizer.py",
    "src/intelligence/__init__.py",
    "src/intelligence/data_ingestion.py",
    "src/intelligence/goal_builder.py",
    "src/intelligence/ml_propensity_models.py",
    "src/intelligence/segmentation.py",
    "src/knowledge_bank/__init__.py",
    "src/knowledge_bank/kb_engine.py",
    "src/learning/__init__.py",
    "src/learning/delta_reporter.py",
    "src/learning/learning_engine.py",
    "src/learning/multi_armed_bandit.py",
    "src/learning/performance_classifier.py",
    "src/learning/statistical_testing.py",
    "src/utils/__init__.py",
    "src/utils/experiment_generator.py",
    "src/utils/metrics.py",
    "src/utils/validation.py",
]

def build_zip():
    zip_path = os.path.join(BASE, ZIP_NAME)
    missing = []

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # 1. iteration_0_before_learning/
        print("Adding iteration_0_before_learning/...")
        for name, path in ITER0_FILES.items():
            if os.path.exists(path):
                zf.write(path, f"iteration_0_before_learning/{name}")
                print(f"  + {name}")
            else:
                missing.append(f"iter0/{name}")
                print(f"  MISSING: {name}")

        # 2. iteration_1_after_learning/
        print("\nAdding iteration_1_after_learning/...")
        for name, path in ITER1_FILES.items():
            if os.path.exists(path):
                zf.write(path, f"iteration_1_after_learning/{name}")
                print(f"  + {name}")
            else:
                missing.append(f"iter1/{name}")
                print(f"  MISSING: {name}")

        # 3. Root-level files
        print("\nAdding root files...")
        for name, path in ROOT_FILES.items():
            if os.path.exists(path):
                zf.write(path, name)
                print(f"  + {name}")
            else:
                missing.append(name)
                print(f"  MISSING: {name}")

        # 4. codebase/
        print("\nAdding codebase/...")
        for rel_path in CODEBASE_INCLUDE:
            full_path = os.path.join(BASE, rel_path)
            if os.path.exists(full_path):
                zf.write(full_path, f"codebase/{rel_path}")
                print(f"  + {rel_path}")
            else:
                missing.append(f"codebase/{rel_path}")
                print(f"  MISSING: {rel_path}")

    # Summary
    print(f"\n{'='*60}")
    if missing:
        print(f"WARNING: {len(missing)} files missing:")
        for m in missing:
            print(f"  - {m}")
    else:
        print("ALL FILES PRESENT")

    size_kb = os.path.getsize(zip_path) / 1024
    print(f"\nZip created: {ZIP_NAME} ({size_kb:.0f} KB)")
    print(f"Location: {zip_path}")


if __name__ == "__main__":
    build_zip()
