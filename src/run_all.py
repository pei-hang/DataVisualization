from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    "01_load_clean_merge.py",
    "02_construct_variables.py",
    "03_tfp_estimation.py",
    "04_descriptive_statistics.py",
    "05_baseline_fe.py",
    "06_xgboost_shap.py",
    "07_mechanism_tests.py",
    "08_heterogeneity_tests.py",
    "09_robustness_tests.py",
    "10_visualizations.py",
]
OPTIONAL_SCRIPTS = {"06_xgboost_shap.py"}


def main() -> None:
    for script in SCRIPTS:
        path = SRC_DIR / script
        print(f"\n===== RUN {script} =====")
        try:
            subprocess.run([sys.executable, str(path)], check=True)
        except subprocess.CalledProcessError:
            if script in OPTIONAL_SCRIPTS:
                print(f"{script} 运行失败，已跳过并继续后续流程。请查看报错信息和 outputs/reports/model_interpretation.md。")
                continue
            raise
    print("\n全部流程已完成。")


if __name__ == "__main__":
    main()
