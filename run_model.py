"""Chạy lại toàn bộ luồng Modeling (Bài tập 4) bằng một lệnh.

Luồng: cafeland_missing_outliers_cleaned.csv -> bảng modeling -> split 80/20
       -> tuning 4 mô hình (5-fold CV) -> chọn mô hình (one-standard-error)
       -> đánh giá trên test -> lưu báo cáo + models/best_model.joblib

Cách dùng:
    python run_model.py                  # chạy đầy đủ, ghi đè reports/ và models/
    python run_model.py --plots          # thêm learning curve + residual plot
    python run_model.py --output-root out_test   # ghi ra thư mục khác, không đụng file gốc

Logic mô hình khớp notebooks/07 và 08 (cùng random_state=42, cùng lưới tham số).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, learning_curve

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.preprocess import (  # noqa: E402
    MODEL_FEATURES,
    TARGET,
    build_model_pipeline,
    build_modeling_table,
    build_preprocessor,
    correlation_report,
    feature_contract,
    save_split_tables,
    split_model_data,
    training_target_quantile_diagnostics,
    vif_report,
)

RANDOM_STATE = 42
SOURCE_FILE = PROJECT_ROOT / "data" / "processed" / "cafeland_missing_outliers_cleaned.csv"
# Thứ tự ưu tiên khi nhiều mô hình có sai số tương đương (đơn giản -> phức tạp)
SIMPLICITY_ORDER = ["LinearRegression", "Ridge", "Ridge_log1p", "RandomForest"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-root", type=Path, default=PROJECT_ROOT,
                   help="Thư mục gốc để ghi data/processed, reports, models (mặc định: gốc repo).")
    p.add_argument("--plots", action="store_true", help="Vẽ learning curve và residual plot.")
    p.add_argument("--n-jobs", type=int, default=-1, help="Số tiến trình cho GridSearchCV (mặc định: tất cả).")
    return p.parse_args()


def build_model_specs(n_jobs: int) -> dict:
    alpha_grid = np.logspace(-3, 3, 13)
    return {
        "LinearRegression": {"estimator": build_model_pipeline(LinearRegression()), "params": None},
        "Ridge": {"estimator": build_model_pipeline(Ridge()),
                  "params": {"model__alpha": alpha_grid}},
        "Ridge_log1p": {"estimator": build_model_pipeline(Ridge(), log_target=True),
                        "params": {"regressor__model__alpha": alpha_grid}},
        "RandomForest": {
            "estimator": build_model_pipeline(RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)),
            "params": {
                "model__n_estimators": [200, 400],
                "model__max_features": ["sqrt", 1.0],
                "model__min_samples_leaf": [1, 3],
            },
        },
    }


def main() -> None:
    args = parse_args()
    out = args.output_root.resolve()
    data_dir, reports_dir = out / "data" / "processed", out / "reports"
    figures_dir, models_dir = reports_dir / "figures", out / "models"
    for d in (data_dir, reports_dir, figures_dir, models_dir):
        d.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"scikit-learn {sklearn.__version__} | ghi kết quả vào: {out}")

    # ---------- 1. Bảng modeling + split (chống leakage) ----------
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Không thấy {SOURCE_FILE}. Hãy chạy `python run_pipeline.py` và các notebook 03-04 trước.")
    source = pd.read_csv(SOURCE_FILE, encoding="utf-8-sig")
    table, audit = build_modeling_table(source)
    split = split_model_data(table, test_size=0.20, random_state=RANDOM_STATE)
    save_split_tables(table, split, data_dir)
    print(f"[1/4] Modeling base: {len(table):,} tin | train {len(split.X_train):,} | test {len(split.X_test):,}")

    audit.to_csv(reports_dir / "preprocessing_scope_audit.csv", index=False, encoding="utf-8-sig")
    feature_contract().to_csv(reports_dir / "feature_contract.csv", index=False, encoding="utf-8-sig")
    correlation_report(table.loc[split.train_index]).to_csv(
        reports_dir / "correlation_train.csv", encoding="utf-8-sig")
    quantile_report, _ = training_target_quantile_diagnostics(split.y_train)
    quantile_report.to_csv(reports_dir / "target_quantile_train_diagnostic.csv", index=False, encoding="utf-8-sig")
    try:
        fitted_pre = build_preprocessor().fit(split.X_train)
        vif_report(fitted_pre, split.X_train).to_csv(reports_dir / "vif_train.csv", index=False, encoding="utf-8-sig")
    except ImportError:
        print("  (bỏ qua VIF: chưa cài statsmodels)")

    X_train, y_train, X_test, y_test = split.X_train, split.y_train, split.X_test, split.y_test

    # ---------- 2. Tuning ----------
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fitted, searches = {}, {}
    for name, spec in build_model_specs(args.n_jobs).items():
        print(f"[2/4] Đang fit: {name}")
        if spec["params"] is None:
            fitted[name] = spec["estimator"].fit(X_train, y_train)
        else:
            search = GridSearchCV(spec["estimator"], spec["params"], scoring="neg_mean_absolute_error",
                                  cv=cv, refit=True, n_jobs=args.n_jobs, error_score="raise")
            search.fit(X_train, y_train)
            searches[name], fitted[name] = search, search.best_estimator_
            print(f"      CV MAE tốt nhất: {-search.best_score_:,.0f} VNĐ/m² | {search.best_params_}")

    # ---------- 3. CV + chọn mô hình TRƯỚC khi mở test ----------
    scoring = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"}
    cv_rows, train_rows = [], []
    for name, est in fitted.items():
        s = cross_validate(est, X_train, y_train, cv=cv, scoring=scoring, n_jobs=args.n_jobs)
        mae, rmse, r2 = -s["test_mae"], -s["test_rmse"], s["test_r2"]
        cv_rows.append({"model": name,
                        "cv_mae_mean": mae.mean(), "cv_mae_std": mae.std(ddof=1),
                        "cv_mae_se": mae.std(ddof=1) / np.sqrt(len(mae)),
                        "cv_rmse_mean": rmse.mean(), "cv_rmse_std": rmse.std(ddof=1),
                        "cv_r2_mean": r2.mean(), "cv_r2_std": r2.std(ddof=1)})
        pred = est.predict(X_train)
        train_rows.append({"model": name, "train_mae": mean_absolute_error(y_train, pred),
                           "train_rmse": mean_squared_error(y_train, pred) ** 0.5,
                           "train_r2": r2_score(y_train, pred)})
    cv_summary = pd.DataFrame(cv_rows).set_index("model")
    train_summary = pd.DataFrame(train_rows).set_index("model")

    best_cv = cv_summary["cv_mae_mean"].idxmin()
    threshold = cv_summary.loc[best_cv, "cv_mae_mean"] + cv_summary.loc[best_cv, "cv_mae_se"]
    eligible = cv_summary.index[cv_summary["cv_mae_mean"] <= threshold].tolist()
    selected = next(n for n in SIMPLICITY_ORDER if n in eligible)
    print(f"[3/4] CV MAE thấp nhất: {best_cv} | mô hình được chọn (1-SE, ưu tiên đơn giản): {selected}")

    # ---------- 4. Test + lưu ----------
    test_rows = []
    for name, est in fitted.items():
        pred = est.predict(X_test)
        test_rows.append({"model": name, "test_mae": mean_absolute_error(y_test, pred),
                          "test_rmse": mean_squared_error(y_test, pred) ** 0.5,
                          "test_r2": r2_score(y_test, pred)})
    comparison = train_summary.join(cv_summary).join(pd.DataFrame(test_rows).set_index("model"))
    comparison["train_cv_mae_gap"] = comparison["cv_mae_mean"] - comparison["train_mae"]
    comparison["train_cv_mae_gap_pct"] = comparison["train_cv_mae_gap"] / comparison["cv_mae_mean"] * 100
    comparison["selected_by_cv"] = comparison.index == selected
    comparison.reset_index().to_csv(reports_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")

    hp = [{"model": "LinearRegression", "best_params": "{}"}]
    hp += [{"model": n, "best_params": json.dumps(s.best_params_, ensure_ascii=False, default=str),
            "best_cv_mae_vnd_m2": -s.best_score_} for n, s in searches.items()]
    pd.DataFrame(hp).to_csv(reports_dir / "model_hyperparameters.csv", index=False, encoding="utf-8-sig")

    joblib.dump(fitted[selected], models_dir / "best_model.joblib")

    if args.plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sizes, tr, va = learning_curve(fitted[selected], X_train, y_train, cv=cv, n_jobs=args.n_jobs,
                                       train_sizes=np.linspace(0.2, 1.0, 5), scoring="neg_mean_absolute_error")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(sizes, -tr.mean(axis=1) / 1e6, marker="o", label="Train MAE")
        ax.plot(sizes, -va.mean(axis=1) / 1e6, marker="o", label="Validation MAE (5-fold)")
        ax.set(title=f"Learning curve — {selected}", xlabel="Số mẫu train", ylabel="MAE (triệu VNĐ/m²)")
        ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
        fig.savefig(figures_dir / "learning_curve_best_model.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

        pred = fitted[selected].predict(X_test)
        res = y_test.to_numpy() - pred
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].scatter(pred / 1e6, res / 1e6, alpha=0.55, edgecolors="none")
        axes[0].axhline(0, color="crimson", linestyle="--")
        axes[0].set(title="Residual và giá dự đoán (test)", xlabel="Giá dự đoán (triệu VNĐ/m²)", ylabel="Residual (triệu VNĐ/m²)")
        axes[1].hist(res / 1e6, bins=35, alpha=0.85, edgecolor="white")
        axes[1].set(title="Phân bố residual (test)", xlabel="Residual (triệu VNĐ/m²)", ylabel="Số tin")
        fig.suptitle(f"Residual analysis — {selected}"); fig.tight_layout()
        fig.savefig(figures_dir / "residual_analysis_best_model.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    best = comparison.loc[selected]
    print(f"[4/4] Test: MAE {best['test_mae']/1e6:.1f} tr/m² | RMSE {best['test_rmse']/1e6:.1f} tr/m² | R² {best['test_r2']:.3f}")
    print(f"Hoàn tất sau {time.time() - t0:.0f}s. Mô hình: {models_dir / 'best_model.joblib'}")


if __name__ == "__main__":
    main()
