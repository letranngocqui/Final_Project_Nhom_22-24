"""Leakage-safe preprocessing utilities for Bài tập 4.

Owner: Chu Minh Tuấn (Model Data Lead)

Design principles
-----------------
1. Build the regression target only from *observed original* price/area
   (``gia_goc`` / ``dien_tich_goc``), never from imputed values.
2. Remove duplicate listings by ``link_nguon`` before splitting.
3. Respect the Project Charter scope: sale listings only; area 10–2,000 m².
4. Split train/test before fitting imputation, scaling, or one-hot encoding.
5. Put every learned preprocessing step inside a scikit-learn Pipeline so it
   is re-fitted inside each cross-validation fold downstream.
6. Never expose ``gia``/``gia_goc`` to X when y = price / area.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "price_per_m2"
ID_COLUMN = "link_nguon"
NUMERIC_FEATURES = ["dien_tich_goc", "so_phong_ngu"]
CATEGORICAL_FEATURES = ["khu_vuc_nhom", "loai_bds"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Meaningful, common reference groups for linear models.  Using the dominant
# groups as references also avoids inflated VIF caused by a very rare baseline.
CATEGORY_REFERENCE_LEVELS = ["TP.HCM (lõi cũ)", "Nhà ở/Nhà phố"]

# Project Charter rule for valid sale-listing area.
AREA_MIN_M2 = 10.0
AREA_MAX_M2 = 2000.0

# Lists inherited from the group's Bài tập 3 feature-engineering notebook.
BINH_DUONG_WARDS = {
    "Bình Dương", "Chánh Hiệp", "Thủ Dầu Một", "Phú Lợi", "Đông Hòa", "Dĩ An",
    "Tân Đông Hiệp", "Thuận An", "Thuận Giao", "An Phú", "Bình Hòa", "Lái Thiêu",
    "Vĩnh Tân", "Bình Cơ", "Tân Uyên", "Tân Khánh", "Tân Hiệp", "Tây Nam",
    "Long Nguyên", "Bến Cát", "Hòa Lợi", "Bắc Tân Uyên", "Thường Tân", "Phú Giáo",
    "Trừ Văn Thố", "Bàu Bàng", "Minh Thạnh", "Long Hòa", "Dầu Tiếng", "Thanh An",
    "Thới Hòa", "Phú An", "Chánh Phú Hòa", "An Long", "Phước Thành", "Phước Hòa",
}

BA_RIA_VUNG_TAU_WARDS = {
    "Vũng Tàu", "Tam Thắng", "Rạch Dừa", "Phước Thắng", "Long Sơn", "Bà Rịa",
    "Long Hương", "Tam Long", "Tân Thành", "Tân Phước", "Tân Hải", "Châu Pha",
    "Ngãi Giao", "Bình Giã", "Kim Long", "Châu Đức", "Xuân Sơn", "Nghĩa Thành",
    "Hồ Tràm", "Xuyên Mộc", "Hòa Hội", "Bàu Lâm", "Bình Châu", "Hòa Hiệp",
    "Đất Đỏ", "Long Hải", "Long Điền", "Phước Hải", "Côn Đảo",
}

EXPLICIT_RENTAL_PATTERN = re.compile(r"\bcho\s*thuê\b|\bthuê\b", flags=re.IGNORECASE)


@dataclass(frozen=True)
class SplitData:
    """Container for one reproducible hold-out split."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_index: pd.Index
    test_index: pd.Index


def _coerce_bool(series: pd.Series) -> pd.Series:
    """Convert common bool/string/0-1 representations to Boolean safely."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
        .fillna(False)
        .astype(bool)
    )


def infer_region_group(vi_tri: object) -> str:
    """Map the first locality token to the three project region groups."""
    ward = str(vi_tri).split(",")[0].strip()
    if ward in BINH_DUONG_WARDS:
        return "Bình Dương (mới sáp nhập)"
    if ward in BA_RIA_VUNG_TAU_WARDS:
        return "Bà Rịa - Vũng Tàu (mới sáp nhập)"
    return "TP.HCM (lõi cũ)"


def infer_property_type(tieu_de: object) -> str:
    """Derive the compact property-type feature used in Bài tập 3."""
    text = str(tieu_de).lower()
    if re.search(r"chung cư|căn hộ|cc\b|apartment", text):
        return "Căn hộ/Chung cư"
    if re.search(r"biệt thự|villa", text):
        return "Biệt thự"
    if re.search(r"\bđất\b|đất nền|thổ cư|đất thổ", text):
        return "Đất nền"
    if re.search(r"nhà|nha pho|nhà phố|nhà mặt tiền|nhà hẻm", text):
        return "Nhà ở/Nhà phố"
    return "Khác"


def _audit_row(step: str, before: int, after: int, reason: str) -> dict[str, object]:
    removed = before - after
    return {
        "step": step,
        "rows_before": int(before),
        "rows_after": int(after),
        "rows_removed": int(removed),
        "removed_pct_of_step": round((removed / before * 100.0) if before else 0.0, 3),
        "reason": reason,
    }


def build_modeling_table(
    source: pd.DataFrame,
    *,
    area_min_m2: float = AREA_MIN_M2,
    area_max_m2: float = AREA_MAX_M2,
    exclude_explicit_rentals: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the leakage-safe modeling base table and an auditable row log.

    The function deliberately starts from ``cafeland_missing_outliers_cleaned``
    rather than the already one-hot-encoded ``cafeland_features.csv``.  This is
    necessary because encoding on the full dataset before the hold-out split
    would let test-set category information leak into preprocessing.

    No quantile/IQR rule is learned from the target here.  Extreme target values
    are retained for diagnostics; any target-distribution rule must be fitted on
    training data only.
    """
    required = {
        "tieu_de", "vi_tri", ID_COLUMN, "gia_goc", "dien_tich_goc",
        "gia_thieu", "dien_tich_thieu", "so_phong_ngu",
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc cho modeling: {missing}")

    df = source.copy()
    df["gia_goc"] = pd.to_numeric(df["gia_goc"], errors="coerce")
    df["dien_tich_goc"] = pd.to_numeric(df["dien_tich_goc"], errors="coerce")
    df["so_phong_ngu"] = pd.to_numeric(df["so_phong_ngu"], errors="coerce")
    df["gia_thieu"] = _coerce_bool(df["gia_thieu"])
    df["dien_tich_thieu"] = _coerce_bool(df["dien_tich_thieu"])

    audit: list[dict[str, object]] = []

    before = len(df)
    df = df.drop_duplicates(subset=[ID_COLUMN], keep="first").copy()
    audit.append(_audit_row(
        "01_deduplicate_link",
        before,
        len(df),
        "Mỗi link_nguon là một tin; giữ bản ghi đầu tiên để train/test không chứa cùng một tin.",
    ))

    # Target eligibility: observed originals only.  This is the most important
    # anti-leakage/data-integrity rule for this project.
    before = len(df)
    observed_mask = (
        ~df["gia_thieu"]
        & ~df["dien_tich_thieu"]
        & df["gia_goc"].notna()
        & df["dien_tich_goc"].notna()
        & (df["gia_goc"] > 0)
        & (df["dien_tich_goc"] > 0)
    )
    df = df.loc[observed_mask].copy()
    audit.append(_audit_row(
        "02_observed_target_only",
        before,
        len(df),
        "Target dùng gia_goc/dien_tich_goc quan sát thực; loại mọi dòng từng thiếu/được impute giá hoặc diện tích.",
    ))

    before = len(df)
    df = df.loc[df["dien_tich_goc"].between(area_min_m2, area_max_m2, inclusive="both")].copy()
    audit.append(_audit_row(
        "03_area_scope_10_2000",
        before,
        len(df),
        f"Theo Project Charter: chỉ giữ diện tích từ {area_min_m2:g} đến {area_max_m2:g} m².",
    ))

    if exclude_explicit_rentals:
        before = len(df)
        rental_mask = df["tieu_de"].fillna("").str.contains(EXPLICIT_RENTAL_PATTERN, regex=True)
        df = df.loc[~rental_mask].copy()
        audit.append(_audit_row(
            "04_sale_scope",
            before,
            len(df),
            "Project Charter đặt tin cho thuê ngoài phạm vi; loại các tiêu đề có từ khóa thuê/cho thuê.",
        ))

    # Target is built ONLY after all observed-target checks.
    df[TARGET] = df["gia_goc"] / df["dien_tich_goc"]
    finite_target = np.isfinite(df[TARGET]) & (df[TARGET] > 0)
    before = len(df)
    df = df.loc[finite_target].copy()
    audit.append(_audit_row(
        "05_positive_finite_target",
        before,
        len(df),
        "Giữ price_per_m2 dương và hữu hạn; không học ngưỡng phân vị từ toàn bộ dataset.",
    ))

    df["khu_vuc_nhom"] = df["vi_tri"].map(infer_region_group)
    df["loai_bds"] = df["tieu_de"].map(infer_property_type)

    # A transparent source id lets us prove there is no overlap across split.
    keep = [
        ID_COLUMN, "tieu_de", "vi_tri", "gia_goc", "dien_tich_goc",
        "so_phong_ngu", "khu_vuc_nhom", "loai_bds", TARGET,
    ]
    df = df[keep].reset_index(drop=True)

    if df[ID_COLUMN].duplicated().any():
        raise AssertionError("Vẫn còn duplicate link_nguon sau bước khử trùng.")
    if df[TARGET].isna().any() or (df[TARGET] <= 0).any():
        raise AssertionError("Target không hợp lệ sau khi dựng modeling table.")

    return df, pd.DataFrame(audit)


def split_model_data(
    modeling_table: pd.DataFrame,
    *,
    test_size: float = 0.20,
    random_state: int = 42,
) -> SplitData:
    """Create a reproducible 80/20 hold-out split BEFORE learned preprocessing."""
    missing = sorted(set(MODEL_FEATURES + [TARGET, ID_COLUMN]) - set(modeling_table.columns))
    if missing:
        raise ValueError(f"Modeling table thiếu cột: {missing}")

    indices = modeling_table.index.to_numpy()
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    train_index = pd.Index(train_idx)
    test_index = pd.Index(test_idx)

    X_train = modeling_table.loc[train_index, MODEL_FEATURES].copy()
    X_test = modeling_table.loc[test_index, MODEL_FEATURES].copy()
    y_train = modeling_table.loc[train_index, TARGET].copy()
    y_test = modeling_table.loc[test_index, TARGET].copy()

    train_links = set(modeling_table.loc[train_index, ID_COLUMN])
    test_links = set(modeling_table.loc[test_index, ID_COLUMN])
    if train_links & test_links:
        raise AssertionError("Data leakage: có link_nguon xuất hiện ở cả train và test.")
    if "gia" in X_train.columns or "gia_goc" in X_train.columns or TARGET in X_train.columns:
        raise AssertionError("Target leakage: giá/target không được phép xuất hiện trong X.")

    return SplitData(X_train, X_test, y_train, y_test, train_index, test_index)


def build_preprocessor() -> ColumnTransformer:
    """Return the unfitted preprocessing graph to place inside every model Pipeline.

    Numerical path:
      median imputation (learned on train/fold only) + missing indicator + scaling.
    Categorical path:
      most-frequent imputation + one-hot encoding with meaningful reference groups.
    """
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            drop=CATEGORY_REFERENCE_LEVELS,
            sparse_output=False,
        )),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_model_pipeline(estimator, *, log_target: bool = False):
    """Compose leakage-safe preprocessing with any downstream estimator.

    Parameters
    ----------
    estimator:
        A scikit-learn regressor supplied by the Modeling Lead.
    log_target:
        If True, fit on log1p(price_per_m2) and automatically invert predictions
        back to VNĐ/m².  Metrics should then be computed on the returned original
        scale predictions.
    """
    regressor = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", estimator),
    ])
    if log_target:
        return TransformedTargetRegressor(
            regressor=regressor,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=True,
        )
    return regressor


def training_target_quantile_diagnostics(
    y_train: pd.Series,
    *,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> tuple[pd.DataFrame, pd.Series]:
    """Fit target quantile thresholds on TRAIN only for audit, not automatic deletion.

    The Project Charter mentions a 1%–99% price/m² rule.  To avoid test leakage,
    thresholds are estimated only from y_train.  The returned mask identifies
    train extremes; this helper intentionally does *not* filter the hold-out set
    and should not be fitted once on all data before cross-validation.
    """
    if not (0 <= lower_q < upper_q <= 1):
        raise ValueError("Cần 0 <= lower_q < upper_q <= 1.")
    lower = float(y_train.quantile(lower_q))
    upper = float(y_train.quantile(upper_q))
    mask = (y_train < lower) | (y_train > upper)
    report = pd.DataFrame([{
        "lower_quantile": lower_q,
        "upper_quantile": upper_q,
        "lower_bound_vnd_m2": lower,
        "upper_bound_vnd_m2": upper,
        "train_rows": int(len(y_train)),
        "flagged_train_rows": int(mask.sum()),
        "flagged_train_pct": round(float(mask.mean() * 100), 3),
        "policy": "diagnostic_only_train_fitted; hold-out test untouched",
    }])
    return report, mask


def transformed_feature_names(fitted_preprocessor: ColumnTransformer) -> list[str]:
    """Get output feature names from a fitted preprocessor."""
    return list(fitted_preprocessor.get_feature_names_out())


def correlation_report(modeling_train: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation on numeric training columns only."""
    cols = ["dien_tich_goc", "so_phong_ngu", TARGET]
    return modeling_train[cols].corr(numeric_only=True)


def vif_report(fitted_preprocessor: ColumnTransformer, X_train: pd.DataFrame) -> pd.DataFrame:
    """Compute VIF on the TRAIN-fitted transformed design matrix.

    statsmodels is imported lazily so preprocessing itself does not depend on it.
    """
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.tools.tools import add_constant
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Cần statsmodels để tính VIF.") from exc

    matrix = fitted_preprocessor.transform(X_train)
    names = transformed_feature_names(fitted_preprocessor)
    frame = pd.DataFrame(matrix, columns=names, index=X_train.index)
    design = add_constant(frame, has_constant="add")

    rows: list[dict[str, object]] = []
    for i, name in enumerate(design.columns):
        if name == "const":
            continue
        value = float(variance_inflation_factor(design.to_numpy(), i))
        rows.append({
            "feature": name,
            "VIF": value,
            "assessment": (
                "thấp" if value < 5 else
                "cần theo dõi" if value < 10 else
                "đa cộng tuyến cao"
            ),
        })
    return pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)


def feature_contract() -> pd.DataFrame:
    """Document allowed and forbidden model inputs for hand-off."""
    rows = [
        ("dien_tich_goc", "ALLOWED", "Numeric", "Diện tích quan sát thực; dùng làm feature và mẫu số target, không phải giá."),
        ("so_phong_ngu", "ALLOWED", "Numeric", "Có thiếu; median + missing indicator được fit trong train/fold."),
        ("khu_vuc_nhom", "ALLOWED", "Categorical", "Nhóm địa lý compact từ Bài tập 3; OneHotEncoder fit trong train/fold."),
        ("loai_bds", "ALLOWED", "Categorical", "Loại BĐS suy ra từ tiêu đề; OneHotEncoder fit trong train/fold."),
        ("gia", "FORBIDDEN", "Leakage", "Giá sau impute; target price_per_m2 = giá/diện tích nên đưa giá vào X gây target leakage."),
        ("gia_goc", "FORBIDDEN", "Leakage", "Tử số trực tiếp của target; tuyệt đối không đưa vào X."),
        ("price_per_m2", "FORBIDDEN", "Target", "Biến mục tiêu y, không phải feature."),
        ("gia_thieu", "FORBIDDEN", "Constant after scope", "Sau observed-target filter cờ này luôn False, không mang thông tin."),
        ("dien_tich_thieu", "FORBIDDEN", "Constant after scope", "Sau observed-target filter cờ này luôn False, không mang thông tin."),
        ("tieu_de", "FORBIDDEN_CURRENT", "Free text", "Chỉ dùng để suy ra loai_bds; chưa có NLP pipeline được kiểm chứng cho Bài 4."),
        ("vi_tri", "FORBIDDEN_CURRENT", "High-cardinality text", "Dùng để suy ra khu_vuc_nhom; không one-hot trực tiếp 159+ nhãn."),
        ("link_nguon", "ID_ONLY", "Identifier", "Chỉ dùng khử trùng/chứng minh split không chồng lặp; không dùng để học."),
    ]
    return pd.DataFrame(rows, columns=["column", "status", "type", "reason"])


def save_split_tables(
    modeling_table: pd.DataFrame,
    split: SplitData,
    output_dir: str | Path,
) -> tuple[Path, Path, Path]:
    """Save reproducible raw train/test hand-off files (preprocessing still unfitted)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_path = output_dir / "modeling_base.csv"
    train_path = output_dir / "model_train_raw.csv"
    test_path = output_dir / "model_test_raw.csv"

    modeling_table.to_csv(base_path, index=False, encoding="utf-8-sig")
    modeling_table.loc[split.train_index].to_csv(train_path, index=False, encoding="utf-8-sig")
    modeling_table.loc[split.test_index].to_csv(test_path, index=False, encoding="utf-8-sig")
    return base_path, train_path, test_path
