import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    candidate = PROJECT_ROOT / path
    return candidate if candidate.exists() or not Path(p).exists() else Path(p)


def load_config(config_path: str = "configs/config.yaml") -> dict[str, Any]:
    path = resolve_path(config_path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_domain_filters(
    df: pd.DataFrame,
    allowed_compounds: list[str],
    outlier_threshold_pct: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Sekwencyjna filtracja danych F1 wraz z rejestrowaniem lejka odrzuceń.
    """
    funnel: list[dict[str, Any]] = []
    initial_count = len(df)

    def log_step(step_name: str, current_df: pd.DataFrame, prev_count: int) -> int:
        curr_count = len(current_df)
        dropped = prev_count - curr_count
        pct_remaining = (curr_count / initial_count) * 100
        funnel.append(
            {
                "step": step_name,
                "remaining_rows": curr_count,
                "dropped_rows": dropped,
                "pct_of_raw": round(pct_remaining, 2),
            }
        )
        logger.info(
            f"[{step_name}] Pozostało: {curr_count:,} | Odrzucono: {dropped:,} ({pct_remaining:.2f}% surowych)"
        )
        return curr_count

    prev_count = log_step("0. Surowy zbiór (Raw)", df, initial_count)

    # 1. Braki w zmiennej celu i kluczowej cesze zużycia opon
    df = df[df["LapTimeSeconds"].notna() & df["TyreLife"].notna()].copy()
    prev_count = log_step("1. Usunięcie NaN w LapTimeSeconds i TyreLife", df, prev_count)

    # 2. Wyłącznie zielona flaga (TrackStatus == '1')
    df = df[df["TrackStatus"].astype(str).str.strip() == "1"].copy()
    prev_count = log_step("2. Filtracja TrackStatus == '1' (zielona flaga)", df, prev_count)

    # 3. Jakość pomiaru transpondera i brak anulowania okrążenia (track limits)
    is_accurate = df["IsAccurate"].fillna(False).astype(bool)
    is_deleted = df["Deleted"].fillna(False).astype(bool)
    df = df[is_accurate & (~is_deleted)].copy()
    prev_count = log_step("3. Filtracja IsAccurate == True oraz Deleted == False", df, prev_count)

    # 4. Wyłącznie suche warunki i dozwolone mieszanki slicków
    is_rainfall = df["Rainfall"].fillna(False).astype(bool)
    df = df[(~is_rainfall) & (df["Compound"].isin(allowed_compounds))].copy()
    prev_count = log_step(
        f"4. Warunki suche (Rainfall == False) i mieszanki {allowed_compounds}",
        df,
        prev_count,
    )

    # 5. Wykluczenie okrążeń zjazdowych i wyjazdowych z alei serwisowej (in-lap / out-lap)
    df = df[df["PitInTimeSeconds"].isna() & df["PitOutTimeSeconds"].isna()].copy()
    prev_count = log_step("5. Usunięcie okrążeń pit-stop (PitIn / PitOut)", df, prev_count)

    # 6. Usunięcie skrajnie wolnych anomalii wg progu 1.07 * mediana per RaceID
    race_medians = df.groupby("RaceID")["LapTimeSeconds"].transform("median")
    upper_bound = outlier_threshold_pct * race_medians
    lower_bound = 30.0
    df = df[(df["LapTimeSeconds"] >= lower_bound) & (df["LapTimeSeconds"] <= upper_bound)].copy()
    log_step(
        f"6. Filtracja outlierów (<= {outlier_threshold_pct} * mediana per RaceID)",
        df,
        prev_count,
    )

    return df, funnel


def engineer_features(
    df: pd.DataFrame,
    train_years: list[int],
) -> pd.DataFrame:
    """
    Inżynieria cech domenowych F1 bez wycieku danych (data leakage).
    """
    df = df.copy()

    # 1. Znormalizowane przybliżenie obciążenia paliwem [0, 1]
    df["FuelProxy"] = (1.0 - (df["LapNumber"] - 1.0) / df["TotalLaps"]).clip(lower=0.0, upper=1.0)

    # 2. Nieliniowe zużycie opony (tyre cliff)
    df["TyreLifeSq"] = df["TyreLife"] ** 2

    # 3. Interakcje fizyczne (opona x temperatura asfaltu, paliwo x opona)
    df["TyreTempInteraction"] = df["TyreLife"] * df["TrackTemp"]
    df["FuelTyreInteraction"] = df["FuelProxy"] * df["TyreLife"]

    # 4. Gradient termiczny asfalt - powietrze (wskaźnik nasłonecznienia)
    df["TrackAirTempDiff"] = df["TrackTemp"] - df["AirTemp"]

    # 5. Kodowanie porządkowe twardości mieszanki oraz świeżości kompletu
    hardness_map = {"SOFT": 1, "MEDIUM": 2, "HARD": 3}
    df["CompoundHardness"] = df["Compound"].map(hardness_map).astype("int64")
    df["FreshTyreInt"] = df["FreshTyre"].fillna(False).astype("int64")

    # 6. Relatywna wydajność bolidu w kwalifikacjach względem Pole Position w dany weekend
    pole_time_per_race = df.groupby("RaceID")["TeamQualiBest"].transform("min")
    df["TeamQualiGapToPole"] = df["TeamQualiBest"] - pole_time_per_race
    df["TeamQualiGapPct"] = (df["TeamQualiGapToPole"] / pole_time_per_race) * 100.0

    # 7. Względny postęp zużycia w danym stincie [0, 1]
    max_stint_life = df.groupby(["RaceID", "Driver", "Stint"])["TyreLife"].transform("max")
    df["StintProgress"] = (df["TyreLife"] / max_stint_life).clip(lower=0.0, upper=1.0)

    # 8. Bazowa mediana czasu okrążenia per tor liczona WYŁĄCZNIE na zbiorze treningowym
    train_mask = df["Year"].isin(train_years)
    circuit_baseline = (
        df.loc[train_mask]
        .groupby("Circuit")["LapTimeSeconds"]
        .median()
        .to_dict()
    )
    df["CircuitMedianLap"] = df["Circuit"].map(circuit_baseline)

    if df["CircuitMedianLap"].isna().any():
        missing_circuits = df.loc[df["CircuitMedianLap"].isna(), "Circuit"].unique().tolist()
        raise ValueError(
            f"Brak bazowej mediany treningowej dla torów: {missing_circuits}. "
            "Sprawdź, czy wszystkie tory występują w train_years."
        )

    return df


def select_and_order_columns(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    meta_cols = [
        "Year",
        "RaceID",
    ]

    feature_cols = [
        "Circuit",
        "CircuitMedianLap",
        "LapNumber",
        "FuelProxy",
        "Compound",
        "CompoundHardness",
        "TyreLife",
        "TyreLifeSq",
        "StintProgress",
        "FreshTyreInt",
        "AirTemp",
        "TrackTemp",
        "TrackAirTempDiff",
        "Humidity",
        "TyreTempInteraction",
        "FuelTyreInteraction",
        "TeamQualiBest",
        "TeamQualiGapToPole",
        "TeamQualiGapPct",
    ]

    selected_cols = meta_cols + feature_cols + [target_col]
    missing = [c for c in selected_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Brakujące oczekiwane kolumny po preprocessingu: {missing}")

    return df[selected_cols].reset_index(drop=True)


def run_preprocessing(config_path: str = "configs/config.yaml") -> None:
    cfg = load_config(config_path)

    raw_path = resolve_path(cfg["data"]["raw_path"])
    processed_path = resolve_path(cfg["data"]["processed_path"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    allowed_compounds = cfg["preprocessing"]["allowed_compounds"]
    outlier_threshold_pct = float(cfg["preprocessing"]["outlier_threshold_pct"])
    train_years = cfg["data"]["train_years"]
    test_years = cfg["data"]["test_years"]
    target_col = cfg["validation"]["target_col"]

    logger.info(f"Wczytywanie surowych danych z: {raw_path}")
    df_raw = pd.read_parquet(raw_path)

    logger.info("=== ETAP 1: FILTRACJA DOMENOWA I USUWANIE OUTLIERÓW ===")
    df_clean, funnel = apply_domain_filters(
        df=df_raw,
        allowed_compounds=allowed_compounds,
        outlier_threshold_pct=outlier_threshold_pct,
    )

    logger.info("=== ETAP 2: INŻYNIERIA CECH ===")
    df_featured = engineer_features(df=df_clean, train_years=train_years)

    logger.info("=== ETAP 3: SELEKCJA KOLUMN I WALIDACJA KOŃCOWA ===")
    df_final = select_and_order_columns(df=df_featured, target_col=target_col)

    null_counts = df_final.isna().sum()
    if null_counts.sum() > 0:
        raise ValueError(f"Wykryto nieoczekiwane braki danych w zbiorze wynikowym:\n{null_counts[null_counts > 0]}")

    train_rows = int(df_final["Year"].isin(train_years).sum())
    test_rows = int(df_final["Year"].isin(test_years).sum())

    logger.info(f"Finalny wymiar zbioru: {df_final.shape[0]:,} wierszy x {df_final.shape[1]} kolumn")
    logger.info(f"Podział czasowy -> Train ({train_years}): {train_rows:,} | Test ({test_years}): {test_rows:,}")

    df_final.to_parquet(processed_path, index=False)
    logger.info(f"Zapisano przetworzony zbiór do: {processed_path}")

    report_path = processed_path.parent / "preprocessing_report.json"
    report = {
        "raw_shape": list(df_raw.shape),
        "final_shape": list(df_final.shape),
        "train_years": train_years,
        "test_years": test_years,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "outlier_threshold_pct": outlier_threshold_pct,
        "allowed_compounds": allowed_compounds,
        "filtering_funnel": funnel,
        "final_columns": df_final.columns.tolist(),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Zapisano raport metodologiczny do: {report_path}")


if __name__ == "__main__":
    run_preprocessing()