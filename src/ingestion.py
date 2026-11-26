import logging
from pathlib import Path
from typing import Any

import fastf1
import pandas as pd
import yaml
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def fetch_team_quali_best(year: int, circuit: str) -> pd.DataFrame:
    session_q = fastf1.get_session(year, circuit, "Q")
    session_q.load(telemetry=False, weather=False, messages=False)

    q_laps = session_q.laps.copy()
    q_laps = q_laps[q_laps["LapTime"].notna()]
    q_laps["QualiLapSeconds"] = q_laps["LapTime"].dt.total_seconds()

    team_best = (
        q_laps.groupby("Team", as_index=False)["QualiLapSeconds"]
        .min()
        .rename(columns={"QualiLapSeconds": "TeamQualiBest"})
    )
    return team_best


def fetch_race_data(year: int, circuit: str) -> pd.DataFrame:
    logger.info(f"Pobieranie danych: {year} - {circuit}")

    team_quali = fetch_team_quali_best(year, circuit)

    session_r = fastf1.get_session(year, circuit, "R")
    session_r.load(telemetry=False, weather=True, messages=True)

    laps = session_r.laps.reset_index(drop=True)
    weather = laps.get_weather_data().reset_index(drop=True)

    weather_cols = ["AirTemp", "TrackTemp", "Humidity", "Rainfall"]
    df = pd.DataFrame(laps).join(weather[weather_cols])

    # Konwersja czasów z Timedelta na sekundy
    df["LapTimeSeconds"] = df["LapTime"].dt.total_seconds()
    df["PitInTimeSeconds"] = df["PitInTime"].dt.total_seconds()
    df["PitOutTimeSeconds"] = df["PitOutTime"].dt.total_seconds()

    # Metadane wyścigu
    df["Year"] = year
    df["Circuit"] = circuit
    df["RaceID"] = f"{year}_{circuit}"
    df["TotalLaps"] = int(df["LapNumber"].max())

    # Dołączenie tempa bazowego zespołu z kwalifikacji
    df = df.merge(team_quali, on="Team", how="left")

    selected_columns = [
        "RaceID",
        "Year",
        "Circuit",
        "Driver",
        "Team",
        "Stint",
        "LapNumber",
        "TotalLaps",
        "LapTimeSeconds",
        "PitInTimeSeconds",
        "PitOutTimeSeconds",
        "Compound",
        "TyreLife",
        "FreshTyre",
        "TrackStatus",
        "IsAccurate",
        "Deleted",
        "AirTemp",
        "TrackTemp",
        "Humidity",
        "Rainfall",
        "TeamQualiBest",
    ]

    available_cols = [col for col in selected_columns if col in df.columns]
    return df[available_cols]


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    resolved_path = Path(config_path) if config_path else PROJECT_ROOT / "configs" / "config.yaml"
    if not resolved_path.is_absolute():
        resolved_path = PROJECT_ROOT / resolved_path
    with open(resolved_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def run_ingestion(config_path: str | Path | None = None) -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    config = load_config(config_path)

    data_cfg = config["data"]
    train_years: list[int] = data_cfg.get("train_years", [])
    test_years: list[int] = data_cfg.get("test_years", [])
    seasons: list[int] = list(dict.fromkeys([*train_years, *test_years]))

    circuits: list[str] = data_cfg["circuits"]
    cache_dir = PROJECT_ROOT / data_cfg["cache_dir"]
    output_path = PROJECT_ROOT / data_cfg["raw_path"]

    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fastf1.Cache.configure(cache_dir=str(cache_dir))

    all_races: list[pd.DataFrame] = []

    for year in seasons:
        for circuit in circuits:
            try:
                race_df = fetch_race_data(year, circuit)
                all_races.append(race_df)
                logger.info(f"Sukces {year} {circuit}: {len(race_df)} okrążeń")
            except Exception as exc:
                logger.error(f"Błąd podczas pobierania {year} {circuit}: {exc}")

    if not all_races:
        raise RuntimeError("Nie udało się pobrać żadnych danych wyścigowych.")

    full_df = pd.concat(all_races, ignore_index=True)

    string_cols = ["RaceID", "Circuit", "Driver", "Team", "Compound", "TrackStatus"]
    for col in string_cols:
        if col in full_df.columns:
            full_df[col] = full_df[col].astype(str)

    full_df.to_parquet(output_path, index=False)
    logger.info(
        f"Zakończono! Zapisano {len(full_df)} wierszy i {full_df.shape[1]} kolumn do {output_path}"
    )D


if __name__ == "__main__":
    run_ingestion()