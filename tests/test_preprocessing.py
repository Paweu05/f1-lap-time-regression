from pathlib import Path

import yaml


def test_config_structure():
    config_path = Path("configs/config.yaml")
    assert config_path.exists(), "Missing configs/config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert len(cfg["data"]["circuits"]) == 4
    assert 2025 in cfg["data"]["test_years"]
    assert cfg["project"]["random_state"] == 42
