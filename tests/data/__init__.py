from pathlib import Path


def get_test_config_path()-> Path:
    return Path(__file__).parent / "configs" / "prior_testing.yaml"