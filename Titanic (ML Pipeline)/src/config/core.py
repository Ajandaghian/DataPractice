import yaml
from pathlib import Path
import logging

log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "configuration.yml"

def load_config() -> dict:
    with open(CONFIG_PATH, "r") as file:
        try:
            config = yaml.safe_load(file)
            return config
        except Exception as e:
            log.error(f"Error loading config: {e}")
            raise e

config = load_config()


if __name__ == "__main__":
    config = load_config()
    log.debug(f"Configuration loaded: {config}")