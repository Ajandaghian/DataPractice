import pandas as pd
from src.config.core import config
import pathlib
import joblib
import logging

log = logging.getLogger(__name__)


def load_data(*, path: str) -> pd.DataFrame:
    """Load data from a CSV file."""

    if not path:
        raise ValueError("Path to the data file must be provided.")

    if pathlib.Path(path).is_dir():
        raise ValueError("The provided path is a directory, not a file. Please provide a valid file path.")

    if pathlib.Path(path).exists() and not pathlib.Path(path).is_file():
        raise ValueError("The provided path does not point to a valid file. Please check the path.")

    try:
        df = pd.read_csv(path)
        log.debug(f"head of data: {df.columns}")
    except Exception as e:
        raise ValueError(f"Error loading data from {path}: {e}")

    return df

def save_data(*, df: pd.DataFrame, path: str):
    """Save DataFrame to a CSV file."""

    if not path:
        raise ValueError("Path to save the data must be provided.")

    try:
        df.to_csv(path, index=False)
    except Exception as e:
        raise ValueError(f"Error saving data to {path}: {e}")



def save_pipeline(*, pipeline, path: str):
    """Save the trained pipeline to a file."""
    if not path:
        raise ValueError("Path to save the pipeline must be provided.")

    try:
        joblib.dump(pipeline, path)

    except Exception as e:
        raise ValueError(f"Error saving pipeline to {path}: {e}")

def load_pipeline(*, path: str):
    """Load a pipeline from a file."""
    if not path:
        raise ValueError("Path to the pipeline file must be provided.")

    try:
        pipeline = joblib.load(path)
    except Exception as e:
        raise ValueError(f"Error loading pipeline from {path}: {e}")

    return pipeline