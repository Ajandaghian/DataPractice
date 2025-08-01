from src.config.core import config
import pandas as pd
import logging

log = logging.getLogger(__name__)


def validate_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate the input data."""
    if not isinstance(data, pd.DataFrame):
        log.debug("Invalid data type. Expected pandas DataFrame.")
        raise ValueError("Data must be a pandas DataFrame.")

    # Check for missing columns
    required_columns = config['features']['input']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        log.debug(f"Missing columns in input data: {missing_columns}")
        raise ValueError(f"Missing columns in input data: {missing_columns}")

    # Check for correct data types
    for col in required_columns:
        if col in data.columns:
            expected_type = config['features']['dtypes'].get(col, None)
            if expected_type and not pd.api.types.is_dtype_equal(data[col].dtype, expected_type):
                log.debug(f"Incorrect data type for column '{col}'. Expected {expected_type}, got {data[col].dtype}.")
                raise ValueError(f"Incorrect data type for column '{col}'. Expected {expected_type}, got {data[col].dtype}.")

    # Check for missing values
    if data[required_columns].isnull().any().any():
        log.error("Input data contains missing values.")

    return data