from src.config.core import config
from src.data_manager.data_loader import load_pipeline, load_data
from src.data_manager.data_validator import validate_data
import pandas as pd
import pathlib as pl

import logging
log = logging.getLogger(__name__)

ROOT_DIR = pl.Path(__file__).resolve().parent

feature_pipeline_path = config['path']['feature_engineering_pkl']
model_pipeline_path = config['path']['model_pkl']

feature_pipeline = load_pipeline(path=ROOT_DIR / feature_pipeline_path)
model_pipeline = load_pipeline(path=ROOT_DIR / model_pipeline_path)
required_features = feature_pipeline.feature_names_in_

def make_predictions(
    data: pd.DataFrame | dict = None
    ) -> dict:
    """Make predictions using the trained model pipeline."""

    if data is None:
        raise ValueError("Data must be provided for making predictions.")
    if isinstance(data, dict):
        data = pd.DataFrame([data])

    validate_data(data)
    X_transformed = feature_pipeline.transform(data[feature_pipeline.feature_names_in_])

    # Make predictions
    predictions = model_pipeline.predict(X_transformed)

    return predictions

if __name__ == "__main__":

    # Example usage
    sample_data = {
        "PassengerId": 1,
        "Pclass": 2,
        "Name": "Braund, Mr. Owen Harris",
        "Sex": "male",
        "Age": 22.3,
        "SibSp": 1,
        "Parch": 0,
        "Ticket": "A/5 21171",
        "Fare": 7.25,
        "Cabin": "none",
        "Embarked": "S"
        }
#    sample_data = load_data(path=ROOT_DIR / config['path']['test'])

    predictions = make_predictions(data=sample_data)
    log.debug(f"Predictions: {predictions}")

