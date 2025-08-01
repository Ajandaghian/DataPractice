from src.config.core import config
from src.data_manager.data_loader import load_data, save_pipeline, save_data
from src.pipeline import feature_pipeline, model_pipeline
from sklearn.metrics import accuracy_score, classification_report
import pathlib as pl
import logging
log = logging.getLogger(__name__)

ROOT_DIR = pl.Path(__file__).resolve().parent

def train_pipeline(
    raw_data_path: str = ROOT_DIR / config['path']['train'],
    save_feature_pipeline_path: str = ROOT_DIR / config['path']['feature_engineering_pkl'],
    save_model_pipeline_path: str = ROOT_DIR / config['path']['model_pkl'],
    save_processed_data_path: str = ROOT_DIR / config['path']['train_processed']
):
    """Train the feature engineering and model pipelines."""

    raw_data = load_data(path=raw_data_path)
    X = feature_pipeline.fit_transform(raw_data[config['features']['input']])
    log.debug(f"feature pipline finished, \n {X.head()}")
    y = raw_data[config['features']['target']]

    log.info('Model training started.')
    model_pipeline.fit(X, y)
    log.info('Model training completed.')

    # Save
    save_data(df=X, path=save_processed_data_path)
    save_pipeline(pipeline=feature_pipeline, path=save_feature_pipeline_path)
    save_pipeline(pipeline=model_pipeline, path=save_model_pipeline_path)


if __name__ == "__main__":
    train_pipeline()
    log.info("Training pipeline executed successfully.")
