from src.config.core import config
from src.predict import make_predictions
from src.data_manager.data_loader import load_data, save_data
import pandas as pd
import pathlib as pl
import logging
log = logging.getLogger(__name__)

ROOT_DIR = pl.Path(__file__).parent
submission_data_path = ROOT_DIR / config['path']['test']

def main():
    submission_data = load_data(path=submission_data_path)

    # Make predictions
    predictions = make_predictions(data=submission_data)

    # create submission DataFrame
    output_path = ROOT_DIR / config['path']['submission']
    final_sub = pd.DataFrame(predictions, columns=['Survived'])
    final_sub['PassengerId'] = submission_data['PassengerId']
    final_sub = final_sub[['PassengerId', 'Survived']]
    save_data(df=final_sub, path=output_path)
    log.info(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
    log.info("Submission script completed successfully.")
