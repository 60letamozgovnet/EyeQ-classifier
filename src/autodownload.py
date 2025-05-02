import os
import logging
import zipfile
from get_path import get_config
import sys
from unzip_all import main as unzip

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def download_dataset(output_dir: os.path, competition: str = 'diabetic-retinopathy-detection'):
    """
    Download data of competition from kaggle using kaggle api python
    :param output_dir: directory for save data
    :param competition: competition slug (for excample 'diabetic-retinopathy-detection')
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        logger.info(f"Error: {e}")
        sys.exit(1)
    
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, f"{competition}.zip")

    logger.info("Downloading of dataset")
    api.competition_download_files(competition, zip_path)
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        os.remove(zip_path)
        logger.info(f"Dataset was downloaded and unzip to path {output_dir}")
    else:
        logger.info(f"Error: file {zip_path} not found")

if __name__ == "__main__":
    config = get_config()
    path = config["DATA_DIR"]
    download_dataset(path)
    unzip()