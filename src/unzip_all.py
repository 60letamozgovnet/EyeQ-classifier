import os
import zipfile
import shutil
import glob
import logging
from tqdm import tqdm

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger()

def combine_split_zips(prefix, output_zip):
    """
    Compiles zip parts (.001, .002, ...) into one zip file.
    """
    parts = sorted(glob.glob(f"{prefix}.zip.*"))
    if parts:
        with open(output_zip, 'wb') as wfd:
            for part in parts:
                with open(part, 'rb') as fd:
                    shutil.copyfileobj(fd, wfd)
        logger.info(f"Compiled {output_zip}")
    else:
        logger.warning(f"Parts for {prefix} are not found.")

def unzip_file(zip_path, target_dir):
    """
    Unzip the zip file in the direcory.
    """
    os.makedirs(target_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        total_files = len(zip_ref.infolist())
        for file in tqdm(zip_ref.infolist(), desc=f"Unpacking {zip_path}", total=total_files):
            zip_ref.extract(file, target_dir)
    logger.info(f"Unpacked: {zip_path} -> {target_dir}")

def delete_zip(zip_path):
    """
    Removed zip file after unpack.
    """
    try:
        os.remove(zip_path)
        logger.info(f"Removed {zip_path}")
    except Exception as e:
        logger.error(f"Error! Can't remove {zip_path}: {e}")

def main():
    os.makedirs('data/train', exist_ok=True)
    os.makedirs('data/test', exist_ok=True)

    # Собираем train.zip и test.zip из частей
    combine_split_zips('train', 'train.zip')
    combine_split_zips('test', 'test.zip')

    # Обрабатываем все zip-файлы в текущей папке
    zip_files = glob.glob('*.zip')

    for zip_file in zip_files:
        if zip_file == 'train.zip':
            unzip_file(zip_file, 'data/train')
        elif zip_file == 'test.zip':
            unzip_file(zip_file, 'data/test')
        else:
            unzip_file(zip_file, 'data')

        # Удаляем zip-файлы после распаковки
        delete_zip(zip_file)

if __name__ == '__main__':
    main()
