import argparse
import yaml
import os

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config.yaml")
    args = parser.parse_args()

    # Текущая директория, где лежит скрипт (get_path.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Если config путь относительный — делаем его относительным к скрипту
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(script_dir, "..", config_path)

    config_path = os.path.normpath(config_path)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    config = get_config()
    DATA_DIR = config['DATA_DIR']
    SRC_DIR = config["SRC_DIR"]

    print(DATA_DIR)