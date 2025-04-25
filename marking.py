import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

path_to_test_csv = r"/data/Label_EyeQ_test.csv"
path_to_train_csv = r"/data/Label_EyeQ_train.csv"

test_df = pd.read_csv(path_to_test_csv)
train_df = pd.read_csv(path_to_train_csv)

df = pd.concat([test_df.drop(test_df.columns[0], axis=1), train_df.drop(train_df.columns[0], axis=1)], ignore_index=True, axis=0)
# logger.info("+"*15)
def remove_rows(df: pd.DataFrame, img_dir) -> pd.DataFrame:
    idxs = []
    img_names = df['image'].values
    for name in img_names:
        img_path = os.path.join(img_dir, name.strip())
        if not os.path.exists(img_path):
            idxs.append(df[df["image"] == name].index[0])
    

    df = df.drop(idxs)
    logger.info(df)
    return df

def marking_dataset(df: pd.DataFrame) -> pd.DataFrame:
    '''
    bin_marks: 0 - can use for train, 1 - can't use for train
    quality: 0 - Good, 1 - Usable, 2 - Reject
    DR_grade - level of diabetic retinopathy
    '''
    
    df['bin_marks'] = df['quality'].apply(lambda x: 0 if x in[0, 1] else 1)
    return df

if __name__ == "__main__":
    img_dir = r"/data/preprocessed/"
    df = remove_rows(df, img_dir)
    df = marking_dataset(df)
    logger.info(df)
