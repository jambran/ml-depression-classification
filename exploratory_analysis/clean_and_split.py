import pandas as pd
from pathlib import Path
import numpy as np
from langdetect import detect
import logging

RANDOM_SEED = 42

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)


def remove_non_english_comments(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("removing non-english comments")
    is_english = "is_english"
    df[is_english] = df.apply(
        lambda row: detect(row.clean_text) == "en",
        axis=1,
    )
    english_only_df = df[df[is_english]]

    # clean up
    df.drop(is_english, inplace=True, axis=1)
    english_only_df.drop(is_english, inplace=True, axis=1)

    return english_only_df


def remove_long_comments(df: pd.DataFrame,
                         max_length: int = 100,
                         ) -> pd.DataFrame:
    logger.info(f"removing comments with length greater than {max_length}")
    word_count = "word_count"
    df[word_count] = df.clean_text.str.split().str.len()
    short_comments = df[df[word_count] <= max_length]

    # clean up
    df.drop(word_count, inplace=True, axis=1)
    short_comments.drop(word_count, inplace=True, axis=1)

    return short_comments


def clean(df: pd.DataFrame,
          ) -> pd.DataFrame:
    """
    from eda.ipynb, we know the data should be cleaned by
    - removing non-english comments
    - removing comments with over 100 words
    :return:
    """
    df = remove_non_english_comments(df)
    df = remove_long_comments(df, max_length=100)

    return df


def save_to_file(df, data_dir):
    logger.info("saving to file")
    train_outfile = data_dir / "train.csv"
    val_outfile = data_dir / "val.csv"
    test_outfile = data_dir / "test.csv"

    # 60% data in train, 20% val, 20% test
    train, validate, test = \
        np.split(
            df.sample(frac=1, random_state=RANDOM_SEED),
            [int(.6 * len(df)), int(.8 * len(df))],
        )

    train.to_csv(train_outfile)
    validate.to_csv(val_outfile)
    test.to_csv(test_outfile)


def main():
    """
    clean the data, then separate into train, dev, and test files for model
    building
    """
    logger.info("loading in data")
    data_dir = Path("../data/")
    file_location = data_dir / "depression_dataset_reddit_cleaned.csv"
    df = pd.read_csv(file_location)
    df = clean(df)

    save_to_file(df, data_dir)

    logger.info("done")


if __name__ == '__main__':
    main()
