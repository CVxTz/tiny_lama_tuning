import pandas as pd
from pathlib import Path
import json


if __name__ == "__main__":
    DATA_PATH = Path(__file__).parents[1] / "data"

    df = pd.DataFrame(
        [json.load(open(f_name)) for f_name in (DATA_PATH / "jsons").glob("*.json")]
    )

    df["context"] = df.apply(lambda x: f"{x['title']} {x['content']}", axis=1)

    df = df[["ID", "context", "bias_text"]]

    train = pd.read_csv(DATA_PATH / "splits" / "random" / "train.tsv", sep="\t")
    valid = pd.read_csv(DATA_PATH / "splits" / "random" / "valid.tsv", sep="\t")
    test = pd.read_csv(DATA_PATH / "splits" / "random" / "test.tsv", sep="\t")

    train = pd.merge(train, df, on="ID", how="inner")
    valid = pd.merge(valid, df, on="ID", how="inner")
    test = pd.merge(test, df, on="ID", how="inner")

    train.to_csv(DATA_PATH / "train.csv")
    valid.to_csv(DATA_PATH / "valid.csv")
    test.to_csv(DATA_PATH / "test.csv")
