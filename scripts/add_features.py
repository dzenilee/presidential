import pandas as pd

from presidential.features import SegmentFeaturizer
from presidential.utils import DATA_DIR


path_to_df0 = DATA_DIR / "2019-2020/processed_transcripts_2019-12-01.csv"
df0 = pd.read_csv(path_to_df0)


if __name__ == "__main__":
    from datetime import date

    s = SegmentFeaturizer()

    df1 = df0
    feature_dict = s.featurize(df1.segment)
    for k, v in feature_dict.items():
        print(k)
        df1[k] = v

    filename = f"2019-2020/processed_transcripts_with_features_{date.today()}.csv"
    df1.to_csv(DATA_DIR / filename, index=False)
    n_rows = df1.shape[0]
    n_features = df1.shape[1]
    print(
        f"Wrote a dataframe containing {n_rows} rows and {n_features} "
        f"features to {filename}."
    )
