import pandas as pd

from presidential.features import SegmentFeaturizer
from presidential.utils import DATA_DIR


path_to_df0 = DATA_DIR / "2019-2020/processed_transcripts_2019-12-01.csv"
df0 = pd.read_csv(path_to_df0)


if __name__ == "__main__":
    from datetime import date

    s = SegmentFeaturizer()

    df1 = df0
    feature_dicts = s.featurize(df1.segment)
    features_df = pd.DataFrame(feature_dicts)

    df2 = pd.concat([df1, features_df], axis=1)  # axis=1 is important!

    filename = f"2019-2020/processed_transcripts_with_features_{date.today()}.csv"
    df2.to_csv(DATA_DIR / filename, index=False)
    n_rows = df2.shape[0]
    n_features = df2.shape[1]
    print(
        f"Wrote a dataframe containing {n_rows} rows and {n_features} "
        f"features to {filename}."
    )
