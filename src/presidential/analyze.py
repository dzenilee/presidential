import pandas as pd
from presidential.utils import DATA_DIR


path_to_csv = DATA_DIR / "new_df2.csv"
df = pd.read_csv(path_to_csv)
