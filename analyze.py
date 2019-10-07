import pandas as pd
from presidential import DATA_DIR


path_to_csv = DATA_DIR / "new_df.csv"
df = pd.read_csv(path_to_csv)
