import pandas as pd


def fetch_all_recipes() -> pd.DataFrame:
    """Fetch all recipes from CSV file"""
    df = pd.read_csv("utils/recipes.csv")
    if not df.empty:
        # Normalize column names to lowercase for consistency
        df.columns = df.columns.str.lower()
        return df
    else:
        return pd.DataFrame()
