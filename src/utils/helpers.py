import pandas as pd


def save_dataframe(df: pd.DataFrame, file_path: str = "data/output.xlsx"):
    """
    Saves DataFrame to Excel.
    """
    df.to_excel(file_path, index=False)