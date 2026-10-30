import pandas as pd


def load_transcripts(file_path: str = "data/transcripts.csv") -> pd.DataFrame:
    """
    Load transcripts dataset.
    """
    df = pd.read_csv(file_path)
    return df