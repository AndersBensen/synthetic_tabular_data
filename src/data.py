
import pandas as pd 


def get_dataset(path: str):

    df = pd.read_csv(path)

    X = df.to_numpy()
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)

    cols = df.columns

    return X, X_mean, X_std, cols, df
