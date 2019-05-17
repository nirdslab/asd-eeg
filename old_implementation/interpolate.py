import pandas as pd


def run():
    df = pd.read_csv("data/AGG.csv")
    df.dropna(axis="columns", inplace=True)
    df.to_csv("data/clean.csv")


run()
