from sklearn.datasets import fetch_california_housing
import pandas as pd
import os

def save_data():
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/california.csv", index=False)
    
if __name__ == "__main__":
    save_data()
