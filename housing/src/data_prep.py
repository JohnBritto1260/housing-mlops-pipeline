import os
from sklearn.datasets import fetch_california_housing


def save_data():
    """Fetch and save California housing dataset as CSV."""
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    os.makedirs("housing/data/raw", exist_ok=True)
    df.to_csv("housing/data/raw/california.csv", index=False)


if __name__ == "__main__":
    save_data()
