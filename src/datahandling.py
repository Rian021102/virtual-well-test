import pandas as pd

def load_data(path):
    data=pd.read_csv(path)
    return data

def main():
    file_path="/home/rianr/pypro/myvenv/virtual-well-test/data/welltest.csv"
    data=load_data(file_path)
    print(data.head())


if __name__ == "__main__":
    main()


