import os
from datasets import load_dataset

DATA_DIR = "./data"
TRAIN_DIR = os.path.join(DATA_DIR, "train.csv")

def main():
    print("Hello from workspace!")
    dataset = load_dataset("csv", data_files=TRAIN_DIR)
    print(dataset)



if __name__ == "__main__":
    main()
