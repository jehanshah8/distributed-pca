import sys
import pandas as pd
import numpy as np
import os


def load_dataset(dataset_path='iris_with_cluster.csv', seperate_feats=True):
    df = pd.read_csv(dataset_path)
    if seperate_feats:
        data_mat = df.iloc[:, :-1]
        label_mat = df.iloc[:, -1]
        return data_mat, label_mat
    else:
        return df


def get_local_dataset_path(dataset_path, id):
    local_dataset_path = dataset_path[:-4] + str(id) + ".csv"
    return local_dataset_path


def split_dataset(dataset_path, n_chunks, write=True):
    df = load_dataset(dataset_path, False)
    chunks = np.array_split(df, n_chunks)

    if write:
        for i in range(len(chunks)):
            chunks[i].to_csv(get_local_dataset_path(
                dataset_path, i), index=False)
    else:
        return chunks


if __name__ == '__main__':
    if len(sys.argv) == 3:
        dataset_path = sys.argv[1]
        n = sys.argv[2]
    else:
        dataset_path = '/datasets/iris/iris_with_cluster.csv'
        dataset_path = os.getcwd() + dataset_path
        n = 7

    split_dataset(dataset_path, n)
