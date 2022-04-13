import src.utils as utils
import src.distributedPCA as distributedPCA

# split dataset into n parts and safe to file as dataset0.csv...datasetn.csv

# spin up coordinator (server)


# loop on n_components,
#   invoke run(n_components) which returns P_hat
#   evaluate
#   write results to file

# 1 run is 1 full execution of algo 1
# We want to repeat runs with various t for given n and mal nodes
# We want to repeat runs with various n given t and mal nodes
# We want to repeat runs with various mal nodes given t and n

# Which question to ask:
# Given n nodes
# How big of a t do we need if we have m mal_nodes
# How many mal nodes can we tolerate if t is set

import sys

if __name__ == '__main__':
    if len(sys.argv) == 5:
        n_nodes = int(sys.argv[1])
        n_mal_nodes = int(sys.argv[2])
        mal_type = sys.argv[3]
        dataset_path = sys.argv[4]
        n_components = sys.argv[5]

    else:
        n_nodes = 1
        n_mal_nodes = 0
        mal_type = 0
        dataset_path = 'iris_with_cluster.csv'
        n_components = 4

    utils.split_dataset(dataset_path, n_components)

    distPCA = distributedPCA.DistributedPCA(
        n_nodes=n_nodes, n_mal_nodes=n_mal_nodes, mal_type=mal_type)
    P_hat = distPCA.fit_transform(
        dataset_path=dataset_path, n_components=n_components)
