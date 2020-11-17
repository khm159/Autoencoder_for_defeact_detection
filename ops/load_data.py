import numpy as np

def load_data(path):
    data = np.load(path)
    in_dim = data.shape[1]
    print("  Dataset loading is complete : {} samples".format(len(data)))
    print("  Dataset sample's feature dim : {}".format(in_dim))
    return data,in_dim