import json
import numpy as np

def JSON_Serialization(NumpyArray):
    ListData = ConvertNumpyArray2List(NumpyArray)
    encoder = json.dumps(ListData)
    return encoder

def JSON_Deserialization(encodedJSON):
    decoder = json.loads(encodedJSON)
    print(decoder)
    numpyData = np.asarray(decoder)
    return numpyData

def ConvertNumpyArray2List(NumpyArray):
    ListArray = np.ndarray.tolist(NumpyArray)
    return ListArray

# Examples
A = np.array([1, 2, 3, 4])
print(A)
print(type(A))

A = JSON_Serialization(A)
print(A)
print(type(A))

A = JSON_Deserialization(A)
print(A)
print(type(A))