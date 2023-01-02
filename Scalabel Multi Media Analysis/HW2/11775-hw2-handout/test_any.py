import pickle
import os
import numpy as np

#name = "OTAxMzE4ODQ5OTg5MzkwMTk5Ng==.pkl"


folder_name = "data/cnn3d"
for name in os.listdir(folder_name):
    with open(f'{folder_name}/{name}', 'rb') as f:
        data = pickle.load(f)
        print(name, data[1].shape)
        

print(len(os.listdir(folder_name)))


# with open("data/cnn_conv/NzQzMDU4MDk5MjY3MDE1NTIwNg==.pkl", 'rb') as f:
#     data = pickle.load(f)
#     print(data[1].shape)

# #feature_path = "data/cnn_conv/NzQzMDU4MDk5MjY3MDE1NTIwNg==.pkl" 
# feature_path = "data/cnn_conv/LTQzOTk0ODM4NzQ3NTYyMTc4MzQ=.pkl"
# features = []
# with open(feature_path, 'rb') as f:
#     while True:
#         try:
#             _, frame_feature = pickle.load(f)
#             #print("feature_path",feature_path)
#             print("frame_feature",frame_feature.shape)
#             features.append(frame_feature)
#         except EOFError:
#             print("False", frame_feature)
#             break
# print("len(features)", len(features))
# print(np.array_equal(features[0],features[1]))