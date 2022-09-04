import pickle
import numpy as np
from numpy.linalg import norm
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
import os
import cv2

features_list = np.array(pickle.load(open('embeddings.pkl','rb')))

filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable= False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])


#importing test images:
#loading image path:
img = image.load_img('test\\OIP-8MsIGi07TOL_Xw503dVRkgHaHR.jpeg',target_size=(224,224))

#converting the img to an array:
img_array = image.img_to_array(img)

#we will get a 3D array from the above process,
#but we need a 4D array instead as we need a batch of images to process further:
expanded_img_array = np.expand_dims(img_array,axis=0)

#forwarding the array to preprocess_input:
preprocessed_img = preprocess_input(expanded_img_array)

#predicting the result & converting it to 1D array using flatten():
result = model.predict(preprocessed_img).flatten()

#normalizing the result to get the predictions between 0 & 1:
normalized_result = result/norm(result)

neighbors = NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean')
neighbors.fit(features_list)
distances, indices = neighbors.kneighbors([normalized_result])

print(indices)


for file in indices[0][1:6]:
    temp_img = cv2.imread(filenames[file])
    cv2.imshow('output',temp_img)
    cv2.waitKey(0)
