import numpy as np
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from numpy.linalg import norm
import os
import pickle
from tqdm import tqdm


model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable= False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

#print(model.summary())

#defining a func to extract the features from the image:
def extract_features(img_path,model):
    #loading image path:
    img = image.load_img(img_path,target_size=(224,224))

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

    return normalized_result

#Creating a list to store filenames:
animals = os.listdir('images')

filenames = []

for animal in animals:
    for file in os.listdir(os.path.join('images',animal)):
        filenames.append(os.path.join('images',animal,file))

features_list = []
for file in tqdm(filenames):
    features_list.append(extract_features(file,model))

#dumping the required files:
pickle.dump(features_list,open('embeddings.pkl','wb'))
pickle.dump(filenames,open('filenames.pkl','wb'))
# print(filenames)
# print(len(filenames))