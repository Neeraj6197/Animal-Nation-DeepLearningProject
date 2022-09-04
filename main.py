import streamlit as st
import pickle
import os
from PIL import Image
import numpy as np
from numpy.linalg import norm
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors

features_list = np.array(pickle.load(open('embeddings.pkl','rb')))

filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable= False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])


st.title("Animal Nation-The Animal Encyclopedia")

#function to save the uploaded file:
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

#creating a func for feature extraction:
def feature_extracion(img_path,model):
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

#defining recomminding function:
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(features_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

#uploading the file:
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        #display the file:
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        #feature extraction:
        features = feature_extracion(os.path.join("uploads",uploaded_file.name),model)
        #st.text(features)
        #recommedation:
        indices = recommend(features,features_list)
        # st.text(indices)

        #display:
        st.header("Seems like " + str(filenames[indices[0][1]].split('\\')[1]))
        col1,col2,col3,col4,col5 = st.columns(5)

        with col1:
            st.image(filenames[indices[0][1]])
        with col2:
            st.image(filenames[indices[0][2]])
        with col3:
            st.image(filenames[indices[0][3]])
        with col4:
            st.image(filenames[indices[0][4]])
        with col5:
            st.image(filenames[indices[0][5]])

        url = 'https://en.wikipedia.org/wiki/{}'.format((filenames[indices[0][1]].split('\\')[1]))
        st.text("To know more about {}, click on the link below:".format((filenames[indices[0][1]].split('\\')[1])))
        st.markdown(url,unsafe_allow_html=True)

    else:
        st.header("Some error occured while uploading the file")



