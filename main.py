import streamlit as st
import tensorflow as tf
import numpy as np
def model_prediction(test_image):
    model=tf.keras.models.load_model('trained_model.h5')
    image=tf.keras.preprocessing.image.load_img(test_image,target_size=(64,64))
    input_arr=tf.keras.preprocessing.image.img_to_array(image)
    input_arr=np.array([input_arr])
    predictions=model.predict(input_arr)
    return np.argmax(predictions)



st.sidebar.title("Dashboard")
app_mode=st.sidebar.selectbox("Select Page",["Home","About","Prediction"])

if(app_mode=="Home"):
    st.header("Fruits and vegetables recognition system")
    st.subheader("This is a FAIML project done by:")
    st.write("1. Rayan Pinto")
    st.write("2. Shreyas K")
elif(app_mode=="About"):
    st.header("About")
    st.write("This is a fruits and vegetables recognition system using machine learning.")
    st.write("It can recognize different types of fruits and vegetables from images.")
    st.write("The system is built using Streamlit and TensorFlow.")
    st.write("The model is trained on a dataset of fruits and vegetables.")
    st.write("The dataset is available on Kaggle.")
    st.write("The model is trained using transfer learning.")
    st.write("The model is deployed using Streamlit.")
elif(app_mode=="Prediction"):
    st.header("Prediction")
    text_image=st.file_uploader("Upload an image" )
    st.write("The model will predict the type of fruit or vegetable in the image.")
    if(st.button("Show image")):
        st.image(text_image,width=4,use_column_width=True)
    if(st.button("Predict")):
        st.write("The model is predicting the type of fruit or vegetable in the image.")
        # Load the model and make predictions here
        # model = load_model('model.h5')
        # img = preprocess_image(text_image)
        # prediction = model.predict(img)
        # st.write(prediction)
        st.write("Prediction will be shown here.")
        result=model_prediction(text_image)
        with open("labels.txt") as f:
            content=f.readlines()
        label=[]
        
        for i in content:
            label.append(i[:-1])
        st.success("The model predicts that the image is of type:{} ".format(label[result]))