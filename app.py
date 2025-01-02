import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st


# Define working directory and paths
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "plant_disease_prediction_model.h5")
class_indices_path = os.path.join(working_dir, "class_indices.json")
disease_database_path = os.path.join(working_dir, 'disease_database.json')

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Load class indices
with open(class_indices_path, 'r') as f:
    class_indices = json.load(f)

# Load the disease information database
with open(disease_database_path, 'r') as f:
    disease_info = json.load(f)

# Function to load and preprocess the image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

# Function to predict the class of an image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Streamlit App Layout
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

if app_mode == "Home":
    st.header("Plant Disease Recognition System")
    image_path = os.path.join(working_dir, "home_page.jpeg")
    st.image(image_path, use_column_width=True)
    st.markdown("""
        Welcome to the Plant Disease Recognition System! 🌿🔍
        
        Our mission is to help identify plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

        ### How It Works
        1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
        2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
        3. **Results:** View the results and recommendations for further action.

        ### Why Choose Us?
        - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
        - **User-Friendly:** Simple and intuitive interface for seamless user experience.
        - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

        ### Get Started
        Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

        ### About Us
        Learn more about the project, our team, and our goals on the **About** page.
    """)

elif app_mode == "About":
    st.header("About")
    st.markdown("""
        #### About Dataset
        This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repository. This dataset consists of about 87K RGB images of healthy and diseased crop leaves categorized into 38 different classes. The total dataset is divided into an 80/20 ratio of training and validation set preserving the directory structure. A new directory containing 33 test images is created later for prediction purposes.

        #### Content
        1. **Train:** 70,295 images
        2. **Test:** 33 images
        3. **Validation:** 17,572 images
    """)

elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if test_image:
        st.image(test_image, use_column_width=True)

        if st.button("Predict"):
            st.write("Analyzing the image...")
            # Save the uploaded file to a temporary location
            temp_image_path = os.path.join(working_dir, "temp_image.png")
            with open(temp_image_path, "wb") as f:
                f.write(test_image.getbuffer())
            
            try:
                prediction = predict_image_class(model, temp_image_path, class_indices)
                formatted_class_name = ' '.join(word.capitalize() for word in prediction.split('___')[1].split('_'))
                st.success(f'Prediction: {formatted_class_name}')
                
                if prediction in disease_info:
                    info = disease_info[prediction]
                    st.write(f"**Predicted Disease:** {formatted_class_name}")
                    st.write(f"**Symptoms:** {', '.join(info['symptoms'])}")
                    st.write(f"**Causes:** {', '.join(info['causes'])}")
                    st.write(f"**Prevention:** {', '.join(info['prevention'])}")
                    st.write(f"**Cure:** {', '.join(info['cure'])}")
                else:
                    st.write(f"**Predicted Disease:** {formatted_class_name}")
                    st.write("No additional information available for this disease.")
            except Exception as e:
                st.error(f"Error in processing the image: {e}")
            finally:
                # Clean up the temporary image file
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
