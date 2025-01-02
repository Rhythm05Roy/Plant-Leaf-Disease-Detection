# Plant Disease Recognition System

This repository contains a machine learning-powered web application designed to identify plant diseases from uploaded images. The project leverages a deep learning model, trained on a diverse dataset of plant leaf images, to provide fast and accurate disease recognition.

## Features
- **Disease Recognition**: Upload an image of a plant leaf, and the system will predict the disease and provide detailed information, including symptoms, causes, prevention, and cure.
- **Interactive Dashboard**: Built with Streamlit, the app offers a user-friendly interface for seamless interaction.
- **Comprehensive Disease Database**: Includes information about a wide range of plant diseases.

## Project Structure
```
├── Experiment.ipynb            # Jupyter Notebook for model experimentation
├── app.py                      # Main Streamlit application file
├── class_indices.json          # Mapping of class indices to disease names
├── disease_database.json       # Database containing disease details
├── home_page.jpeg              # Image displayed on the home page
├── plant_disease_prediction_model.h5  # Pre-trained model file
└── README.md                   # Project documentation (this file)
```

## Requirements
To run the project locally, ensure the following dependencies are installed:
- Python 3.8+
- TensorFlow
- Streamlit
- Pillow
- NumPy

Install the required packages using:
```bash
pip install -r requirements.txt
```

## Usage
### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/Plant_Leaf_Disease_Detection.git
cd Plant_Leaf_Disease_Detection
```

### 2. Run the Application
Start the Streamlit app using the following command:
```bash
streamlit run app.py
```

### 3. Interact with the App
- Navigate to the **Disease Recognition** page to upload an image of a plant leaf.
- View predictions and detailed disease information.

## Dataset Information
This project uses a dataset recreated through offline augmentation from the original dataset of 87,000 images of healthy and diseased crop leaves, categorized into 38 classes. The data is split as follows:
- **Train**: 70,295 images
- **Validation**: 17,572 images
- **Test**: 33 images

## How It Works
1. **Image Preprocessing**: Uploaded images are resized to 224x224 pixels and normalized.
2. **Model Prediction**: The pre-trained model predicts the disease class from the image.
3. **Disease Information**: Based on the prediction, detailed information is fetched from the database.


## Future Enhancements
- Add support for additional plant species and diseases.
- Improve model accuracy with larger datasets.
- Integrate real-time detection using a mobile application.

## Contributing
We welcome contributions! Feel free to submit issues and pull requests.

## License
This project is licensed under the MIT License. See the LICENSE file for details.


