# 🚗 AI-Powered Used Car Price Prediction

This project is a complete machine learning application that predicts the price of used cars based on their features. The project includes a full data processing and model training pipeline, and a user-friendly web interface built with Streamlit to serve the trained model.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR_STREAMLIT_APP_URL_HERE)
*(Note: Replace the link above with your actual Streamlit Cloud app URL after deployment.)*

## ✨ Features

- **End-to-End ML Pipeline:** The project is structured with a modular pipeline that handles data loading, cleaning, preprocessing, feature engineering, and model training.
- **Data-Driven Cleaning:** The pipeline automatically handles missing data, removes outliers, and drops irrelevant or problematic columns.
- **Interactive Web App:** A web interface built with Streamlit allows users to input car features and get instant price predictions.
- **Cloud-Ready:** The application is designed for cloud deployment, with the ability to download the pre-trained model from a remote URL (like GitHub Releases) if not found locally.

## 🛠️ Tech Stack

- **Python 3.12**
- **Scikit-learn:** For data preprocessing and building the ML pipeline.
- **XGBoost:** As the core regression model for price prediction.
- **Pandas:** For data manipulation and analysis.
- **Streamlit:** For building and serving the interactive web application.
- **Joblib:** For saving and loading the trained model pipeline.

## ⚙️ Setup and Installation

To run this project on your local machine, follow these steps:

**1. Clone the Repository**
```bash
git clone [https://github.com/SuleymanToklu/car-price-prediction.git](https://github.com/SuleymanToklu/car-price-prediction.git)
cd car-price-prediction

2. Create and Activate a Virtual Environment
It's highly recommended to use a virtual environment to manage project dependencies.

# Create the virtual environment
python -m venv .venv

# Activate the environment (on Windows)
.\.venv\Scripts\Activate.ps1

3. Install Dependencies
All required libraries are listed in the requirements.txt file.

pip install -r requirements.txt

4. Download the Data
The dataset (vehicles.csv) is not included in this repository due to its large size. Please download it from this Kaggle page and place it inside a data/ folder in the project's root directory.
🚀 Usage

The project has two main scripts: one for training the model and one for running the application.

1. Train the Model
Before you can run the application, you need to train the model and generate the price_prediction_pipeline.joblib file.

Run the following command in your terminal:

python main_training_pipeline.py

This script will perform all the data processing steps and save the trained pipeline to the saved_pipeline/ directory.

2. Run the Streamlit Application
Once the model pipeline is saved, you can start the interactive web application.

Run the following command:

streamlit run app.py

This will open the application in your default web browser.
📁 Project Structure

/car-price-prediction
|
|-- .gitignore          # Specifies files to be ignored by Git
|-- app.py              # The main Streamlit application file
|-- main_training_pipeline.py # Main script to run the training pipeline
|-- requirements.txt    # Lists all project dependencies
|-- README.md           # This file
|
|-- data/               # (Not on Git) To store the dataset
|-- saved_pipeline/     # (Not on Git) To store the trained .joblib pipeline
|
|-- src/                # Source code for the project
|   |-- __init__.py
|   |-- config.py       # Configuration and constant variables
|   |-- data_preprocessor.py # Functions for data cleaning and preprocessing
|   |-- model_trainer.py     # Functions for model and pipeline creation

*This project was developed by Süleyman Toklu as part of an AI Engineer learning
