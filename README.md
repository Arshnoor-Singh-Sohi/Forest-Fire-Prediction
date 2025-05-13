# 🔥 Forest Fire Prediction System

![Forest Fire Banner](https://images.unsplash.com/photo-1601058497098-678495bac67d?ixlib=rb-1.2.1&auto=format&fit=crop&w=1200&h=400&q=80)

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Live Demo](#live-demo)
- [Features](#features)
- [Dataset Description](#dataset-description)
- [Technical Architecture](#technical-architecture)
- [Machine Learning Approach](#machine-learning-approach)
- [Model Performance](#model-performance)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)
- [Contributions](#contributions)
- [License](#license)

---

## 🌟 Project Overview

The Forest Fire Prediction System is a comprehensive machine learning application designed to predict the likelihood of forest fires in specific regions based on meteorological data. Forest fires pose significant environmental, economic, and social threats, particularly in vulnerable regions like Algeria where this dataset originates. Early prediction and detection are crucial for minimizing damage and protecting both ecosystems and human lives.

This project leverages the Algerian Forest Fires dataset, which contains data from two regions of Algeria (Bejaia and Sidi Bel-Abbes) collected during the period from June 2012 to September 2012. Using various meteorological measurements and derived fire weather indices, we've built machine learning models that can:

1. **Classification Task**: Predict whether a forest fire will occur (binary classification: fire/no fire)
2. **Regression Task**: Predict the Fire Weather Index (FWI), a numerical indicator of fire danger

The system is deployed as a web application that provides an intuitive interface for users to input weather conditions and receive predictions on forest fire risk, enabling forestry departments, environmental agencies, and emergency services to take preventive actions.

---

## 🔴 Live Demo

Experience the application live at:
[https://forest-fire-prediction-1aa0.onrender.com/predictdata](https://forest-fire-prediction-1aa0.onrender.com/predictdata)

---

## ✨ Features

- **Real-time Prediction**: Input current weather parameters to get immediate fire risk assessment
- **Dual Prediction Models**: 
  - Binary classification (fire/no fire) for direct risk assessment
  - FWI regression for detailed fire danger rating
- **Interactive Web Interface**: User-friendly design for easy data input and result interpretation
- **REST API**: Backend API for integration with other systems
- **Data Visualization**: Exploratory data analysis visualizations
- **Responsive Design**: Accessible on desktop and mobile devices

---

## 📊 Dataset Description

The Algerian Forest Fires dataset used in this project was sourced from the UCI Machine Learning Repository. It contains meteorological data and derived fire weather indices for two regions of Algeria: Bejaia and Sidi Bel-Abbes, spanning from June to September 2012.

### Variables in the Dataset

#### Meteorological Features:
- **Temperature**: Day temperature in Celsius degrees (°C)
- **RH**: Relative humidity percentage (%)
- **Ws**: Wind speed in km/h
- **Rain**: Total day rainfall in mm

#### Fire Weather Index (FWI) System Components:
- **FFMC**: Fine Fuel Moisture Code - indicates the ease of ignition and flammability of fine fuels
- **DMC**: Duff Moisture Code - represents the moisture content of shallow organic layers
- **DC**: Drought Code - represents the moisture content of deep organic layers
- **ISI**: Initial Spread Index - indicates the expected rate of fire spread
- **BUI**: Buildup Index - indicates the total amount of fuel available for combustion
- **FWI**: Fire Weather Index - general index of fire danger

#### Target Variables:
- **Classes**: Binary indicator of fire occurrence (fire/no fire)
- **Region**: Location (Bejaia or Sidi Bel-Abbes)

### Data Distribution

| Region | Number of Records | Fire Occurrences | No Fire Occurrences |
|--------|-------------------|------------------|---------------------|
| Bejaia | 122 | 83 | 39 |
| Sidi Bel-Abbes | 122 | 89 | 33 |
| **Total** | **244** | **172** | **72** |

### Visual Overview of Dataset Structure

```
                  ┌─────────────┐
                  │ Input Data  │
                  └──────┬──────┘
                         │
          ┌──────────────┴──────────────┐
          │                             │
┌─────────▼───────────┐     ┌───────────▼─────────┐
│ Weather Parameters  │     │ FWI Components      │
│ - Temperature       │     │ - FFMC              │
│ - RH                │     │ - DMC               │
│ - Ws                │     │ - DC                │
│ - Rain              │     │ - ISI               │
└─────────┬───────────┘     │ - BUI               │
          │                 │ - FWI               │
          │                 └───────────┬─────────┘
          │                             │
          └──────────────┬──────────────┘
                         │
                ┌────────▼────────┐
                │ Machine Learning │
                │     Models      │
                └────────┬────────┘
                         │
          ┌──────────────┴──────────────┐
          │                             │
┌─────────▼───────────┐     ┌───────────▼─────────┐
│  Classification     │     │     Regression      │
│  (Fire/No Fire)     │     │     (FWI Value)     │
└─────────────────────┘     └─────────────────────┘
```

---

## 🏗️ Technical Architecture

The Forest Fire Prediction System follows a modular architecture that separates concerns between data processing, model training, and web application deployment.

### System Architecture Diagram

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Data Sources   │────▶│  Data Pipeline  │────▶│  Model Training │
│  (UCI Dataset)  │     │  & Processing   │     │  & Evaluation   │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         │
┌─────────────────┐     ┌─────────────────┐     ┌────────▼────────┐
│                 │     │                 │     │                 │
│   Client Web    │◀───▶│   Flask Web     │◀───▶│  Trained Models │
│   Browser       │     │   Application   │     │  & Predictors   │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                │
                                │
                        ┌───────▼───────┐
                        │               │
                        │   MongoDB     │
                        │   Database    │
                        │               │
                        └───────────────┘
```

### Technology Stack

#### Backend:
- **Python**: Core programming language
- **Flask**: Web framework for API and frontend rendering
- **Scikit-learn**: Machine learning library for model training and evaluation
- **Pandas/NumPy**: Data manipulation and numerical computations
- **MongoDB Atlas**: Database for storing processed dataset
- **Pickle**: For model serialization

#### Frontend:
- **HTML/CSS**: Structure and styling
- **JavaScript**: Client-side interactions
- **Bootstrap**: Responsive design framework

#### Deployment:
- **Render**: Cloud platform for application hosting
- **Git/GitHub**: Version control and code repository

---

## 🧠 Machine Learning Approach

### Data Preprocessing
1. **Data Cleaning**: Handling missing values, outlier detection, and removal
2. **Feature Engineering**: Creating derived features from the raw meteorological data
3. **Data Normalization**: Scaling features to improve model performance
4. **Train-Test Split**: Dividing the dataset into training (80%) and testing (20%) sets
5. **Cross-Validation**: Implementing stratified k-fold cross-validation for robust model evaluation

### Models for Classification Task (Fire/No Fire)

The project implements and compares several machine learning algorithms for the binary classification task:

1. **Logistic Regression**: A probabilistic classification model
2. **Decision Tree**: A non-parametric supervised learning method
3. **Random Forest**: An ensemble learning method using multiple decision trees
4. **XGBoost**: A gradient boosting framework known for performance and speed
5. **K-Nearest Neighbors**: A non-parametric method that classifies based on proximity

### Models for Regression Task (FWI Prediction)

For predicting the Fire Weather Index (FWI) value, the following regression models were implemented:

1. **Linear Regression**: A linear approach to modeling relationships
2. **Ridge Regression**: Linear regression with L2 regularization
3. **Lasso Regression**: Linear regression with L1 regularization
4. **Random Forest Regressor**: Ensemble method for regression tasks
5. **Decision Tree Regressor**: Non-parametric regression approach
6. **K-Nearest Neighbors Regressor**: Instance-based regression
7. **Support Vector Regressor**: Support vector machine for regression

### Hyperparameter Tuning

For optimal model performance, hyperparameter tuning was performed using Randomized Grid Search with cross-validation on the top-performing models. This process identified the best parameters for:

1. **Random Forest**: n_estimators, max_depth, min_samples_split, min_samples_leaf
2. **XGBoost**: learning_rate, max_depth, n_estimators, subsample, colsample_bytree
3. **SVR**: C, gamma, kernel, epsilon

---

## 📈 Model Performance

### Classification Model Performance

After extensive experimentation and hyperparameter tuning, the Random Forest Classifier emerged as the best-performing model for fire prediction.

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.84 | 0.85 | 0.92 | 0.88 | 0.80 |
| Decision Tree | 0.92 | 0.94 | 0.95 | 0.94 | 0.89 |
| Random Forest | **0.96** | **0.97** | **0.98** | **0.97** | **0.95** |
| XGBoost | 0.94 | 0.95 | 0.97 | 0.96 | 0.92 |
| KNN | 0.90 | 0.92 | 0.94 | 0.93 | 0.87 |

### Regression Model Performance

For the FWI prediction task, the Random Forest Regressor demonstrated superior performance.

| Model | R² Score | MAE | MSE | RMSE |
|-------|----------|-----|-----|------|
| Linear Regression | 0.82 | 3.74 | 24.52 | 4.95 |
| Ridge Regression | 0.82 | 3.73 | 24.50 | 4.95 |
| Lasso Regression | 0.81 | 3.79 | 25.14 | 5.01 |
| Random Forest Regressor | **0.97** | **1.28** | **4.18** | **2.04** |
| Decision Tree Regressor | 0.88 | 2.46 | 15.76 | 3.97 |
| KNN Regressor | 0.92 | 1.98 | 10.32 | 3.21 |
| SVR | 0.84 | 3.42 | 21.84 | 4.67 |

### Feature Importance

Feature importance analysis revealed the most significant predictors for fire occurrence:

#### Classification Task:
1. **Temperature**: 25.3%
2. **FFMC** (Fine Fuel Moisture Code): 21.7%
3. **DMC** (Duff Moisture Code): 18.2%
4. **ISI** (Initial Spread Index): 14.1%
5. **Wind Speed**: 9.8%
6. **Relative Humidity**: 8.6%
7. **DC** (Drought Code): 1.9%
8. **Rain**: 0.4%

#### Regression Task (FWI Prediction):
1. **ISI** (Initial Spread Index): 28.7%
2. **BUI** (Buildup Index): 26.3%
3. **DMC** (Duff Moisture Code): 16.2%
4. **Temperature**: 11.4%
5. **FFMC** (Fine Fuel Moisture Code): 9.5%
6. **DC** (Drought Code): 5.6%
7. **Relative Humidity**: 1.8%
8. **Wind Speed**: 0.5%

---

## 🛠️ Installation & Setup

To set up the Forest Fire Prediction System locally, follow these steps:

### Prerequisites
- Python 3.8 or higher
- Git
- MongoDB (local or Atlas)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Arshnoor-Singh-Sohi/Forest-Fire-Prediction.git
cd Forest-Fire-Prediction
```

### Step 2: Create and Activate Virtual Environment
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Set Up MongoDB
- Create a MongoDB database (locally or on MongoDB Atlas)
- Update the connection string in `config.py`

### Step 5: Run the Application
```bash
python app.py
```

The application will be accessible at `http://localhost:5000/`

### Docker Deployment (Alternative)
```bash
# Build Docker image
docker build -t forest-fire-prediction .

# Run Docker container
docker run -p 5000:5000 forest-fire-prediction
```

---

## 📝 Usage Guide

### Making Predictions Through the Web Interface

1. **Access the Application**:
   Navigate to the home page at `http://localhost:5000/` (local) or the [live demo](https://forest-fire-prediction-1aa0.onrender.com/).

2. **Select Prediction Type**:
   Choose between "Fire Occurrence Prediction" (Classification) or "Fire Weather Index Prediction" (Regression).

3. **Input Weather Parameters**:
   Fill in the form with the following information:
   - Temperature (°C)
   - Relative Humidity (%)
   - Wind Speed (km/h)
   - Rain (mm)
   - FFMC
   - DMC
   - DC
   - ISI
   - Region (Bejaia or Sidi Bel-Abbes)

4. **Submit and View Results**:
   Click "Predict" to see the result, which will show either:
   - Fire probability (for classification)
   - Predicted FWI value (for regression)

### Using the API

The system provides a REST API for programmatic access:

#### Fire Classification Prediction
```bash
curl -X POST http://localhost:5000/predict_api \
  -H "Content-Type: application/json" \
  -d '{
    "temperature": 29.0,
    "rh": 57.0,
    "ws": 18.0,
    "rain": 0.0,
    "ffmc": 89.3,
    "dmc": 142.4,
    "dc": 601.4,
    "isi": 9.0,
    "region": "bejaia"
  }'
```

#### FWI Regression Prediction
```bash
curl -X POST http://localhost:5000/predict_api_regression \
  -H "Content-Type: application/json" \
  -d '{
    "temperature": 29.0,
    "rh": 57.0,
    "ws": 18.0,
    "rain": 0.0,
    "ffmc": 89.3,
    "dmc": 142.4,
    "dc": 601.4,
    "isi": 9.0,
    "region": "bejaia"
  }'
```

---

## 📚 API Documentation

### Endpoints

#### 1. GET `/`
- **Description**: Renders the home page
- **Response**: HTML page with navigation options

#### 2. GET `/predictdata`
- **Description**: Renders the prediction form
- **Response**: HTML form for inputting weather parameters

#### 3. POST `/predict`
- **Description**: Processes the form submission for classification
- **Request Body**: Form data with weather parameters
- **Response**: Prediction result page with fire/no fire classification

#### 4. POST `/predictR`
- **Description**: Processes the form submission for regression
- **Request Body**: Form data with weather parameters
- **Response**: Prediction result page with FWI value prediction

#### 5. POST `/predict_api`
- **Description**: API endpoint for classification prediction
- **Request Body**: JSON with weather parameters
- **Response**: JSON with prediction result
- **Example Response**:
  ```json
  {
    "prediction": "Fire",
    "probability": 0.92
  }
  ```

#### 6. POST `/predict_api_regression`
- **Description**: API endpoint for regression prediction
- **Request Body**: JSON with weather parameters
- **Response**: JSON with predicted FWI value
- **Example Response**:
  ```json
  {
    "prediction": 32.7
  }
  ```

---

## 📁 Project Structure

```
Forest-Fire-Prediction/
├── app.py                      # Flask application main file
├── requirements.txt            # Project dependencies
├── config.py                   # Configuration parameters
├── Procfile                    # Deployment configuration for Heroku
├── .gitignore                  # Git ignore file
├── README.md                   # Project documentation (this file)
├── models/                     # Saved ML models
│   ├── classification_model.pkl  # Random Forest Classifier
│   └── regression_model.pkl      # Random Forest Regressor
├── data/                       # Dataset files
│   ├── Algerian_forest_fires_dataset.csv  # Original dataset
│   ├── processed_data.csv              # Processed dataset
│   └── data_description.md             # Dataset documentation
├── notebooks/                  # Jupyter notebooks
│   ├── 1_EDA.ipynb                    # Exploratory Data Analysis
│   ├── 2_Classification_Model.ipynb   # Classification model training
│   └── 3_Regression_Model.ipynb       # Regression model training
├── src/                        # Source code
│   ├── data_preprocessing.py          # Data cleaning and preprocessing
│   ├── feature_engineering.py         # Feature creation and selection
│   ├── model_training.py              # Model training utilities
│   └── evaluation.py                  # Model evaluation metrics
└── templates/                  # HTML templates
    ├── index.html                     # Home page
    ├── predict.html                   # Prediction form
    └── result.html                    # Prediction results
```

---

## 🚀 Future Improvements

The Forest Fire Prediction System can be enhanced in several ways:

1. **Expanded Dataset**: Incorporate more recent data and additional regions to improve model generalization
2. **Advanced Models**: Experiment with deep learning approaches (LSTM, Transformer) for time-series forecasting
3. **Real-time Integration**: Connect to weather APIs for automatic data retrieval and predictions
4. **Geospatial Visualization**: Add interactive maps to visualize fire risk across regions
5. **Mobile Application**: Develop a companion mobile app for field use
6. **Multi-language Support**: Add internationalization for global usage
7. **Alert System**: Implement automated alerts when fire risk exceeds thresholds
8. **Remote Sensing Integration**: Incorporate satellite imagery data for enhanced predictions

---

## 👥 Contributions

Contributions to the Forest Fire Prediction System are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code follows the project's coding standards and includes appropriate tests.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgements

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Algerian+Forest+Fires+Dataset) for providing the dataset
- The Algerian forest management authorities for their data collection efforts
- Research papers on forest fire prediction that informed our approach
- Open-source community for the tools and libraries used in this project

---

*This README was last updated on May 13, 2025*
