

# Anxiety Level Predictor

This project uses a dataset of lifestyle, demographic, and medical attributes to predict anxiety levels using a machine learning regression model.

## 📊 Overview

The model is built using `SGDRegressor` from Scikit-Learn and is trained on a cleaned and encoded dataset. The main script performs the following tasks:

- Loads the dataset
- Preprocesses the data (label encoding, feature transformation)
- Trains a regression model to predict anxiety level (scale of 1–10)

## 🗂 Dataset

Ensure that the dataset `enhanced_anxiety_dataset.csv` is placed in a folder named `Dataset/`:

```

project\_root/
├── Dataset/
│   └── enhanced\_anxiety\_dataset.csv
├── main.py
└── README.md

````

## 🛠 Requirements

Install the required libraries with:

```bash
pip install -r requirements.txt
````

Or manually install:

```bash
pip install pandas numpy scikit-learn
```

## 🚀 Running the Project

Run the `main.py` script:

```bash
python main.py
```
## 🧠 Model Used

The script uses `SGDRegressor`, a linear model trained using stochastic gradient descent. Data is standardized using `StandardScaler` to improve performance.

## 🧹 Preprocessing Steps

* Binary encoding for Yes/No columns
* Label encoding for Gender and Occupation
* Feature scaling before model training

## 📌 Future Improvements

* Add model evaluation metrics (MAE, MSE)
* Export trained model for deployment
* Create a web interface for input and prediction

## 👤 Author

Swarnim Tripathi
