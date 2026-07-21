# Obesity Level Prediction using Machine Learning

## Description

This project predicts obesity levels based on an individual's lifestyle and physical characteristics using a Machine Learning classification model.

The application is developed with Gradio to provide an interactive web interface.

---

## Project Structure

```
.
├── app.py
├── requirements.txt
├── README.md
├── best_model_obesity.pkl
├── preprocessor_obesity.pkl
├── label_encoder_obesity.pkl
└── ObesityDataSet_raw_and_data_sinthetic.csv
```

---

## Installation

Install all required packages.

```bash
pip install -r requirements.txt
```

---

## Run Application

```bash
python app.py
```

or

```bash
gradio app.py
```

---

## Input Features

- Gender
- Age
- Height
- Weight
- Family History with Overweight
- Frequent High Calorie Food Consumption
- Vegetable Consumption
- Number of Main Meals
- Snacking Habit
- Smoking Habit
- Water Consumption
- Calories Monitoring
- Physical Activity
- Technology Usage
- Alcohol Consumption
- Transportation Method

---

## Output

The application predicts one of the following obesity categories:

- Insufficient Weight
- Normal Weight
- Overweight Level I
- Overweight Level II
- Obesity Type I
- Obesity Type II
- Obesity Type III

---

## Libraries

- Gradio
- Pandas
- NumPy
- Scikit-learn
- Joblib

---

## Author

Fundamental Data Science Final Project