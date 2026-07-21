import gradio as gr
import pandas as pd
import joblib

# ==========================================================
# LOAD MODEL
# ==========================================================

model = joblib.load("best_model_obesity.pkl")
preprocessor = joblib.load("preprocessor_obesity.pkl")
label_encoder = joblib.load("label_encoder_obesity.pkl")

# ==========================================================
# DESKRIPSI HASIL
# ==========================================================

description = {
    "Insufficient_Weight":
        "The individual is below the normal weight range. It is recommended to maintain a balanced diet and consult a healthcare professional if necessary.",

    "Normal_Weight":
        "The individual has a healthy body weight. Continue maintaining a balanced diet and regular physical activity.",

    "Overweight_Level_I":
        "The individual is slightly overweight. Regular exercise and healthier eating habits are recommended.",

    "Overweight_Level_II":
        "The individual is moderately overweight. Lifestyle improvements are recommended to reduce future health risks.",

    "Obesity_Type_I":
        "The individual is classified as Obesity Type I. Medical consultation and weight management are recommended.",

    "Obesity_Type_II":
        "The individual is classified as Obesity Type II. Professional medical advice and a structured weight-loss program are strongly recommended.",

    "Obesity_Type_III":
        "The individual is classified as Obesity Type III. Immediate medical supervision is highly recommended due to increased health risks."
}

# ==========================================================
# PREDICTION FUNCTION
# ==========================================================

def predict_obesity(
    Gender,
    Age,
    Height,
    Weight,
    FamilyHistory,
    FAVC,
    FCVC,
    NCP,
    CAEC,
    SMOKE,
    CH2O,
    SCC,
    FAF,
    TUE,
    CALC,
    MTRANS,
):

    input_data = pd.DataFrame({

        "Gender":[Gender],
        "Age":[Age],
        "Height":[Height],
        "Weight":[Weight],
        "family_history_with_overweight":[FamilyHistory],
        "FAVC":[FAVC],
        "FCVC":[FCVC],
        "NCP":[NCP],
        "CAEC":[CAEC],
        "SMOKE":[SMOKE],
        "CH2O":[CH2O],
        "SCC":[SCC],
        "FAF":[FAF],
        "TUE":[TUE],
        "CALC":[CALC],
        "MTRANS":[MTRANS]

    })

    # preprocessing
    X = preprocessor.transform(input_data)

    # prediction
    prediction = model.predict(X)

    predicted_class = label_encoder.inverse_transform(prediction)[0]

    # probability
    probability = model.predict_proba(X)[0]

    probability_dict = {
        label_encoder.classes_[i]: float(probability[i])
        for i in range(len(probability))
    }

    # BMI
    bmi = Weight / (Height ** 2)

    # explanation
    explanation = description[predicted_class]

    return (
        round(bmi,2),
        predicted_class,
        probability_dict,
        explanation
    )

# ==========================================================
# INTERFACE
# ==========================================================

demo = gr.Interface(

    fn=predict_obesity,

    inputs=[

        gr.Dropdown(
            ["Male","Female"],
            value="Male",
            label="Gender"
        ),

        gr.Number(
            value=21,
            label="Age"
        ),

        gr.Number(
            value=1.70,
            label="Height (meter)"
        ),

        gr.Number(
            value=70,
            label="Weight (kg)"
        ),

        gr.Dropdown(
            ["yes","no"],
            value="no",
            label="Family History with Overweight"
        ),

        gr.Dropdown(
            ["yes","no"],
            value="no",
            label="Frequent High Calorie Food Consumption"
        ),

        gr.Slider(
            1,
            3,
            step=1,
            value=2,
            label="Vegetable Consumption (FCVC)"
        ),

        gr.Slider(
            1,
            4,
            step=1,
            value=3,
            label="Number of Main Meals (NCP)"
        ),

        gr.Dropdown(
            ["no","Sometimes","Frequently","Always"],
            value="Sometimes",
            label="Snacking Between Meals (CAEC)"
        ),

        gr.Dropdown(
            ["yes","no"],
            value="no",
            label="Smoking"
        ),

        gr.Slider(
            1,
            3,
            step=0.5,
            value=2,
            label="Daily Water Intake (CH2O)"
        ),

        gr.Dropdown(
            ["yes","no"],
            value="no",
            label="Calories Consumption Monitoring (SCC)"
        ),

        gr.Slider(
            0,
            3,
            step=0.5,
            value=1,
            label="Physical Activity Frequency (FAF)"
        ),

        gr.Slider(
            0,
            2,
            step=0.5,
            value=1,
            label="Technology Usage Time (TUE)"
        ),

        gr.Dropdown(
            ["no","Sometimes","Frequently","Always"],
            value="Sometimes",
            label="Alcohol Consumption (CALC)"
        ),

        gr.Dropdown(
            [
                "Walking",
                "Bike",
                "Motorbike",
                "Public_Transportation",
                "Automobile"
            ],
            value="Walking",
            label="Transportation Method (MTRANS)"
        )

    ],

    outputs=[

        gr.Number(label="BMI"),

        gr.Textbox(label="Predicted Obesity Level"),

        gr.Label(
            label="Prediction Probability",
            num_top_classes=7
        ),

        gr.Textbox(
            label="Recommendation"
        )

    ],

    title="Obesity Level Prediction Using Machine Learning",

    description="""
This application predicts obesity levels based on physical characteristics
and lifestyle using a Gradient Boosting Machine Learning model.
""",

    theme=gr.themes.Soft(),

)

if __name__ == "__main__":
    demo.launch()