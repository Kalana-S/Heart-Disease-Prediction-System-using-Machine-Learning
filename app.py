from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import json

app = Flask(__name__)

model = pickle.load(open("best_xgb_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

with open("model_columns.json", "r") as f:
    meta = json.load(f)
    model_columns = meta["columns"]
    numerical_columns = meta["numerical_columns"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    form_data = request.form.to_dict()
    
    input_df = pd.DataFrame([form_data])

    for col in numerical_columns:
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

    input_encoded = pd.get_dummies(input_df, drop_first=True)

    for col in model_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    input_encoded = input_encoded[model_columns]

    input_encoded[numerical_columns] = scaler.transform(input_encoded[numerical_columns])

    prediction = model.predict(input_encoded)[0]
    result = "has Heart Disease" if prediction == 1 else "does NOT have Heart Disease"

    return render_template("result.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
