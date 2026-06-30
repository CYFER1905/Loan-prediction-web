
from flask import Flask, render_template, request
import pickle
import numpy as np

# Create Flask app
app = Flask(__name__)

# Load trained model
with open("loan_model.pkl", "rb") as file:
    model = pickle.load(file)


# Home Page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read form values
        Gender = int(request.form["Gender"])
        Married = int(request.form["Married"])
        Dependents = int(request.form["Dependents"])
        Education = int(request.form["Education"])
        Self_Employed = int(request.form["Self_Employed"])

        ApplicantIncome = float(request.form["ApplicantIncome"])
        CoapplicantIncome = float(request.form["CoapplicantIncome"])
        LoanAmount = float(request.form["LoanAmount"])
        Loan_Amount_Term = float(request.form["Loan_Amount_Term"])

        Credit_History = float(request.form["Credit_History"])
        Property_Area = int(request.form["Property_Area"])

        # Create feature array
        features = np.array([[
            Gender,
            Married,
            Dependents,
            Education,
            Self_Employed,
            ApplicantIncome,
            CoapplicantIncome,
            LoanAmount,
            Loan_Amount_Term,
            Credit_History,
            Property_Area
        ]])

        # Predict
        prediction = model.predict(features)

        if prediction[0] == 1:
            result = "✅ Loan Approved"
        else:
            result = "❌ Loan Rejected"

        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template(
            "index.html",
            prediction=f"Error: {e}"
        )


if __name__ == "__main__":

    app.run(host="0.0.0.0", port=10000, debug=True)
    

