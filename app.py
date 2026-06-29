from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('loan_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route("/predict", methods=["POST"])
def predict():

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

    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "✅ Loan Approved"
    else:
        result = "❌ Loan Rejected"

    return render_template("index.html", prediction=result)