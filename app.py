from flask import Flask,url_for, redirect, render_template , request
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from pandas import DataFrame
import pickle
1
model = pickle.load(open(r"C:\Users\AKHIL\OneDrive\Documents\PROJECT_COLLEGE\Mental-Health-Prediction-Using-Machine-Learning\Mental-Health-Prediction-Using-Machine-Learning\model.pkl", 'rb'))
ct = pickle.load(open(r"C:\Users\AKHIL\OneDrive\Documents\PROJECT_COLLEGE\Mental-Health-Prediction-Using-Machine-Learning\Mental-Health-Prediction-Using-Machine-Learning\ct.pkl", "rb"))
le = pickle.load(open(r"C:\Users\AKHIL\OneDrive\Documents\PROJECT_COLLEGE\Mental-Health-Prediction-Using-Machine-Learning\Mental-Health-Prediction-Using-Machine-Learning\le.pkl", "rb"))
app = Flask(__name__) 

@app.route('/')
def hello_world():
    return render_template('index.html')
    


@app.get('/form')
def show_form():
     return render_template('form.html')

@app.post('/submit_form')
def submit_form():

    data = [{
         "Age": int(request.form["inputAge"]),
         "Gender": request.form["inputGender"],
         "self_employed": request.form["inputSelfEmployed"],
         "family_history": request.form["inputFamilyHistory"],
         "work_interfere": request.form["inputWorkInterference"],
         "no_employees": request.form["inputNoOfEmp"],
         "remote_work": request.form["inputRemoteWork"],
         "tech_company": request.form["inputTechCompany"],
         "benefits": request.form["inputBenefits"],
         "care_options": request.form["inputCareOptions"],
         "wellness_program": request.form["inputWellnessProgram"],
         "seek_help": request.form["inputSeekHelp"],
         "anonymity": request.form["inputAnonymity"],
         "leave": request.form["inputLeave"],
         "mental_health_consequence": request.form["inputMentalHealthConsequence"],
         "phys_health_consequence": request.form["inputPhysHealthConsequence"],
         "coworkers": request.form["inputCoworkers"],
         "supervisor": request.form["inputSupervisor"],
         "mental_health_interview": request.form["inputMentalHealthInterview"],
         "phys_health_interview": request.form["inputPhysHealthInterview"],
         "mental_vs_physical": request.form["inputMentalVsPhysical"],
         "obs_consequence": request.form["inputObsConsequence"],
     }]
    df = DataFrame.from_records(data)
    x = ct.transform(df)
    y = model.predict(x)
    treatment = le.inverse_transform(y)[0]

    if(treatment=="Yes"):
        return render_template('treatmentY.html')
    else:
        return render_template('treatmentN.html')


@app.route('/teampage')
def teampage():
    return 'Welcome to Team page.'
    # redirect"{{url_for(hello_world)}}"


if __name__=="__main__":
    app.run(debug=False, port=8000)