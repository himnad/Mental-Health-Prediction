from flask import Flask,url_for, redirect, render_template , request
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from pandas import DataFrame
import pickle
import matplotlib.pyplot as plt
import matplotlib
import base64
from io import BytesIO
import pandas as pd
import numpy as np

matplotlib.use('Agg')
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
        #  "work_interfere": request.form["inputWorkInterference"],
        #  "no_employees": request.form["inputNoOfEmp"],
         "remote_work": request.form["inputRemoteWork"],
        #  "tech_company": request.form["inputTechCompany"],
        #  "benefits": request.form["inputBenefits"],
         "care_options": request.form["inputCareOptions"],
        #  "wellness_program": request.form["inputWellnessProgram"],
        #  "seek_help": request.form["inputSeekHelp"],
         "anonymity": request.form["inputAnonymity"],
         "leave": request.form["inputLeave"],
        #  "mental_health_consequence": request.form["inputMentalHealthConsequence"],
        #  "phys_health_consequence": request.form["inputPhysHealthConsequence"],
         "coworkers": request.form["inputColleague"],
         "supervisor": request.form["inputColleague"],
        #  "mental_health_interview": request.form["inputMentalHealthInterview"],
        #  "phys_health_interview": request.form["inputPhysHealthInterview"],
        #  "mental_vs_physical": request.form["inputMentalVsPhysical"],
         "obs_consequence": request.form["inputObsConsequence"],
     }]
    df = DataFrame.from_records(data)
    x = ct.transform(df)
    y = model.predict(x)
    treatment = le.inverse_transform(y)[0]

    dataf = pd.read_csv(r"C:\Users\AKHIL\OneDrive\Documents\PROJECT_COLLEGE\Mental-Health-Prediction-Using-Machine-Learning\Mental-Health-Prediction-Using-Machine-Learning\processed_data.csv")
    print(dataf.columns)
    yes_counts = []
    no_counts = []
    # Categories on X-axis
    labels = ['A','B','C','D','E','F','G','H','I','J']
    cat = ['Age','Gender', 'self_employed', 'family_history','remote_work','care_options', 
           'anonymity', 'leave', 'coworkers', 'supervisor','obs_consequence']
    for i in cat:
        ycount = dataf[(dataf['treatment'] == 'Yes') & (dataf[i] == data[0][i])].shape[0]
        ncount = dataf[(dataf['treatment'] == 'No') & (dataf[i] == data[0][i])].shape[0]
        yes_counts.append(ycount)
        no_counts.append(ncount)

    yes_counts[8] += yes_counts[9]
    yes_counts.pop(9)
    
    no_counts[8] += no_counts[9]
    no_counts.pop(9)
    # Bar width
    width = 0.35  

    # Position of bars on X-axis
    x = np.arange(len(labels)) 

    # Plotting the bars
    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, yes_counts, width, label='Yes', color='skyblue')
    plt.bar(x + width/2, no_counts, width, label='No', color='orange')

    # Labeling
    plt.xlabel('Categories')
    plt.ylabel('Count')
    plt.title('Treatment Counts by Category')
    plt.xticks(x, labels)
    plt.legend()

    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    if(treatment=="Yes"):
        return render_template('treatmentY.html',image_base64 = image_base64)
    else:
        return render_template('treatmentN.html',image_base64 = image_base64)


@app.route('/teampage')
def teampage():
    return 'Welcome to Team page.'


if __name__=="__main__":
    app.run(debug=False, port=8000)