from flask import Flask, render_template, request, redirect, url_for, flash
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm, Form
from wtforms import StringField, SubmitField, TextField, IntegerField, FloatField
from wtforms import TextAreaField, SubmitField, RadioField, SelectField
from wtforms import validators, ValidationError
from wtforms.validators import NumberRange
from wtforms.validators import DataRequired
import pandas as pd
import numpy as np
import json
import plotly
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import os
import joblib
import glob
import shap
import pickle
import matplotlib.pyplot as plt



# load the model from disk
global mdl
mdl = pickle.load(open("data/tree_decision", "rb"))
global explainer
explainer = pickle.load(open("data/explainer_shap_tree", "rb"))

global feat
feat = ['AMT_INCOME_TOTAL',
 'AMT_ANNUITY',
 'AGE',
 'YEARS_EMPLOYED',
 'OWN_CAR_AGE',
 'CREDIT_active',
 'CREDIT_MEAN_OVERDUE_active',
 'CREDIT_MEAN_active',
 'CREDIT_MEAN_OVERDUE_closed',
 'CREDIT_MEAN_closed',
 'proportion_OVERDUE_closed',
 'CREDIT_ask',
 'Number_years_Loan_Theorical',
 'INCOME_ANNUITY_RATIO',
 'CNT_FAM_MEMBERS']
#clean folder
#if os.path.isfile("data/customer_info.csv"):
#    os.remove("data/customer_info.csv")
#    print("File Removed!")

global df
global shap_values
df = pd.read_csv('data/clean_data2.csv')
df = df.sample(n=5000, random_state=42)

Xtest = df[feat].to_numpy()
shap_values = explainer.shap_values(Xtest, check_additivity=False)
figure = plt.gcf()
#shap.summary_plot(shap_values, Xtest, feature_names=feat, show=False)
shap.summary_plot(shap_values, Xtest, feature_names=feat, show=False)
figure.set_size_inches(8, 8)
plt.savefig(fname = "static/temp.jpg", dpi = 100, pad_inches = 0, bbox_inches='tight')
plt.close()

global url
url = "http://localhost:5000/"
global new_url
new_url = "http://jaysen.pythonanywhere.com/"

class ContactForm(Form):
    Credit_ask = FloatField("Credit ask", validators=[NumberRange(min=0,
    message='Credit above 0 is expected')])
    AMT_income = FloatField("Income (during the year)", validators=[NumberRange(min=0,
    message='Income above 0 is expected')])
    Years_refund = FloatField("Number of years for payment credit")
    Age = FloatField("Age")
    Age_car = FloatField("Car's age")
    YEARS_EMPLOYED = FloatField("Years of Employment")
    CNT_FAM_MEMBERS = FloatField("Number of familly")
    CREDIT_active = FloatField("Number of current credit")
    CREDIT_MEAN_OVERDUE_active = FloatField("Overdue amount for current credit")
    CREDIT_MEAN_active = FloatField("Amount for current credit")
    CREDIT_MEAN_closed = FloatField("Mean amount for closed credit")
    CREDIT_MEAN_OVERDUE_closed = FloatField("Overdue amount for closed credit")
    proportion_OVERDUE_closed = FloatField("Ratio of credit with overdue",
    validators=[NumberRange(min=0, max=1, message='Number between  0 and 1 is \
    expected Ratio of credit with overdue')])
    #env = os.getcwd()
    #print(env)
    #if env.split("\")[2] == 'jayse':
    #    env='http://localhost:5000/contact'
    #else:
    #    env='http://jaysen.pythonanywhere.com/contact'
    submit = SubmitField("Send")


app = Flask(__name__)
app.secret_key = 'development key'

print("!!!!!!!!!!!!!!!!!!!!")
env = os.getcwd()
print(env)
print("!!!!!!!!!!!!!!!!!!!!")


def change_url(file):
    tmp = ""
    with open(file, "r") as lines:
        for line in lines:
            if len(line.split()) != 7:
                tmp += line
            elif line.split("\"")[0].replace(" ","") == '<formaction=':
                tmp += line.replace(url, new_url)
    with open(file, "w") as filout:
        filout.write(tmp)


def init_url():
    print("!!!!!!!!!!!!!!!!!!!!")
    env = os.getcwd()
    if env.split('\\')[2]  == "jayse":
        print("No change")
    else:
        files = glob.glob("templates/*.html")
        for file in files:
            change_url(file)


def predict_credit(data):
    AMT_income = np.log10(float(data['AMT_income']) + 1)
    Credit_ask = np.log10(float(data['Credit_ask']) + 1)
    Years_refund = float(data['Years_refund'])
    tmp = float(data['Credit_ask'])/Years_refund
    Loan_annuity = np.log10(tmp + 1)
    INCOME_ANNUITY_RATIO = float(data['AMT_income'])/tmp
    Age = float(data['Age'])
    Employed = float(data['YEARS_EMPLOYED'])
    Age_car = float(data['Age_car'])
    CREDIT_active = float(data['CREDIT_active'])
    CREDIT_MEAN_OVERDUE_active = np.log10(float(data['CREDIT_MEAN_active']) + 1)
    CREDIT_MEAN_active = np.log10(float(data['CREDIT_MEAN_active']) + 1)
    CREDIT_MEAN_OVERDUE_closed = float(data['CREDIT_MEAN_OVERDUE_closed'])
    CREDIT_MEAN_closed = np.log10(float(data['CREDIT_MEAN_closed']) + 1)
    proportion_OVERDUE_closed = float(data['proportion_OVERDUE_closed'])
    CNT_FAM_MEMBERS = float(data['CNT_FAM_MEMBERS'])
    global dt
    dt = np.array([AMT_income, Loan_annuity, Age, Employed, Age_car,
    CREDIT_active, CREDIT_MEAN_OVERDUE_active, CREDIT_MEAN_active,
    CREDIT_MEAN_OVERDUE_closed, CREDIT_MEAN_closed, proportion_OVERDUE_closed,
    Credit_ask, Years_refund, INCOME_ANNUITY_RATIO, CNT_FAM_MEMBERS])
    print(dt)
    y_pred = mdl.predict(dt.reshape(1, -1))
    print("Result for credit {0}".format(y_pred))
    data = {"AMT_INCOME_TOTAL" : [float(data['AMT_income'])],
    "AMT_ANNUITY" : [tmp],
    "AGE" : [Age],
    "YEARS_EMPLOYED" : [Employed],
    "OWN_CAR_AGE" : [Age_car],
    "CREDIT_active" : [CREDIT_active],
    "CREDIT_MEAN_OVERDUE_active": [float(CREDIT_MEAN_OVERDUE_active)],
    "CREDIT_MEAN_active" : [float(data['CREDIT_MEAN_active'])],
    "CREDIT_MEAN_OVERDUE_closed" : [CREDIT_MEAN_OVERDUE_closed],
    "CREDIT_MEAN_closed" : [float(data['CREDIT_MEAN_closed'])],
    "proportion_OVERDUE_closed" : [proportion_OVERDUE_closed],
    "CREDIT_ask" : [float(data['Credit_ask'])],
    "Number_years_Loan_Theorical" : [Years_refund],
    "INCOME_ANNUITY_RATIO": [INCOME_ANNUITY_RATIO],
    "CNT_FAM_MEMBERS" : [CNT_FAM_MEMBERS] }
    dt = pd.DataFrame(data)
    dt.to_csv("data/customer_info.csv")
    ###################################
    Xtest2 = dt.to_numpy()
    #shap.initjs()
    #figure = plt.gcf()
    shap_values2 = explainer.shap_values(dt.to_numpy())
    shap.force_plot(
    explainer.expected_value, shap_values2[0,:], Xtest2[0,:],
    feature_names=dt.columns, show=False, matplotlib=True
    ).savefig('static/temp2.jpg')
    #shap.summary_plot(shap_values, dt.to_numpy(), feature_names=feat, show=False)
    #figure.set_size_inches(8, 8)
    #plt.savefig(fname = "static/temp2.jpg", dpi = 100, pad_inches = 0, bbox_inches='tight')
    #plt.close()
    return y_pred[0]



def pieplot_familly():
    val = df[df["TARGET"] == 1]["CNT_FAM_MEMBERS"].value_counts(normalize=True)*100
    # Create subplots: use 'domain' type for Pie subplot
    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
    fig.add_trace(go.Pie(labels=val.index, values=val.to_numpy(), name="Family number"),
                  1, 1)
    val = df[df["TARGET"] == 0]["CNT_FAM_MEMBERS"].value_counts(normalize=True)*100
    fig.add_trace(go.Pie(labels=val.index, values=val.to_numpy(), name="Family number"),
                  1, 2)
    # Use `hole` to create a donut-like pie chart
    fig.update_traces(hole=.4, hoverinfo="label+percent+name")

    fig.update_layout(
        title_text="Familly number repartition as function of credit status",
        # Add annotations in the center of the donut pies.
        annotations=[dict(text='Rejected', x=0.17, y=0.5, font_size=20, showarrow=False),
                     dict(text='Accepted', x=0.835, y=0.5, font_size=20, showarrow=False)])
    #fig.show()
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def hist_income():
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df[df["TARGET"] == 1]["INCOME_ANNUITY_RATIO"].values,
        histnorm='probability density',
        name='Rejected'))
    fig.add_trace(go.Histogram(
        x=df[df["TARGET"] == 0]["INCOME_ANNUITY_RATIO"].values,
        histnorm='probability density',
        name='Accepted'))
    # Overlay both histograms
    fig.update_layout(barmode='overlay',
                     title_text="Income annuity density plot",
                     xaxis_title_text='Income annuity',
                     yaxis_title_text= 'Probabily density')
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.7)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def plot_ratio():
        fig = px.scatter(df, x = "AMT_ANNUITY" , y = "INCOME_ANNUITY_RATIO", color="TARGET")
        fig.update_traces(opacity=0.7)
        #fig.show()
        # Overlay both histograms
        fig.update_layout(title_text="Income vs Income annuity ratio",
                         xaxis_title_text='Annuity loan',
                         yaxis_title_text= 'Income annuity ratio')
        if os.path.isfile("data/customer_info.csv"):
            print("add customer point !")
            new_dt = pd.read_csv('data/customer_info.csv')
            fig.add_scatter(x = new_dt["AMT_ANNUITY"].values,
            y = new_dt["INCOME_ANNUITY_RATIO"].values, mode="markers",
                marker=dict(size=20))
        # Reduce opacity to see both histograms
        fig.update_traces(opacity=0.7)
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/')
def index():
    #init_url()
    return render_template('index.html')




@app.route('/contact', methods = ['GET', 'POST'])
def contact():
    form = ContactForm()
    if request.method == 'POST':
        if form.validate() == False:
            flash('All fields are required.')
            return render_template('question.html', form = form)
        else:
            print("ok")
            data = request.form.to_dict()
            print(data)
            if predict_credit(data) == 1:
                return render_template('prediction_refused.html', form = form)
            else:
                return render_template('prediction_accepted.html', form = form)
    elif request.method == 'GET':
        return render_template('question.html', form = form)


@app.route('/familly')
def chart2():
    graphJSON = pieplot_familly()
    header="Number of familly member"
    if dt is None:
        description = """
        Credit's criterions depands on many features these charts show differences between accepted and rejected credits.
        """
    else:
        description = "Credit's criterions depands on many features these charts show differences between accepted and rejected credits.\
In according to form customer has {0} member(s).".format(dt["CNT_FAM_MEMBERS"])
    return render_template('summary.html', graphJSON=graphJSON, header=header,description=description)


@app.route('/income')
def income_plot():
    histJSON = hist_income()
    header="Income annuity ratio distribution for customer with accepted or rejected credit"
    description = ""
    return render_template('summary.html', graphJSON=histJSON, header=header,description=description)


@app.route('/income_loan_ratio')
def income_loan_ratio():
    scatterJSON = plot_ratio()
    header="Annuity loan is high and correlated with income annuity ratio"
    description = ""
    return render_template('summary.html', graphJSON=scatterJSON, header=header,description=description)

@app.route('/summary_data')
def summary_data_info():
    graphJSON = pieplot_familly()
    header="Data summary"
    histJSON = hist_income()
    description1 = "Income "
    scatterJSON = plot_ratio()
    if os.path.isfile("data/customer_info.csv"):
        print("add customer point !")
        new_dt = pd.read_csv('data/customer_info.csv')
        description1 = "Rejected loan are more present for \
        familly with more than 4 member. You have {0:.1f} members".format(new_dt.loc[0, "CNT_FAM_MEMBERS"])
        description2 = "Ratio income/loan annuity is important to determine \
        if candidate can pay its loans. Your ratio is {0:.2f}".format(new_dt.loc[0,"INCOME_ANNUITY_RATIO"])
        description3 = "Income distribution for accepted and rejected candidats.\
        Your income is {0:.2f}".format(new_dt.loc[0,"AMT_INCOME_TOTAL"])
    else:
        description1 = "Rejected loan are more present for familly with more than 4 member"
        description2 = "Ratio income/loan annuity is important to determine if candidate can pay its loans"
        description3 = "Income distribution for accepted and rejected candidats"
    return render_template('summary_data.html', graphJSON=[graphJSON, scatterJSON, histJSON], header=header,description=[description1, description2, description3])


@app.route("/test_shap")
def cusomer_data():
    return render_template('index.html')

if __name__ == '__main__':
   app.run(debug = True)
