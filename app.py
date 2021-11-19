from flask import Flask, render_template, request, redirect, url_for, flash
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm, Form
from wtforms import StringField, SubmitField, TextField, IntegerField, FloatField
from wtforms import TextAreaField, SubmitField, RadioField, SelectField
from wtforms import validators, ValidationError
from wtforms.validators import NumberRange
from wtforms.validators import DataRequired
import pandas as pd
#import numpy as np
import math
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
 'NAME_CONTRACT_TYPE_Cash loans',
 'CREDIT_active',
 'CREDIT_MEAN_OVERDUE_active',
 'CREDIT_MEAN_active',
 'proportion_OVERDUE_active',
 'CREDIT_MEAN_OVERDUE_closed',
 'CREDIT_MEAN_closed',
 'proportion_OVERDUE_closed',
 'CREDIT_ask',
 'Number_years_Loan_Theorical',
 'INCOME_ANNUITY_RATIO',
 'CNT_FAM_MEMBERS']


global df
global shap_values
df = pd.read_csv('data/register_customer_test.csv')
df = df.sample(n=5000, random_state=42)

global df_candidats
df_candidats = pd.read_csv('data/register_customer_test.csv')
df_candidats = df_candidats.sort_values(by=['SK_ID_CURR'])

global user_data #customer info

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
    proportion_OVERDUE_active = FloatField('proportion_OVERDUE_active')
    NAME_CONTRACT_TYPE_Cash_loans = FloatField("NAME_CONTRACT_TYPE_Cash_loans")
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
    submit = SubmitField("Send")

class RegisteredForm(Form):
    # ID_customers = FloatField("ID_customers", validators=[NumberRange(min=10000, message='Customer id')])
    Credit_ask = FloatField("Credit ask", validators=[NumberRange(min=1000, message="plus d'argent" )])
    Years_refund = FloatField("Number of years for payment credit", validators=[NumberRange(min=0)])
    # submit = SubmitField("Send")

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

def shap_values_candidat(dt):
    Xtest2 = dt.to_numpy()
    shap_values2 = explainer.shap_values(dt.to_numpy())
    print('shap graphic for customer')
    if os.path.exists("static/temp2.jpg"):
        print(("remove file"))
        os.remove("static/temp2.jpg")
    shap.force_plot(
    explainer.expected_value, shap_values2[0,:], Xtest2[0,:],
    feature_names=dt.columns, show=False, matplotlib=True
    ).savefig('static/temp2.jpg')
    print("image saved")



def predict_credit(data):
    print(data.keys())
    #transform some features by using log10
    AMT_income = math.log10(float(data['AMT_income']) + 1)
    Credit_ask = math.log10(float(data['Credit_ask']) + 1)
    Years_refund = float(data['Years_refund'])
    tmp = float(data['Credit_ask'])/Years_refund
    Loan_annuity = math.log10(tmp + 1)
    INCOME_ANNUITY_RATIO = float(data['AMT_income'])/tmp
    Age = float(data['Age'])
    Employed = float(data['YEARS_EMPLOYED'])
    proportion_OVERDUE_active = float(data['proportion_OVERDUE_active'])
    CREDIT_active = float(data['CREDIT_active'])
    CREDIT_MEAN_OVERDUE_active = math.log10(float(data['CREDIT_MEAN_active']) + 1)
    CREDIT_MEAN_active = math.log10(float(data['CREDIT_MEAN_active']) + 1)
    CREDIT_MEAN_OVERDUE_closed = float(data['CREDIT_MEAN_OVERDUE_closed'])
    CREDIT_MEAN_closed = math.log10(float(data['CREDIT_MEAN_closed']) + 1)
    proportion_OVERDUE_closed = float(data['proportion_OVERDUE_closed'])
    CNT_FAM_MEMBERS = float(data['CNT_FAM_MEMBERS'])
    NAME_CONTRACT_TYPE_Cash_loans = float(data['NAME_CONTRACT_TYPE_Cash_loans'])
    global dt
    dt = [AMT_income, Loan_annuity, Age, Employed,
    NAME_CONTRACT_TYPE_Cash_loans,
    CREDIT_active, CREDIT_MEAN_OVERDUE_active,
    CREDIT_MEAN_active, proportion_OVERDUE_active,
    CREDIT_MEAN_OVERDUE_closed, CREDIT_MEAN_closed, proportion_OVERDUE_closed,
    Credit_ask, Years_refund, INCOME_ANNUITY_RATIO, CNT_FAM_MEMBERS]
    print("***DATA FRAME FOR NEW CUSTOMER***")
    print(dt)
    y_pred = mdl.predict_proba(dt.reshape(1, -1))[0][1]
    print("score for reject credit {0}".format(y_pred))
    #save raw data
    dt = [AMT_income, Loan_annuity, Age, Employed,
    NAME_CONTRACT_TYPE_Cash_loans,
    CREDIT_active, CREDIT_MEAN_OVERDUE_active,
    CREDIT_MEAN_active, proportion_OVERDUE_active,
    CREDIT_MEAN_OVERDUE_closed, CREDIT_MEAN_closed, proportion_OVERDUE_closed,
    Credit_ask, Years_refund, INCOME_ANNUITY_RATIO, CNT_FAM_MEMBERS]
    data = {"AMT_INCOME_TOTAL" : [float(data['AMT_income'])],
    "AMT_ANNUITY" : [tmp],
    "AGE" : [Age],
    "YEARS_EMPLOYED" : [Employed],
    "NAME_CONTRACT_TYPE_Cash loans" : [NAME_CONTRACT_TYPE_Cash_loans],
    "proportion_OVERDUE_active" : [proportion_OVERDUE_active],
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
    shap_values_candidat(dt)
    ###################################
    #shap.summary_plot(shap_values, dt.to_numpy(), feature_names=feat, show=False)
    #figure.set_size_inches(8, 8)
    #plt.savefig(fname = "static/temp2.jpg", dpi = 100, pad_inches = 0, bbox_inches='tight')
    #plt.close()
    return y_pred


@app.route('/')
def index():
    #init_url()
    return render_template('index.html')

def close_candidats(df_candidats, id):
    """
    Return subdataframe with individu similar to the candidat.
    Arguments:
        df_candidats: dataframe with all individus
        id: candidat's id
    Return:
        sub dataframe
    """
    tmp = df_candidats[df_candidats["SK_ID_CURR"] == id]
    val = []
    col_name = []
    for col in df_candidats.filter(regex=("NAME_EDUCATION.*")).columns:
      col_name.append(col)
      val.append(tmp[col].to_numpy()[0])
    for col in df_candidats.filter(regex=("OCCUPATION_TYPE.*")).columns:
      col_name.append(col)
      val.append(tmp[col].to_numpy()[0])
    tmp_df = df_candidats[
    (df_candidats[col_name[0]] == val[0]) &
    (df_candidats[col_name[1]] == val[1]) &
    (df_candidats[col_name[2]] == val[2]) &
    (df_candidats[col_name[3]] == val[3]) &
    (df_candidats[col_name[4]] == val[4]) &
    (df_candidats[col_name[5]] == val[5]) &
    (df_candidats[col_name[6]] == val[6]) &
    (df_candidats[col_name[7]] == val[7]) &
    (df_candidats[col_name[8]] == val[8]) &
    (df_candidats[col_name[9]] == val[9]) &
    (df_candidats[col_name[10]] == val[10]) &
    (df_candidats[col_name[11]] == val[11]) &
    (df_candidats[col_name[12]] == val[12]) &
    (df_candidats[col_name[13]] == val[13]) &
    (df_candidats[col_name[14]] == val[14]) &
    (df_candidats[col_name[15]] == val[15]) &
    (df_candidats[col_name[16]] == val[16])
      ]
    tmp_df.to_csv("data/sub_df.csv")
    return tmp_df

def score_credit(dt, credit_ask, Years_refund):
    print("*"*50)
    print("Compute score")
    print(dt.keys())
    data = dt.copy()
    AMT_income = math.log10(float(data['AMT_INCOME_TOTAL']) + 1)
    Credit_ask = math.log10(credit_ask + 1)
    Years_refund = float(Years_refund)
    tmp_annuity = credit_ask/Years_refund
    Loan_annuity = math.log10(tmp_annuity + 1)
    INCOME_ANNUITY_RATIO = float(data['AMT_INCOME_TOTAL'])/tmp_annuity
    Age = float(data['AGE'])
    Employed = float(data['YEARS_EMPLOYED'])
    NAME_CONTRACT_TYPE_Cash_loans = float(data['NAME_CONTRACT_TYPE_Cash loans'])
    proportion_OVERDUE_active = float(data['proportion_OVERDUE_active']),
    print("proportion_OVERDUE_active")
    print(type(proportion_OVERDUE_active))
    CREDIT_active = float(data['CREDIT_active'])
    CREDIT_MEAN_OVERDUE_active = math.log10(float(data['CREDIT_MEAN_active']) + 1)
    CREDIT_MEAN_active = math.log10(float(data['CREDIT_MEAN_active']) + 1)
    CREDIT_MEAN_OVERDUE_closed = float(data['CREDIT_MEAN_OVERDUE_closed'])
    CREDIT_MEAN_closed = math.log10(float(data['CREDIT_MEAN_closed']) + 1)
    proportion_OVERDUE_closed = float(data['proportion_OVERDUE_closed'])
    CNT_FAM_MEMBERS = float(data['CNT_FAM_MEMBERS'])
    dt_tmp = [AMT_income, Loan_annuity, Age, Employed,
    NAME_CONTRACT_TYPE_Cash_loans, CREDIT_active, CREDIT_MEAN_OVERDUE_active,
    CREDIT_MEAN_active, proportion_OVERDUE_active[0], CREDIT_MEAN_OVERDUE_closed,
    CREDIT_MEAN_closed, proportion_OVERDUE_closed,
    Credit_ask, Years_refund, INCOME_ANNUITY_RATIO, CNT_FAM_MEMBERS]
    print(dt_tmp)
    print("9*9"*50)
    y_pred = mdl.predict_proba(dt_tmp.reshape(1, -1))[0][1]
    col = ["AMT_income", "Loan_annuity", "Age", "Employed", "NAME_CONTRACT_TYPE_Cash_loans",
    "CREDIT_active", "CREDIT_MEAN_OVERDUE_active",
    "CREDIT_MEAN_active", "proportion_OVERDUE_active",
    "CREDIT_MEAN_OVERDUE_closed", "CREDIT_MEAN_closed", "proportion_OVERDUE_closed",
    "Credit_ask", "Years_refund", "INCOME_ANNUITY_RATIO", "CNT_FAM_MEMBERS"]
    print(len(col))
    dt_tmp = pd.DataFrame(data = [dt_tmp], columns = col)
    return y_pred, dt_tmp
    # return y_pred, pd.DataFrame(data = dt_tmp, columns = ["AMT INCOME TOTAL",
    # "Loan annuity", "Age", "Year employed", "Age car", "CREDIT active",
    # "CREDIT_MEAN_OVERDUE_active", "CREDIT MEAN_OVERDUE closed"])

def give_edu_job(user_data):
    education = "NAME_EDUCATION_TYPE_Other"
    for ed in user_data.filter(regex=("NAME_EDUCATION.*")).columns:
        if user_data[ed].to_numpy()[0] == 1:
            education = ed
    print(education)
    job_type = "OCCUPATION_TYPE_Other"
    for job in user_data.filter(regex=("OCCUPATION_TYPE.*")).columns:
        if user_data[job].to_numpy()[0] == 1:
            job_type = job
    print(job_type)
    return education, job_type

def pieplot_familly(df = df):
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

def hist_age(df = df):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df[df["TARGET"] == 1]["AGE"].values,
        histnorm='probability density',
        name='Rejected'))
    fig.add_trace(go.Histogram(
        x=df[df["TARGET"] == 0]["AGE"].values,
        histnorm='probability density',
        name='Accepted'))
    # Overlay both histograms
    fig.update_layout(barmode='overlay',
                     title_text="Candidats's age density plot",
                     xaxis_title_text='Age',
                     yaxis_title_text= 'Probabily density')
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.7)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def hist_income(df = df):
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

def plot_ratio(df = df):
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


def summary_data_info(dataFrame):
    graphJSON = pieplot_familly(dataFrame)
    header="Comparison against similar candidat"
    histAge = hist_age(dataFrame)
    histJSON = hist_income(dataFrame)
    description1 = "Income "
    scatterJSON = plot_ratio(dataFrame)

    if os.path.isfile("data/customer_info.csv"):
        print("add customer point !")
        new_dt = pd.read_csv('data/customer_info.csv')
        description1 = "You have {0:.1f} members in your familly".format(new_dt.loc[0, "CNT_FAM_MEMBERS"])
        description2 = "Ratio income/loan annuity is important to determine \
        if candidate can pay its loans. Your ratio is {0:.2f}".format(new_dt.loc[0,"INCOME_ANNUITY_RATIO"])
        description3 = "Income distribution for accepted and rejected candidats.\
        Your income is {0:.2f}".format(new_dt.loc[0,"AMT_INCOME_TOTAL"])
        description4 = "Candidat is {0:.1f} years old".format(new_dt.loc[0, "AGE"])
    else:
        description1 = "Rejected loan are more present for familly with more than 4 member"
        description2 = "Ratio income/loan annuity inform how loan could impact candidate's purchasing power.\n\
        Ratio bellow 1 means loan annuity is more important than total income. higher is the ration and more flexibility has the candidate."
        description3 = "Income distribution for accepted and rejected candidats"
        description4 = "Candidats's age distribution for accepted and rejected candidats"
    return [graphJSON, scatterJSON, histJSON, histAge], header, [description1, description2, description3, description4]

def parse_data(request, df_candidats):
    """
    Parse registered_customers.html and return customer data
    Arguments
        request: wtforms flask object
        df_candidats: dataframe with all customers
    """
    users_id = int(request.form.to_dict()["users_id"])
    credit_ask = float(request.form.to_dict()['Credit_ask'])
    Years_refund = float(request.form.to_dict()['Years_refund'])
    # ID_customers = float(request.form.to_dict()['ID_customers'])
    print("*****all informations*****")
    print(request.form.to_dict()) #all informations
    # print(ID_customers, users_id)
    user_data = df_candidats[df_candidats["SK_ID_CURR"] == users_id]
    print("*****LOAD USER*****")
    user_data.to_csv("data/customer_info.csv")
    print("*****USER LOADED*****")
    #return score model and customer's dataframe with 15 features
    score, user_data_shap = score_credit(user_data, credit_ask, Years_refund)
    shap_values_candidat(user_data_shap)
    sub_df = close_candidats(df_candidats, users_id)
    return user_data, score, sub_df

@app.route('/registered_customers', methods = ['GET', 'POST'])
def registered_customer():
    print("*"*100)
    form = RegisteredForm()
    users = df_candidats.iloc[0:500, 0].tolist()
    if request.method != 'POST':
        print("method is not post")
        return render_template('registered_customers.html', form = form, users = users)
    if form.validate() == False:
        flash('All fields are required.')
        return render_template('registered_customers.html', form = form, users = users)
    print("method is post")
    user_data, score, sub_df = parse_data(request, df_candidats)
    education, job_type = give_edu_job(user_data)
    print(education)
    graphJSON, header, description = summary_data_info(sub_df)
    if score >= 0.5:
        print("score > 0.5")
        return render_template('prediction_refused.html',
        form = form, education = education, job_type = job_type,
        graphJSON=graphJSON, header=header, description = description ,
        score = round(score*100,2))
    print("score < 0.5")
    return render_template('prediction_accepted.html',
    form = form, education = education, job_type = job_type,
    graphJSON=graphJSON, header=header, description = description ,
    score = round(score*100,2))


def close_candidats2(df_candidats, request):
    tmp_df = df_candidats[
    (df_candidats[request.form.to_dict()["ed_id"]] == 1) &
    (df_candidats[request.form.to_dict()["job_id"]] == 1)
    ]
    tmp_df.to_csv("data/sub_df.csv")
    tmp_df.to_csv("data/sub_df.csv")
    return tmp_df

@app.route('/contact', methods = ['GET', 'POST'])
def contact():
    form = ContactForm()
    educations = df_candidats.filter(regex=("NAME_EDUCATION.*")).columns
    job_type = df_candidats.filter(regex=("OCCUPATION_TYPE.*")).columns
    if request.method == 'POST':
        if form.validate() == False:
            flash('All fields are required.')
            return render_template('question.html', form = form,
            educations = educations, job_type = job_type)
        else:
            sub_df = close_candidats2(df_candidats, request)
            print("form validated !")
            data = request.form.to_dict()
            print(data)
            score = predict_credit(data)
            graphJSON, header, description = summary_data_info(sub_df)
            education = request.form.to_dict()["ed_id"]
            job_type = request.form.to_dict()["job_id"]
            if score >= 0.5:
                return render_template('prediction_refused.html',
                form = form, education = education, job_type = job_type,
                graphJSON=graphJSON, header=header, description = description ,
                score = round(score*100,2))
            else:
                return render_template('prediction_accepted.html', form = form,
                score = round(score*100,2))
    elif request.method == 'GET':
        return render_template('question.html', form = form,
        educations = educations, job_type = job_type)



if __name__ == '__main__':
   app.run(debug = True)
