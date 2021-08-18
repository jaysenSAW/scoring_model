from flask import Flask, render_template, request, redirect, url_for, flash
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm, Form
from wtforms import StringField, SubmitField, TextField, IntegerField, FloatField
from wtforms import TextAreaField, SubmitField, RadioField, SelectField
from wtforms import validators, ValidationError
from wtforms.validators import DataRequired
import pandas as pd
import numpy as np
import json
import plotly
import plotly.express as px
import joblib

# load the model from disk
mdl = joblib.load("tree_decision")


class ContactForm(Form):
    Credit_ask = FloatField("Credit ask")
    AMT_income = FloatField("Income (during the year)")
    Years_refund = FloatField("Number of years for payment credit")
    Age = FloatField("Age")
    Age_car = FloatField("Car's age")
    YEARS_EMPLOYED = FloatField("Years of Employment")
    CNT_FAM_MEMBERS = FloatField("Number of familly")
    CREDIT_active = FloatField("Number of current credit")
    CREDIT_MEAN_active = FloatField("Amount for current credit")
    CREDIT_MEAN_closed = FloatField("Mean amount for closed credit")
    CREDIT_MEAN_OVERDUE_closed = FloatField("Overdue amount for closed credit")
    proportion_OVERDUE_closed = FloatField("Ratio of credit with overdue")
    submit = SubmitField("Send")


app = Flask(__name__)
app.secret_key = 'development key'

def predict_credit(data):
    AMT_income = np.log10(float(data['AMT_income']) + 1)
    Credit_ask = np.log10(float(data['Credit_ask']) + 1)
    Years_refund = float(data['Years_refund'])
    #tmp = Credit_ask/Years_refund)
    #Loan_annuity = np.log10(tmp + 1)
    #INCOME_ANNUITY_RATIO = data['AMT_income'])/tmp
    Age = float(data['Age'])
    #Employed = float(data['YEARS_EMPLOYED'])
    Age_car = float(data['Age_car'])
    CREDIT_active = float(data['CREDIT_active'])
    CREDIT_MEAN_active = np.log10(float(data['CREDIT_MEAN_active']) + 1)
    CREDIT_MEAN_OVERDUE_closed = float(data['CREDIT_MEAN_OVERDUE_closed'])
    CREDIT_MEAN_closed = np.log10(float(data['CREDIT_MEAN_closed']) + 1)
    proportion_OVERDUE_closed = float(data['proportion_OVERDUE_closed'])
    CNT_FAM_MEMBERS = data['CNT_FAM_MEMBERS']

    #dt = np.array([AMT_income, Loan_annuity, Age, Employed, Age_car,
    #CREDIT_active, CREDIT_MEAN_active, CREDIT_MEAN_OVERDUE_closed,
    #CREDIT_MEAN_closed, proportion_OVERDUE_closed, Credit_ask, Years_refund,
    #INCOME_ANNUITY_RATIO, CNT_FAM_MEMBERS])
    #print(dt)
    #y_pred = mdl.predict(dt.reshape(1, -1))
    #return y_pred[0]

@app.route('/')
def index():
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
                return render_template('prediction_accepted.html', form = form)
            else:
                return render_template('prediction_refused.html', form = form)
    elif request.method == 'GET':
        return render_template('question.html', form = form)



@app.route('/chart1')
def chart1():
    df = pd.DataFrame({
        "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
        "Amount": [4, 1, 2, 2, 4, 5],
        "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
    })

    fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    header="Fruit in North America"
    description = """
    A academic study of the number of apples, oranges and bananas in the cities of
    San Francisco and Montreal would probably not come up with this chart.
    """
    return render_template('notdash2.html', graphJSON=graphJSON, header=header,description=description)

@app.route('/chart2')
def chart2():
    df = pd.DataFrame({
        "Vegetables": ["Lettuce", "Cauliflower", "Carrots", "Lettuce", "Cauliflower", "Carrots"],
        "Amount": [10, 15, 8, 5, 14, 25],
        "City": ["London", "London", "London", "Madrid", "Madrid", "Madrid"]
    })

    fig = px.bar(df, x="Vegetables", y="Amount", color="City", barmode="stack")

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    header="Vegetables in Europe"
    description = """
    The rumor that vegetarians are having a hard time in London and Madrid can probably not be
    explained by this chart.
    """
    return render_template('notdash2.html', graphJSON=graphJSON, header=header,description=description)


if __name__ == '__main__':
   app.run(debug = True)
