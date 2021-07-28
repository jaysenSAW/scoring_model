# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import re
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import argparse
import plotly.graph_objects as go
import numpy as np
import joblib

# load the model from disk
mdl = joblib.load("logisitc_model1_gridSearch.joblib")


def generate_table(dataframe, max_rows=3):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

#####################################################################
#                   style sheets parameters                         #
#####################################################################

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


#####################################################################
#                           Load data                                #
#####################################################################
parser = argparse.ArgumentParser(description='')
parser.add_argument('-g', action="store", dest="g", type=str,
help="data file (CSV).\nExample: data.csv", default ="data_for_server.csv")

#Load data

dt = pd.read_csv('data_for_server.csv')
bad_loan = dt[dt["TARGET"] == 1 ]
good_loan = dt[dt["TARGET"] == 0 ]
targets = dt.TARGET.unique()


df = px.data.tips()
days = df.day.unique()

#app = dash.Dash(__name__)
#####################################################################
#                   style sheets parameters                         #
#####################################################################
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)








#############################################################################
#                                  graphics                                 #
#############################################################################

credit_status=['Credit accepted', 'Credit rejected']
val = dt["TARGET"].value_counts(normalize=True) *100
fig1 = go.Figure([go.Bar(x=credit_status, y=[val[0], val[1]])])
fig1.update_layout(
    yaxis_title='Pourcentage',
    title_text="Credit acceptance", title_x=0.5
)


#Education type
labels = []
ed_accepted = np.array([])
ed_rejected = np.array([])
for col in dt.filter(regex=("NAME_EDUCATION.*")).columns:
    ed_accepted = np.append(ed_accepted, good_loan[col].sum())
    ed_rejected = np.append(ed_rejected, bad_loan[col].sum())
    labels.append(re.sub('NAME_EDUCATION_TYPE_', '', col))
ed_accepted = ed_accepted/np.sum(ed_accepted) * 100
ed_rejected = ed_rejected/np.sum(ed_rejected) * 100

trace1 = go.Bar(    #setup the chart for Resolved records
    x=labels, #x for Resolved records
    y=ed_accepted,#y for Resolved records
    marker_color=px.colors.qualitative.Dark24[0],  #color
    text=ed_accepted, #label/text
    textposition="outside", #text position
    name="Accepted", #legend name
)
trace2 = go.Bar(   #setup the chart for Unresolved records
    x=labels,
    y=ed_rejected,
    text=ed_rejected,
    marker_color=px.colors.qualitative.Dark24[1],
    textposition="outside",
    name="Rejected",
)

data = [trace1, trace2] #combine two charts/columns
layout = go.Layout(barmode="group", title="Accepted vs Rejected") #define how to display the columns
fig2 = go.Figure(data=data, layout=layout)
fig2.update_layout(
    title=dict(x=0.5), #center the title
    xaxis_title="Education type",#setup the x-axis title
    yaxis_title="Pourcentage", #setup the x-axis title
    margin=dict(l=20, r=20, t=60, b=20),#setup the margin
    paper_bgcolor="aliceblue", #setup the background color
    yaxis_range=[0, 100]
)
fig2.update_traces(texttemplate="%{text:.2s}") #text formart



fig14 = px.box(dt, x="TARGET", y="AMT_INCOME_TOTAL")
fig14.update_layout(
    yaxis_title='INCOME TOTAL',
    xaxis_title="Credit acceptance (YES/NO)",
    title_text="Candidats's income", title_x=0.5
)


hist = px.histogram(dt, x="CNT_FAM_MEMBERS", color="TARGET",
histnorm='probability density', nbins=35, marginal="box")
hist.update_layout(
    xaxis_title="Family members"
)

label_x = {'INCOME' :'AMT_INCOME_TOTAL',
            'ANNUITY': 'AMT_ANNUITY',
            "CANDIDAT'S AGE" : 'AGE',
            'YEARS EMPLOYED': 'YEARS_EMPLOYED',
            'FAMILIY MEMBERS': 'CNT_FAM_MEMBERS'
            }

label_y = {'Credit status' :'TARGET'
            }

ALLOWED_TYPES = (
    "text", "number", "password", "email", "search",
    "tel", "url", "range", "hidden",
)

#############################################################################
#                                   layout                                 #
#############################################################################

app.layout = html.Div(
    children=[

    dcc.Tabs([
    dcc.Tab(label='Status informations', children=[
        dcc.Input(
        id="age_dcc",
        placeholder='Age',
        type='number',
        value='Age'
        ),
        dcc.Input(
        id="family_number",
        placeholder='Family numbers',
        type='number',
        value='Family number'
        ),

        dcc.Dropdown(
        id="education_type",
        options=[
            {'label': 'Higher education', 'value': 'NAME_EDUCATION_TYPE_Higher education'},
            {'label': 'Incomplete higher', 'value': 'NAME_EDUCATION_TYPE_Incomplete higher'},
            {'label': 'Secondary / secondary special', 'value': 'NAME_EDUCATION_TYPE_Secondary / secondary special'},
            {'label': 'Other', 'value': 'NAME_EDUCATION_TYPE_Other'}

        ],
        value='NAME_EDUCATION_TYPE_Higher education'
        ),

        dcc.Dropdown(
        id="status_type",
        options=[
            {'label': 'Single / not married', 'value': 'Single'},
            {'label': 'Married', 'value': 'Married'},
            {'label': 'Civil marriage', 'value': 'CV'},
            {'label': 'Married', 'value': 'Married'},
            {'label': 'Separated', 'value': 'Separated'},
            {'label': 'Widow', 'value': 'Widow '}
        ],
        value='Single'
        ),

        dcc.Dropdown(
        id="contrat_type",
        options=[
            {'label': 'Cash', 'value': 'NAME_CONTRACT_TYPE_Cash loans'},
            {'label': 'Revolving', 'value': 'NAME_CONTRACT_TYPE_Revolving loans'}
        ],
        value='NAME_CONTRACT_TYPE_Cash loans'
        ),

    ]),
    dcc.Tab(label='Incomes Informations', children=[
        dcc.Input(
        id="income",
        placeholder='Income for one year',
        type='number',
        value='Income for one year'
        ),

        dcc.Input(
        id="annuity",
        placeholder='Monthly payment for loan',
        type='number',
        value='Monthly payment for loan'
        ),

        dcc.Input(
        id="years_em",
        placeholder='Years employed',
        type='number',
        value='Years employed'
        ),

        dcc.Input(
        id="car_age",
        placeholder="Owner's car age",
        type='number',
        value="Owner's car age"
        ),
        dcc.Dropdown(
        id="occupation_type",
        options=[
            {'label': 'Accountants', 'value': 'OCCUPATION_TYPE_Accountants'},
            {'label': 'Core staff', 'value': 'OCCUPATION_TYPE_Core staff'},
            {'label': 'Drivers', 'value': 'OCCUPATION_TYPE_Drivers'},
            {'label': 'HR staff', 'value': 'OCCUPATION_TYPE_HR staff'},
            {'label': 'High skill tech staff', 'value': 'OCCUPATION_TYPE_High skill tech staff'},
            {'label': 'Laborers', 'value': 'OCCUPATION_TYPE_Laborers'},
            {'label': 'Managers', 'value': 'OCCUPATION_TYPE_Managers'},
            {'label': 'Medecine staff', 'value': 'OCCUPATION_TYPE_Medecine staff'},
            {'label': 'Other', 'value': 'OCCUPATION_TYPE_Other'}
        ],
        value='OCCUPATION_TYPE_Accountants'
        ),

    ]),
    dcc.Tab(label='Financial history', children=[

        dcc.Input(
        id="cdt_amount_active",
        placeholder='# credit active',
        type='number',
        value='# of current credit'
        ),
        dcc.Input(
        id="cdt_active",
        placeholder='Current credit payment',
        type='number',
        value='Current credit payment'
        ),
        dcc.Input(
        id="cdt_overdue_active",
        placeholder='Overdue in current credit',
        type='number',
        value='Overdue in current credit payment'
        ),

        dcc.Input(
        id="cdt_amount_sold",
        placeholder='# of solded credit',
        type='number',
        value='# of solded credit'
        ),
        dcc.Input(
        id="cdt_sold",
        placeholder='Amount of money solded',
        type='number',
        value='Amount of solded credit'
        ),
        dcc.Input(
        id="cdt_overdue_sold",
        placeholder='Overdue for solded credit',
        type='number',
        value='Overdue for solded credit'
        ),

        dcc.Input(
        id="cdt_amount_closed",
        placeholder='# credit closed',
        type='number',
        value='# of closed credit'
        ),
        dcc.Input(
        id="cdt_closed",
        placeholder='Amount of money for closed credit',
        type='number',
        value='Closed credit payment'
        ),
        dcc.Input(
        id="cdt_overdue_closed",
        placeholder='Overdue for closed credit',
        type='number',
        value='Overdue for closed credit payment'
        ),

        dcc.Input(
        id="cdt_amount_bad",
        placeholder='# credit bad',
        type='number',
        value='# of bad credit'
        ),
        dcc.Input(
        id="cdt_bad",
        placeholder='Amount of money for bad credit',
        type='number',
        value='Bad credit payment'
        ),
        dcc.Input(
        id="cdt_overdue_bad",
        placeholder='Overdue for bad credit',
        type='number',
        value='Overdue for bad credit payment'
        ),

        dcc.Input(
        id="previous_cdt",
        placeholder='Amount of money for previous credit',
        type='number',
        value='Previous amount for credit'
        )

    ]),
    ]),

    html.H1(children='Data preview'),
    html.Hr(),
    html.Div(id="number-out"),
    generate_table(dt),
    html.H1(children='Comparison between good and bad loan',
            style={'textAlign': 'center'}),

    dcc.Graph(id="bar_credit", figure = fig1),

    dcc.Graph(id="bar_edu", figure = fig2),

    #dcc.Graph(id="pie-chart", figure = fig_pie),


    html.H1(children='Distribution comparaison'),
    html.P(children="75% of accepted candidats have an income above {0}$.".format(dt["AMT_INCOME_TOTAL"].quantile(0.25))),

    html.P("x-axis:"),
    dcc.Checklist(
        id='x-axis',
        options=[{'value': label_y[x], 'label': x}
                 for x in label_y.keys()],
        value=['TARGET'],
        labelStyle={'display': 'inline-block'}
    ),
    html.P("y-axis:"),


    dcc.RadioItems(
        id='y-axis',
        options=[{'value': label_x[x], 'label': x}
                 for x in label_x.keys()],
        value='AGE',
        labelStyle={'display': 'inline-block'}
    ),
    dcc.Graph(id="box-plot"),

    html.Div(children='''
            Age doesn't have an impact about the capacity to refund a loan.
    '''),
    dcc.Graph(
        id='hist-CNT_FAM_MEMBERS',
        figure=hist
    ),


    html.H1(children='Prediction')

    ]
)


@app.callback(
    Output("box-plot", "figure"),
    [Input("x-axis", "value"),
     Input("y-axis", "value")])
def generate_chart(x, y):
    fig3 = px.box(dt, x=x, y=y)
    return fig3

@app.callback(
    Output("number-out", "children"),
    Input("income", "value"),
    Input("annuity", "value"),
    Input("age_dcc", "value"),
    Input("years_em", "value"),
    Input("car_age", "value"),
    Input("family_number", "value"),
    Input("education_type", "value"),
    Input("status_type", "value"),
    Input("contrat_type", "value"),
    Input("occupation_type", "value"),
    Input("cdt_amount_active", "value"),
    Input("cdt_active", "value"),
    Input("cdt_overdue_active", "value"),
    Input("cdt_amount_sold", "value"),
    Input("cdt_sold", "value"),
    Input("cdt_overdue_sold", "value"),
    Input("cdt_amount_closed", "value"),
    Input("cdt_closed", "value"),
    Input("cdt_overdue_closed", "value"),
    Input("cdt_amount_bad", "value"),
    Input("cdt_bad", "value"),
    Input("cdt_overdue_bad", "value"),
    Input("previous_cdt", "value")
)
def assign_credit(fincome, fannuity, fage, fyear, fcar, fcnt_family,
val_edu, val_fam_status, val_contrat_type, val_occupation,
fcdt_active, fcdt_overdue_active, fcredit_mean_active,
fcdt_sold,  fcdt_overdue_sold, fcredit_mean_sold,
fcdt_closed,  fcdt_overdue_closed, fcredit_mean_closed,
fcdt_bad,  fcdt_overdue_bad, fcredit_mean_bad, fprevious):
    col = ['AMT_INCOME_TOTAL', 'AMT_ANNUITY', 'AGE', 'YEARS_EMPLOYED',
    'OWN_CAR_AGE', 'CNT_FAM_MEMBERS', 'NAME_EDUCATION_TYPE_Higher education',
    'NAME_EDUCATION_TYPE_Incomplete higher', 'NAME_EDUCATION_TYPE_Other',
    'NAME_EDUCATION_TYPE_Secondary / secondary special',
    'NAME_FAMILY_STATUS_Civil marriage', 'NAME_FAMILY_STATUS_Married',
    'NAME_FAMILY_STATUS_Separated', 'NAME_FAMILY_STATUS_Single / not married',
    'NAME_FAMILY_STATUS_Widow', 'NAME_CONTRACT_TYPE_Cash loans',
    'NAME_CONTRACT_TYPE_Revolving loans', 'OCCUPATION_TYPE_Accountants',
    'OCCUPATION_TYPE_Core staff', 'OCCUPATION_TYPE_Drivers',
    'OCCUPATION_TYPE_HR staff', 'OCCUPATION_TYPE_High skill tech staff',
    'OCCUPATION_TYPE_IT staff', 'OCCUPATION_TYPE_Laborers',
    'OCCUPATION_TYPE_Managers', 'OCCUPATION_TYPE_Medicine staff',
    'OCCUPATION_TYPE_Other', 'OCCUPATION_TYPE_Private service staff',
    'OCCUPATION_TYPE_Realty agents', 'OCCUPATION_TYPE_Sales staff',
    'CREDIT_active', 'CREDIT_MEAN_OVERDUE_active', 'CREDIT_MEAN_active', 'proportion_OVERDUE_active',
    'CREDIT_sold', 'CREDIT_MEAN_OVERDUE_sold', 'CREDIT_MEAN_sold', 'proportion_OVERDUE_sold',
    'CREDIT_closed', 'CREDIT_MEAN_OVERDUE_closed', 'CREDIT_MEAN_closed', 'proportion_OVERDUE_closed',
    'CREDIT_bad', 'CREDIT_MEAN_OVERDUE_bad', 'CREDIT_MEAN_bad', 'proportion_OVERDUE_bad',
    'previous_CREDIT', 'Number_years_Loan_Theorical', 'AMT_CREDIT_ANNUITY_RATIO']
    tmp = {}
    for c in col:
        tmp[c] = [0]
    tmp['AMT_INCOME_TOTAL'] = fincome
    tmp["AMT_ANNUITY"] = fannuity
    tmp["AGE"] = fage
    tmp["YEARS_EMPLOYED"] = fyear
    tmp['OWN_CAR_AGE'] = fcar
    tmp['CNT_FAM_MEMBERS'] = fcnt_family
    tmp[val_edu] = 1
    tmp[val_fam_status] = 1
    tmp[val_contrat_type] = 1
    tmp[val_occupation] = 1
    tmp['CREDIT_active'] = fcdt_active
    tmp['CREDIT_MEAN_OVERDUE_active'] = fcdt_overdue_active
    tmp["CREDIT_MEAN_active"] = fcredit_mean_active
    tmp["CREDIT_sold"] = fcdt_sold
    tmp["CREDIT_MEAN_OVERDUE_sold"] = fcdt_overdue_sold
    tmp['CREDIT_MEAN_sold'] = fcredit_mean_sold
    tmp['CREDIT_closed'] = fcdt_closed
    tmp['CREDIT_MEAN_OVERDUE_closed'] = fcdt_overdue_closed
    tmp['CREDIT_MEAN_closed'] = fcredit_mean_closed
    tmp['CREDIT_bad'] = fcdt_bad
    tmp['CREDIT_MEAN_OVERDUE_bad'] = fcdt_overdue_bad
    tmp['CREDIT_MEAN_bad'] = fcredit_mean_bad
    return "dfalse: {}, dtrue: {}, range: {}".format(val_edu, val_contrat_type, fage)



if __name__ == '__main__':
    app.run_server(debug=True)
