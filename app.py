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
mdl = joblib.load("logisitc_model.joblib")


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
help="data file (CSV).\nExample: data.csv", default ="clean_data.csv")

#Load data

dt = pd.read_csv('clean_data.csv')
bad_loan = dt[dt["TARGET"] == 1 ]
good_loan = dt[dt["TARGET"] == 0 ]
targets = dt.TARGET.unique()

# For bad loan we look sex ratio
tmp1 = good_loan['CODE_GENDER_M'].value_counts(normalize = True)
tmp2 = bad_loan['CODE_GENDER_M'].value_counts(normalize = True)

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

label_y = {'Credit status' :'TARGET',
            'Sex (F/M)': 'CODE_GENDER_M',
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
    html.H1(children='Data preview'),
    generate_table(dt),
    html.H1(children='Comparison between good and bad loan',
            style={'textAlign': 'center'}),

    dcc.Graph(id="bar_credit", figure = fig1),

    dcc.Graph(id="bar_edu", figure = fig2),

    #dcc.Graph(id="pie-chart", figure = fig_pie),

    html.P(children="For good loan sex ratio between femal and male are {0:.2f}/{1:.2f}.".format(
    tmp1[0], tmp1[1])),
    html.P(children="For bad loan sex ratio between femal and male are {0:.2f}/{1:.2f}.".format(
    tmp2[0], tmp2[1])),


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


    html.H1(children='Prediction'),

    dcc.Tabs([
    dcc.Tab(label='Status informations', children=[
        dcc.Input(
        placeholder='Age',
        type='text',
        value='Age'
        ),
        dcc.Input(
        placeholder='Family numbers',
        type='text',
        value='Family number'
        ),

        dcc.Dropdown(
        options=[
            {'label': 'Higher education', 'value': 'NAME_EDUCATION_TYPE_Higher education'},
            {'label': 'Incomplete higher', 'value': 'NAME_EDUCATION_TYPE_Incomplete higher'},
            {'label': 'Secondary / secondary special', 'value': 'NAME_EDUCATION_TYPE_Secondary / secondary special'},
            {'label': 'Other', 'value': 'NAME_EDUCATION_TYPE_Other'}

        ],
        value='NAME_EDUCATION_TYPE_Higher education'
        ),

        dcc.Dropdown(
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
        options=[
            {'label': 'Cash', 'value': 'NAME_CONTRACT_TYPE_Cash loans'},
            {'label': 'Revolving', 'value': 'NAME_CONTRACT_TYPE_Revolving loans'}
        ],
        value='NAME_CONTRACT_TYPE_Cash loans'
        ),

    ]),
    dcc.Tab(label='Incomes Informations', children=[
        dcc.Input(
        placeholder='Income for one year',
        type='text',
        value='Income for one year'
        ),
        dcc.Input(
        placeholder='AMT ANNUITY',
        type='text',
        value='Monthly payment for loan'
        )
    ]),
    dcc.Tab(label='Tab three', children=[
        dcc.Input(
        placeholder='years employed',
        type='text',
        value='years employed'
        )
    ]),
    ])

    ]
)


@app.callback(
    Output("box-plot", "figure"),
    [Input("x-axis", "value"),
     Input("y-axis", "value")])
def generate_chart(x, y):
    fig3 = px.box(dt, x=x, y=y)
    return fig3


if __name__ == '__main__':
    app.run_server(debug=True)
