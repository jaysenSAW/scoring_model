# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import argparse


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


df = px.data.tips()
days = df.day.unique()

#app = dash.Dash(__name__)
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#####################################################################
#                   style sheets parameters                         #
#####################################################################

#app.layout = html.Div([
#    dcc.Dropdown(
#        id="dropdown",
#        options=[{"label": x, "value": x} for x in days],
#        value=days[0],
#        clearable=False,
#    ),
#    dcc.Graph(id="bar-chart"),
#])


# For bad loan we look sex ratio
tmp1 = good_loan['CODE_GENDER_M'].value_counts(normalize = True)
tmp2 = bad_loan['CODE_GENDER_M'].value_counts(normalize = True)

fig = px.box(dt, x="TARGET", y="AGE")
fig.update_layout(
    yaxis_title='Age',
    xaxis_title="Credit acceptance (YES/NO)",
    title_text="Candidats's age", title_x=0.5
)

hist = px.histogram(dt, x="CNT_FAM_MEMBERS", color="TARGET",
histnorm='probability density', nbins=35, marginal="box")
hist.update_layout(
    xaxis_title="Family members"
)

hist_ratio = px.histogram(dt, x="Ratio_Credit_limit", color="TARGET",
histnorm='probability density', marginal="box", nbins=35)
hist_ratio.update_layout(
    xaxis_title="Ratio credit and income"
)


app.layout = html.Div(children=[
    html.H1(children='Data preview'),
    generate_table(dt),
    html.H1(children='Comparison between good and bad loan',
    style={
            'textAlign': 'center'
        }),
    html.P(children="For good loan sex ratio between femal and male are {0:.2f}/{1:.2f}.".format(
    tmp1[0], tmp1[1])),
    html.P(children="For bad loan sex ratio between femal and male are {0:.2f}/{1:.2f}.".format(
    tmp2[0], tmp2[1])),

    html.P("Names:"),


    html.H1(children='Age repartion for good and bad loan'),
    html.Div(children='''
        Age doesn't have an impact about the capacity to refund a loan.
    '''),
    dcc.Graph(
        id='example-graph',
        figure=fig
    ),
    html.Div(children='''
            Age doesn't have an impact about the capacity to refund a loan.
    '''),
    dcc.Graph(
        id='hist-CNT_FAM_MEMBERS',
        figure=hist
    ),

    dcc.Graph(
        id='hist-ratio-credit',
        figure=hist_ratio
    )
    ]
)




if __name__ == '__main__':
    app.run_server(debug=True)
