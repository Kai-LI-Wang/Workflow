from dash import Dash, dcc, html, Input, Output,dash_table
import os
from PIL import Image
from 已稽查業者 import GraphPlot
import pandas as pd 
from sklearn.metrics import auc
import plotly.express as px 
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import re 
import numpy as np 

graph = GraphPlot()

def get_graph1(threshold):
    
    fig = graph.ROC_AUC(threshold)
    return fig 

def get_graph2(data, X, y ):
    
    fig = graph.FeatureImportance(data, X, y)
    return fig 

# pil_image = Image.open("auc.png")
threshold = pd.read_excel('./result.xlsx', sheet_name='因子選擇後threshold(測試)')
模型選擇結果 = pd.read_excel("./result.xlsx", sheet_name='模型選擇結果').fillna(method='ffill')
因子篩選 = pd.read_excel("./result.xlsx", sheet_name='因子篩選')

idx = np.where(因子篩選['features'].apply(lambda x: 1 if re.search('[\u4e00-\u9fff]+',x) else 0)==0)[0]

因子篩選a = 因子篩選.dropna(axis = 0)
因子篩選b = 因子篩選.iloc[idx, :].dropna(axis = 1)

模型選擇結果[模型選擇結果.columns[7:]] = 模型選擇結果[模型選擇結果.columns[7:]].round(decimals=2)
模型選擇結果[模型選擇結果.columns[:7]] = 模型選擇結果[模型選擇結果.columns[:7]].round(decimals=0)

get_graph2(data = 因子篩選a.sort_values('anova_chi', ascending=False), X = 'anova_chi', y = 'features')

ModelName = ['XGBClassifier','LogisticRegression', 'BernoulliNB','RandomForestClassifier']
models = [{'label': i, 'value':i} for i in ModelName]
ModelDict = dict(zip(ModelName, ['xgb','logis','BerNB','rf'] ))
columns = [{'name':i, 'id': i}  for i in 模型選擇結果.columns]


# --------------
# ---Dash App---
# --------------
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

    
app.layout = html.Div(children=[
    html.H2('Model Performance',style={'textAlign':'center'}),
    html.Br(),
    
    dbc.Row(
        [dbc.Col([
            html.P("Model selection", className="lead1"),
            dcc.Dropdown(
                options = models,
                placeholder="Select a model",
                id='dropdown', style ={}),
             html.P("Feature selection", className="lead2"),
             dcc.Dropdown(
                options = ['statistics', 'lasso', 'randomforest'],
                placeholder="Select a feature method.",
                id='dropdown2')]),
         dbc.Col([dcc.Graph(id='graph1',style={'display': 'inline-block'}),
                  dcc.Graph(id='graph2',style={'display': 'inline-block'})],
                #  style =  { 'margin-left':'250px'}
                 ),
                 
         dbc.Col(dash_table.DataTable(id = 'table',columns = columns),
                 style =  { #'margin-left':'250px', 
                           'margin-top':'7px', 
                           'margin-right':'600px'},
                 width = 9)
        
        ])
    ])   
    

col_convert = {'statistics':'anova_chi', 'lasso':'lasso','randomforest':'rf'}

@app.callback(Output('graph1', 'figure'),
              [Input('dropdown', 'value')])
def FirstGraph(value):
    if value == 'XGBClassifier':
        GetGraph1 = get_graph1(threshold)
    return GetGraph1  

@app.callback(Output('graph2', 'figure'),
              [Input('dropdown2', 'value')])
def SecondGraph(value):
    name = col_convert[value]
    GetGraph2 = get_graph2(data = 因子篩選a.sort_values(name, ascending=False), 
               X = name, 
               y = 'features')
    
    return GetGraph2 

@app.callback(Output('table', 'data'),
              [Input('dropdown', 'value')])
def FirstTable(value):
    df = 模型選擇結果
    data = df[df['model']==ModelDict[value]]
    return data.to_dict('records')

if __name__ == '__main__':
    app.run_server(
        port=8050,
        host='0.0.0.0',
        debug = True,
        use_reloader=False 
    )
    
