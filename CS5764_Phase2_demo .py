import time

import dash
from dash import dcc,html
from dash.dependencies import Input, Output
import plotly.express as px
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import statistics


df = pd.read_csv('realtor-data-50000.csv')
df_numerical = df[['price','bed','bath','acre_lot','house_size']]

external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css']

my_app = dash.Dash('my_app', external_stylesheets=external_stylesheets,suppress_callback_exceptions=True)

my_app.layout = html.Div([
    html.H2('CS 5764 Final Project: Analyzing the real estate dataset',style={'textAlign':'center'}),
    html.Br(),
    dcc.Tabs(id='hw-questions',children=[dcc.Tab(label = 'Numerical features analysis',value = 't1'),
                                         dcc.Tab(label = 'Categorical features analysis',value = 't2'),
                                         dcc.Tab(label = 'Statistics',value = 't3'),
                                         #dcc.Tab(label = 'Question 4',value = 'q4'),
                                         #dcc.Tab(label = 'Question 5',value = 'q5'),
                                         #dcc.Tab(label = 'Question 6',value = 'q6'),
                                         ]),
    html.Div(id = 'layout')

])


#Bar Plot
#df_numerical = df[['price','bed','bath','acre_lot','house_size','price_per_sqft','total_rooms']]

tab1_layout = html.Div([
    html.H3('Choose the type of graph to be plotted.'),
    dcc.Dropdown(id='graph-dropdown',options=[
              {'label':'Lineplot','value':'lineplot'},
              {'label': 'Histogram', 'value': 'histogram'},
              {'label': 'Histogram plot with KDE', 'value': 'histogram_kde'},
              {'label': 'Distplot', 'value': 'distplot'},
              {'label': 'QQ-plot', 'value': 'qqplot'},
              {'label': 'KDE', 'value': 'kde'},
    ],multi = False,placeholder="Select the type of graph to be plotted for the numerical feature"),
    html.H3('Plotting numerical feature distribution for states'),
    dcc.Dropdown(id='feature-dropdown',options=[
              {'label':'price','value':'price'},
              {'label': 'bed', 'value': 'bed'},
              {'label': 'bath', 'value': 'bath'},
              {'label': 'acre_lot', 'value': 'acre_lot'},
               {'label': 'house_size', 'value': 'house_size'},
    ],multi = False,placeholder="Select The Numerical Features To Be Analyzed"),
        html.Div(id = 'div-id'),
        dcc.Graph(id='numerical-graph')
])

@my_app.callback(
    Output('numerical-graph', 'figure'),
        [Input('graph-dropdown', 'value'), Input('feature-dropdown', 'value')]

)

def update_graph(graph_type, feature):
    print('inside!!')
    print(df.columns)
    print(graph_type, feature)
    
    if graph_type == 'lineplot':
        fig = px.line(df, x=df.index, y=feature, title=f'Lineplot for {feature}')
    elif graph_type == 'histogram':
        fig = px.histogram(df, x=feature)
    elif graph_type == 'distplot':
        fig = ff.create_distplot(df[feature], ['distplot'])
    return fig

#Question 2 layout

tab2_layout = html.Div([
    html.H3('Choose the type of graph to be plotted.'),
    dcc.Dropdown(id='cat-graph-dropdown',options=[
              {'label':'Piechart','value':'pie'},
              {'label': 'Countplot', 'value': 'countplot'},
    ],multi = False,placeholder="Select the type of graph to be plotted for the categorical feature"),
    html.H3('Select the categorical feature'),
    dcc.Dropdown(id='cat-feature-dropdown',options=[
              {'label':'status','value':'status'},
              {'label': 'city', 'value': 'city'},
              {'label': 'state', 'value': 'state'},
              {'label': 'zip_code', 'value': 'zip_code'},
    ],multi = False,placeholder="Select The Categorical Feature To Be Analyzed"),
        html.Div(id = 'cat-div-id'),
        dcc.Graph(id='categorical-graph')

])
@my_app.callback(
    Output('categorical-graph', 'figure'),
        [Input('cat-graph-dropdown', 'value'), Input('cat-feature-dropdown', 'value')]

)

def update_graph(graph_type, feature):
    print('inside categorical!!')
    print(df.columns)
    print(graph_type, feature)
    
    if graph_type == 'pie':
        new_df = pd.DataFrame([])
        new_df['feature_counts'] = df[feature].value_counts()
        new_df['labels'] = df[feature].unique()
        fig = px.pie(new_df, values='feature_counts', names='labels', title=f'Pie chart for {feature}')

    elif graph_type == 'countplot':
        new_df = df.groupby(by=[feature]).size().reset_index(name="counts")
        fig = px.bar(data_frame=new_df, x=feature, y="counts", barmode="group")
    return fig

tab3_layout = html.Div([
    dcc.Dropdown(id='barplot-dropdown3',options=[
              {'label':'state','value':'state'},
              {'label': 'zip_code', 'value': 'zip_code'},
              {'label': 'brokered_by', 'value': 'brokered_by'},
               # {'label': 'price_per_sqft', 'value': 'price_per_sqft'},
               # {'label': 'total_rooms', 'value': 'total_rooms'},
               # {'label': 'year', 'value': 'year'}

    ],multi = False,placeholder="Select The Numerical Features To Be Plotted Against State"),
        html.Div(id = 'div-id3'),
        dcc.Graph(id='barplot-graph3')

])
@my_app.callback(
    Output('barplot-graph3', 'figure'),
        [Input('barplot-dropdown3', 'value')]

)

def update_graph(feature):
    print(feature)
    fig = px.bar(df, x=feature)
    fig.update_layout(title='Countplot')
    return fig


#Question 3 layout
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

tab3_layout = html.Div([
    dcc.Checklist(
        id = 'my-checklist',
        options = [
            {'label': 'mean', 'value': 'mean'},
            {'label': 'median', 'value': 'median'},
            {'label': 'mode', 'value': 'mode'},
            {'label': 'std_dev', 'value': 'std_dev'},
            {'label': 'variance', 'value': 'variance'}]),

    html.Br(),
    html.Br(),

    dcc.Dropdown(id='numerical-dropdown',options=[
              {'label':'price','value':'price'},
              {'label': 'bed', 'value': 'bed'},
              {'label': 'bath', 'value': 'bath'},
              {'label': 'acre_lot', 'value': 'acre_lot'},
               {'label': 'house_size', 'value': 'house_size'}],multi = False,placeholder="Select the type of graph to be plotted for the numerical feature"),
    #html.Button('Submit', id='textarea-button', n_clicks=0)
    html.Br(),
    html.Br(),
    dcc.Loading(
        id = 'loading-id',
        type='circle',
        children = [dcc.Textarea(
        id='textarea',
        readOnly=True,
        #value= "f The output value is {output}",
        style={'width': '100%', 'height': 200}),
                        ])])


@my_app.callback(
                  Output('textarea', 'value'),
                 [#Input('textarea-button', 'n_clicks'),
                     Input('my-checklist', 'value'),
                  Input('numerical-dropdown', 'value'),
                  ]
)


def update_output(checklist,variable):
    output = ""
    print(variable, checklist)
    # if n_clicks < 1 or
    if checklist == None or variable == None:
         output = f" Click on a mode of operation,select a feature and click the button"
         return output
    # if n_clicks > 0:
    #     print("HI")
    time.sleep(2)
    if 'mean' in checklist:
        output += f"The mean value is {np.mean(df[variable])}.\n"
        print(output)
    if 'median' in checklist:
        output += f" The median value is {np.median(df[variable])}.\n"
    if 'mode' in checklist:
        output += f" The mode value is {statistics.mode(df[variable])}.\n"
    if 'std_dev' in checklist:
        output += f" The std_dev value is {np.std(df[variable])}.\n"
    if 'variance' in checklist:
        output += f" The variance value is {np.var(df[variable])}.\n"
    print(output)
    return output

#Question 4 layout
x_q4 = np.linspace(-2,2,1000)
question4_layout = html.Div([
    html.H1('Please enter the polynomial order'),
    dcc.Input(id = 'my-input1', type = 'number'),

    html.Div(id='output-id'),
    dcc.Graph(id='graph-id')

])
@my_app.callback(
    Output('graph-id', 'figure'),
    [Input('my-input1', 'value'),
     ]

)

def update_output(input_value):
    if input_value is not None:
        input_value = float(input_value)
    else:
        input_value = 0.0
    output = x_q4**input_value
    fig = px.line(output)
    return fig

#Question 5 layout

question5_layout = html.Div([
    html.H3('Please enter the number of sinusoidal cycles'),
    dcc.Input(id = 'my-input1', type = 'number',value = 4),
    html.H3('Please enter the mean of white noise'),
    dcc.Input(id = 'my-input2', type = 'number',value = 1),
    html.H3('Please enter the standard deviation of the white noise'),
    dcc.Input(id = 'my-input3', type = 'number',value = 1),
    html.H3('Please enter the number of samples'),
    dcc.Input(id = 'my-input4', type = 'number',value = 1000),

    dcc.Graph(id='graph1-sin'),
    dcc.Graph(id='graph2-fft'),
])

@my_app.callback(
    [Output('graph1-sin', 'figure'),
     Output('graph2-fft', 'figure')],
    [Input('my-input1', 'value'),
     Input('my-input2', 'value'),
     Input('my-input3', 'value'),
     Input('my-input4', 'value')
     ]
)
def update_output(input1, input2, input3, input4):
    input1 = float(input1) if input1 is not None else 0.0
    input2 = float(input2) if input2 is not None else 0.0
    input3 = float(input3) if input3 is not None else 0.0
    input4 = int(input4) if input4 is not None else 0

    print(input1, input2, input3, input4)
    x_q5 = np.linspace(-np.pi, np.pi, input4)
    y_q5 = np.sin(x_q5*input1) + np.random.normal(input2, input3, input4)
    from dash.exceptions import PreventUpdate
    if x_q5.size != 0:
        fig1 = px.line(x=x_q5, y=y_q5)
        fyy = np.fft.fft(y_q5)
        fig21 = np.abs(fyy)
        fig2 = px.line(x=x_q5, y=fig21)
        return fig1, fig2
    elif x_q5.size == 0:
        raise PreventUpdate

#Question 6 layout
question6_layout = html.Div([html.Img(id='img', src='assets/image.png'),
html.H3("b1"),
dcc.Slider(
        id='b1',
        min=-10,
        max=10,
        step=0.001,
        value=5,
        marks={i: str(i) for i in range(-10, 11)}),
                             html.H3("b2"),
dcc.Slider(
        id='b2',
        min=-10,
        max=10,
        step=0.001,
        value=5,
        marks={i: str(i) for i in range(-10, 11)}
    ),
                             html.H3("w1"),
dcc.Slider(
        id='w1',
        min=-10,
        max=10,
        step=0.001,
        value=5,
        marks={i: str(i) for i in range(-10, 11)}
    ),
                             html.H3("w2"),
dcc.Slider(
        id='w2',
        min=-10,
        max=10,
        step=0.001,
        value=5,
        marks={i: str(i) for i in range(-10, 11)}
    ),
                             html.H3("b3"),
dcc.Slider(
        id='b3',
        min=-10,
        max=10,
        step=0.001,
        value=5,
        marks={i: str(i) for i in range(-10, 11)}
    ),
                             html.H3("w3"),
dcc.Slider(
        id='w3',
        min=-10,
        max=10,
        step=0.001,
        value=5,
        marks={i: str(i) for i in range(-10, 11)}
    ),
                             html.H3("w4"),
dcc.Slider(
        id='w4',
        min=-10,
        max=10,
        step=0.001,
        value=5,
        marks={i: str(i) for i in range(-10, 11)}
    ),
        dcc.Graph(id='graph-q6')
],style={'width': '30%', 'display': 'inline-block', 'vertical-align':'middle'})

@my_app.callback(
        Output('graph-q6', 'figure'),
        [Input('b1', 'value'),
         Input('b2', 'value'),
         Input('w1', 'value'),
         Input('w2', 'value'),
         Input('b3', 'value'),
         Input('w3', 'value'),
         Input('w4', 'value')
         ]

)

def update_output(b1, b2, w1, w2, b3, w3, w4):
        print(b1, b2, w1, w2, b3, w3, w4)
        p = np.linspace(-5,5,1000)
        x1 = p*w1 + b1
        x2 = p*w2 + b2
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        a1 = sigmoid(x1)
        a2 = sigmoid(x2)
        a3 = w3 * a1 + w4 * a2 +b3
        fig = px.line(x =p,y =a3)
        return fig

#

@my_app.callback(
    Output('layout', 'children'),
    [Input('hw-questions', 'value')
     ]
)
def update_layout(tab):
    if tab == 't1':
        return tab1_layout
    elif tab == 't2':
        return tab2_layout
    elif tab == 't3':
         return tab3_layout
    # elif ques == 't4':
    #     return question4_layout
    # elif ques == 't5':
    #     return question5_layout
    # elif ques == 't6':
    #     return question6_layout


my_app.run_server(debug=True,port=8061)