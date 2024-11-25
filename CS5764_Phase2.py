import dash
from dash import dcc,html
from dash.dependencies import Input, Output
import plotly.express as px
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import matplotlib.pyplot as plt


external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css']

my_app = dash.Dash('my_app', external_stylesheets=external_stylesheets,suppress_callback_exceptions=True)

my_app.layout = html.Div([
    html.H1('Homework 6',style={'textAlign':'center'}),
    html.Br(),
    dcc.Tabs(id='hw-questions',children=[dcc.Tab(label = 'Question 1',value = 'q1'),
                                         #dcc.Tab(label = 'Question 2',value = 'q2'),
                                         #dcc.Tab(label = 'Question 3',value = 'q3'),
                                         #dcc.Tab(label = 'Question 4',value = 'q4'),
                                         #dcc.Tab(label = 'Question 5',value = 'q5'),
                                         #dcc.Tab(label = 'Question 6',value = 'q6'),
                                         ]),
    html.Div(id = 'layout')

])

#Question 1 Layout
df = pd.read_csv('realtor-data.csv')
df_cleaned = df.drop(df.isnull().sum())
df = df_cleaned
df = df[:1000]
df['status'] = df['status'].astype(pd.StringDtype())
#Bar Plot
#df_numerical = df[['price','bed','bath','acre_lot','house_size','price_per_sqft','total_rooms']]

question1_layout = html.Div([
    html.H3('Plotting feature distribution for states'),
    dcc.Dropdown(id='barplot-dropdown',options=[
              {'label':'price','value':'price'},
              {'label': 'bed', 'value': 'bed'},
              {'label': 'bath', 'value': 'bath'},
              {'label': 'acre_lot', 'value': 'acre_lot'},
               {'label': 'house_size', 'value': 'house_size'},
               # {'label': 'price_per_sqft', 'value': 'price_per_sqft'},
               # {'label': 'total_rooms', 'value': 'total_rooms'},
               # {'label': 'year', 'value': 'year'}

    ],multi = False,placeholder="Select The Numerical Features To Be Plotted Against State"),
        html.Div(id = 'div-id'),
        dcc.Graph(id='barplot-graph')

])

@my_app.callback(
    Output('barplot-graph', 'figure'),
        [Input('barplot-dropdown', 'value')]

)

def update_graph(feature):
    print(df.columns)
    print(feature)
    print(df.dtypes)
    print(df.shape)
    #fig = px.line(x = [0,10],y = [23,10])
    #fig = df[feature].value_counts().sort_index().plot(kind='bar')
    #fig = px.bar(df,x= 'state',y = 'price')
    plt.pie(df['status'].value_counts(), labels=df['status'].unique(), autopct='%1.1f%%')

    labels = [str(label) for label in plt.gca().get_xticklabels()]
    values = plt.gca().get_yticks()
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    # print("feature",feature)
    # new_df = df.groupby('state').agg({feature: ['mean', 'var']}).reset_index()
    # print("newdf",new_df)
    #
    # fig = make_subplots(rows=2, cols=1)
    # fig.add_trace(px.line(new_df,x='state', y='mean'), row=1, col=1)
    # fig.add_trace(px.line(new_df,x='state', y='var'), row=2, col=1)
    #fig = df[feature].value_counts().sort_values(ascending=False).plot(kind='bar')
    return fig

#Question 2 layout
x_q2 = np.linspace(-2,2,1000)
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

question2_layout = html.Div([
    html.H1("Calculating ax^2+bx+c"),
    html.H4('Provide input for slider 1/a',style={'textAlign':'center'}),
dcc.Slider(
        id='slider-1',
        min=-10,
        max=10,
        step=0.5,
        value=5,
        marks={i: str(i) for i in range(-10, 11)}),
    html.H4('Provide input for slider 2/b',style={'textAlign':'center'}),
dcc.Slider(
        id='slider-2',
        min=-10,
        max=10,
        step=0.5,
        value=5,
        marks={i: str(i) for i in range(-10, 11)}
    ),
    html.H4('Provide input for slider 3/c',style={'textAlign':'center'}),
dcc.Slider(
        id='slider-3',
        min=-10,
        max=10,
        step=0.5,
        value=5,
        marks={i: str(i) for i in range(-10, 11)}
    ),
    html.Div(id='slider-output'),
    dcc.Graph(id='Output-q2')

])
@my_app.callback(
    Output('Output-q2', 'figure'),
    [Input('slider-1', 'value'),
     Input('slider-2', 'value'),
     Input('slider-3', 'value'),
     ]

)

def update_slider(slider1, slider2, slider3):
    output = slider1*(x_q2**2) + slider2*x_q2 + slider3
    fig = px.line(output)
    return fig

#Question 3 layout
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

question3_layout = html.Div([
    html.H1("Calculator",style={'textAlign':'center','color':'black'}),
    html.H1("Please enter the first number"),
    html.H1('Input'),
    dcc.Input(id = 'my-input1', type = 'number'),
    html.H1('Please enter the second number'),
    html.H1('Input'),
    dcc.Input(id='my-input2', type='number'),
    html.Br(),
    html.H1('Choose operation'),
    dcc.Dropdown(
        id='my-dropdown',
        options=[
                {'label':'+','value':'+'},
              {'label': '-', 'value': '-'},
              {'label': '*', 'value': '*'},
              {'label': '/', 'value': '/'},
               {'label': 'log', 'value': 'log'},
                {'label': 'root', 'value': 'root'},
        ]),
    html.Div(id='output')
])



@my_app.callback(Output('output', 'children'),
                 [Input('my-input1', 'value'),
                  Input('my-input2', 'value'),
                  Input('my-dropdown', 'value')]
)
def update_output(input1, input2, dropdown):
    print(input1, input2, dropdown)
    if input1 is not None:
        input1 = float(input1)
    else:
        input1 = 0.0
    if input2 is not None:
        input2 = float(input2)
    else:
        input2 = 0.0

    if dropdown == '+':
        output = input1 + input2
        return f"The output value is {output}"

    if dropdown == '-':
        output = input1 - input2
        return f"The output value is {output}"

    if dropdown == '*':
        output = input1 * input2
        return f"The output value is {output}"

    if dropdown == '/':
        if input2 == 0:
            return "Cannot divide by zero"
        else:
            output = input1 / input2
            return f"The output value is {output}"
    if dropdown == 'log':
        if isinstance(input1, (int, float)) and input1 > 0 and input2 > 1:
            output = np.log(input1)/ np.log(input2)
            return f"The output value is {output}"
        else:
            return  "a should be a positive real number and  b should be greater than one"
    if dropdown == 'root':
        if input1 == 0 or input2 <= 0:
            return "Error: a cannot be zero, and b must be a positive integer."
        if  input1 < 0 and input2 % 2 == 0:
            return "Error: b must be odd for negative values of a."
        output = input1 ** (1 / input2)
        return f"The output value is {output}"

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
def update_layout(ques):
    if ques == 'q1':
        return question1_layout
    # elif ques == 'q2':
    #     return question2_layout
    # elif ques == 'q3':
    #     return question3_layout
    # elif ques == 'q4':
    #     return question4_layout
    # elif ques == 'q5':
    #     return question5_layout
    # elif ques == 'q6':
    #     return question6_layout


my_app.run_server(debug=True,port=8069)