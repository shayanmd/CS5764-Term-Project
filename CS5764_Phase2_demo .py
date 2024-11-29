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
    dcc.Tabs(id='questions',children=[dcc.Tab(label = 'Numerical features analysis',value = 't1'),
                                         dcc.Tab(label = 'Categorical features analysis',value = 't2'),
                                         dcc.Tab(label = 'Statistics',value = 't3'),
                                         dcc.Tab(label = 'numerical only',value = 't4'),
                                         dcc.Tab(label = 'tab 5',value = 't5'),
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
tab4_layout = html.Div([
    html.H3('Choose the type of graph to be plotted.'),
    dcc.Dropdown(id='graph-dropdown_tab4', options=[
        {'label': 'Scatter-matrix', 'value': 'Scatter-matrix'},
        {'label': 'Scatter', 'value': 'Scatter'},
        {'label': 'Bar', 'value': 'Bar'},
        {'label': 'Line', 'value': 'Line'},
        {'label': 'Area', 'value': 'Area'},
        {'label': 'Timeline', 'value': 'Timeline'},
        {'label': 'Pie', 'value': 'Pie'},
        {'label': 'Histogram', 'value': 'Histogram'},
        {'label': 'Box', 'value': 'Box'},
        {'label': 'Violin', 'value': 'Violin'},
        {'label': 'ecdf', 'value': 'ecdf'},
        {'label': 'Density-contour', 'value': 'Density-contour'},
        {'label': 'Density-heatmap', 'value': 'Density-heatmap'},
        {'label': 'Imshow', 'value': 'Imshow'},
        {'label': 'Scatter-Map', 'value': 'Scatter-Map'},
    ], multi=False, placeholder="Select the type of graph to be plotted for the numerical feature"),
    html.H3('Plotting numerical feature distribution for states'),
    dcc.Dropdown(id='feature-dropdown-tab4', options=[
        {'label': 'price', 'value': 'price'},
        {'label': 'bed', 'value': 'bed'},
        {'label': 'bath', 'value': 'bath'},
        {'label': 'acre_lot', 'value': 'acre_lot'},
        {'label': 'house_size', 'value': 'house_size'},
    ], multi=True, placeholder="Select The Numerical Features To Be Analyzed"),
    html.Div(id='div-id-numerical-only'),
    dcc.Graph(id='numerical-only-graph')
])


@my_app.callback(
    Output('numerical-only-graph', 'figure'),
    [Input('graph-dropdown_tab4', 'value'), Input('feature-dropdown-tab4', 'value')]

)
def update_graph(graph_type, selected_features):
        if not selected_features:
            return {}
        if graph_type == 'Scatter-matrix':
            # Ensure at least 2 features are selected for a scatter-matrix
            if len(selected_features) < 2:
                return {}
            fig = px.scatter_matrix(df, dimensions=selected_features)
            return fig

        elif graph_type == 'Scatter':
            if len(selected_features) != 2:
                return {}
            fig = px.scatter(df, x=selected_features[0], y=selected_features[1])
            return fig

        elif graph_type == 'Bar':
            if len(selected_features) != 2:
                return {}
            fig = px.bar(df, x=selected_features[0], y=selected_features[1])
            return fig
        elif graph_type == 'Line':
            if len(selected_features) != 2:
                return {}
            fig = px.line(df, x=selected_features[0], y=selected_features[1])
            return fig
        elif graph_type == 'Area':
            if len(selected_features) != 2:
                return {}
            fig = px.area(df, x=selected_features[0], y=selected_features[1])
            return fig
        # elif graph_type == 'Timeline':
        #     if len(selected_features) != 2:
        #         return {}
        #     fig = px.timeline(df,x_start =  )
        elif graph_type == 'pie':
            if len(selected_features) != 2:
                return {}
            fig = px.pie(df, values='price', names='state')
            return fig
        elif graph_type == 'Violin':
            if len(selected_features) != 2:
                return {}
            fig = px.violin(df, x=selected_features[0], y=selected_features[1])
            return fig
        elif graph_type == 'ecdf':
            if len(selected_features) != 2:
                return {}
            fig = px.ecdf(df, x=selected_features[0], y=selected_features[1])
            return fig
        elif graph_type == 'Density-contour':
            if len(selected_features) != 2:
                return {}
            fig = px.density_contour(df, x=selected_features[0], y=selected_features[1])
            return fig
        elif graph_type == 'Density-heatmap':
            if len(selected_features) != 2:
                return {}
            fig = px.density_heatmap(df, x=selected_features[0], y=selected_features[1])
            return fig
        elif graph_type == 'imshow':
            if len(selected_features) != 2:
                return {}
            fig = px.imshow(df, x=selected_features[0], y=selected_features[1])
            return fig
        elif graph_type == 'Scatter-Map':
            if len(selected_features) != 2:
                return {}
            fig = px.scatter_map(df)
            return fig






    # if graph_type == 'Scatter-matrix':
    #     fig = px.scatter_matrix(df)
    # elif graph_type == 'Scatter':
    #     fig = px.histogram(df, x=feature1,y=feature2)
    # elif graph_type == 'Bar':
    #     fig = px.bar(df, x=feature1, y=feature2)
    # elif graph_type == 'Line':
    #     fig = px.line(df, x=feature1, y=feature2)
    # elif graph_type == 'Area':
    #     fig = px.area(df, x=feature1, y=feature2)
    # return fig
    # elif graph_type == 'Timeline':
    #     fig = px.histogram(df, x=feature)
    # elif graph_type == 'Pie':
    #     fig = ff.create_distplot(df[feature], ['distplot'])
    # elif graph_type == 'Histogram':
    #     fig = px.histogram(df, x=feature)
    # elif graph_type == 'Box':
    #     fig = ff.create_distplot(df[feature], ['distplot'])
    # elif graph_type == 'Violin':
    #     fig = px.histogram(df, x=feature)
    # elif graph_type == 'ecdf':
    #     fig = ff.create_distplot(df[feature], ['distplot'])
    # elif graph_type == 'Density-contour':
    #     fig = px.histogram(df, x=feature)
    # elif graph_type == 'Density-heatmap':
    #     fig = ff.create_distplot(df[feature], ['distplot'])
    # elif graph_type == 'Imshow':
    #     fig = px.histogram(df, x=feature)
    # elif graph_type == 'Scatter-Map':
    #     fig = ff.create_distplot(df[feature], ['distplot'])



#Question 5 layout

question5_layout = html.Div([
    html.H3('Bar Plot'),
    html.Button('Submit', id='submit-val', n_clicks=0),


    # ]),
    dcc.Graph(id='graph-tab5'),
])

@my_app.callback(
    Output('graph-tab5', 'figure'),
    Input('submit-val', 'n_clicks'),
    #Input('bar-dropdown-tab5', 'value'),

)
def update_output(n_clicks):
    if n_clicks <100:
        avg_price_per_state = df.groupby('state')['price'].mean()
        fig = px.bar(avg_price_per_state)
        return fig

#Question 6 layout
question6_layout = html.Div([html.Img(id='img', src='assets/image.png'),
html.H3("b1"),
dcc.Slider(
        id='bed',
        min=0,
        max=10,
        step=1,
        value=3,
        marks={i: str(i) for i in range(0, 11)}),
        html.H3("bedroom"),
dcc.Slider(
        id='bath',
        min=0,
        max=10,
        step=1,
        value=2,
        marks={i: str(i) for i in range(0, 11)}
    ),
    html.H3("bathroom"),
        dcc.Graph(id='graph-q6')
],style={'width': '30%', 'display': 'inline-block', 'vertical-align':'middle'})

@my_app.callback(
        Output('graph-q6', 'figure'),
        [Input('bed', 'value'),
         Input('bath', 'value'),
         ]

)

def update_output(bed,bath):
        print(bed,bath)

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
    [Input('questions', 'value')
     ]
)
def update_layout(tab):
    if tab == 't1':
        return tab1_layout
    elif tab == 't2':
        return tab2_layout
    elif tab == 't3':
         return tab3_layout
    elif tab == 't4':
         return tab4_layout
    elif tab == 't5':
         return question5_layout
    # elif ques == 't6':
    #     return question6_layout


my_app.run_server(debug=True,port=8061)