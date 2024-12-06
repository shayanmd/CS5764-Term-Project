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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import shapiro
from scipy.stats import kstest
from scipy.stats import normaltest


import seaborn as sns
from sklearn.model_selection import train_test_split
import statistics


# Read the data
df = pd.read_csv("realtor-data-50K-part2.csv")

df['year'] = df['prev_sold_date'].str.strip().str[:4]
df['year'] = df['year'].astype(int)
df['status'] = df['status'].astype('category')

df = df.sort_values(by='prev_sold_date')
df_numerical = df[['price','bed','bath','acre_lot','house_size']]

avg_price_per_year = pd.DataFrame(df.groupby('year')['price'].mean())
print(df.columns)

# Outlier detection and removal
lower_bounds, upper_bounds = {}, {}
original_df = df.copy()
df_copy = df.copy()

for feature in df_numerical.columns:
    q1 = df_copy[feature].quantile(0.25)
    q3 = df_copy[feature].quantile(0.75)

    iqr = q3-q1
    lb = q1-1.5*iqr
    ub = q3+1.5*iqr

    lower_bounds[feature] = lb
    upper_bounds[feature] = ub

    df_cleaned = df_copy[(df_copy[feature] >= lb) & (df_copy[feature] <= ub)]
    df_copy = df_cleaned.copy()

df = df_cleaned
df_numerical = df[['price','bed','bath','acre_lot','house_size']] # updating after cleaning


# Normality test functions
def ks_test(t, feature):
    mean = np.mean(t)
    std = np.std(t)
    dist = np.random.normal(mean, std, len(t))
    stats, p = kstest(t, dist)

    output = ""
    output += '='*50
    output += "\n"
    output += f'K-S test for {feature} feature: statistics= {stats:.2f} p-value = {p:.2f} \n' 
    alpha = 0.015
    if p > alpha :
        output += f'K-S test: {feature} feature is Normal \n'
    else:
        output += f'K-S test : {feature} feature is Not Normal \n'
        output += '=' * 50 
        output += '\n'
    return output

def shapiro_test(x, feature):
    stats, p = shapiro(x)

    output = ""
    output += '=' * 50
    output += '\n'
    output += f'Shapiro test for {feature} feature : statistics = {stats:.2f} p-value of={p:.2f}\n'
    alpha = 0.01
    if p > alpha :
        output += f'Shapiro test: {feature} feature is Normal\n'
    else:
        output += f'Shapiro test: {feature} feature is NOT Normal\n'
        output += '=' * 50
        output += '\n'
    return output


def da_k_squared_test(x, feature):
    stats, p = normaltest(x)

    output = ""
    output += '='*50
    output += '\n'
    output += f'da_k_squared test for {feature} feature: statistics= {stats:.2f} p-value ={p:.2f}\n' 
    alpha = 0.01
    if p > alpha :
       output += f'da_k_squaredtest: {feature} feature is Normal\n'
    else:
        output += f'da_k_squared test : {feature} feature is Not Normal\n'
        output += '=' * 50
        output += '\n'
    return output

# Data transformation

def normalized(df):
    new_df = pd.DataFrame()
    for col in df.columns:
        max_value = np.max(df[col])
        min_value = np.min(df[col])
        normalized_col = (df[col]-min_value)/(max_value-min_value)
        new_df[col] = normalized_col
    return new_df

rename_col_names = {0:'price', 1:'bed', 2:'bath', 3:'acre_lot', 4:'house_size'}

scaler = StandardScaler()
standardized_np = scaler.fit_transform(df_numerical)
standardized_df = pd.DataFrame(standardized_np)
standardized_df = standardized_df.reset_index()
standardized_df.rename(columns=rename_col_names, inplace=True)

normalized_df = normalized(standardized_df)
normalized_df = normalized_df.reset_index()

print(avg_price_per_year.head())

# Correlation
correlation_matrix = df_numerical.corr()

external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css']

my_app = dash.Dash('my_app', external_stylesheets=external_stylesheets,suppress_callback_exceptions=True)

my_app.layout = html.Div([
    html.H2('CS 5764 Final Project: Analyzing the real estate dataset',style={'textAlign':'center'}),
    html.Br(),
    dcc.Tabs(id='questions',children=[ 
                                        dcc.Tab(label = 'Dataset',value = 'dataset_tab'),
                                        dcc.Tab(label = 'Data Loading Process',value = 'dataset_loading_tab'),
                                        dcc.Tab(label = 'Data Cleaning Techniques',value = 'dataset_cleaning_tab'),
                                        dcc.Tab(label = 'Outlier Detection and Removal',value = 'outlier_removal_tab'),
                                        dcc.Tab(label = 'Correlation Between Features',value = 'correlation_tab'),
                                        dcc.Tab(label = 'Dimensionality Reduction Techniques',value = 'dimred_tab'),
                                        dcc.Tab(label = 'Normality Tests',value = 'normality_test_tab'),
                                        dcc.Tab(label = 'Data Transformation Techniques',value = 'data_transformation_tab'),
                                        dcc.Tab(label = 'Price analysis',value = 'price_analysis_tab'),
                                        dcc.Tab(label = 'Categorical features analysis',value = 'categorical_features_tab'),
                                        dcc.Tab(label = 'Statistics',value = 'statistics_tab'),
                                        dcc.Tab(label = 'numerical only',value = 'numeric_only_tab'),
                                        dcc.Tab(label = 'tab 5',value = 'tab_extra'),
                                         ]),
    html.Div(id = 'layout')

])


# dataset explanation
dataset_layout = html.Div([
    html.H1('Dataset'),
])

# data loading
data_loading_layout = html.Div([
    html.H1('Dataset'),
])

# data cleaning
data_cleaning_layout = html.Div([
    html.H1('Dataset'),
])

# outlier detection
outlier_layout = html.Div([
    html.H2("See the effects of outlier removal on the data"),
    html.H6("Select the numeric feature"),
    dcc.Dropdown(id='outlier-feature-dropdown',options=[
              {'label': 'price','value':'price'},
              {'label': 'bed', 'value': 'bed'},
              {'label': 'bath', 'value': 'bath'},
              {'label': 'acre_lot', 'value': 'acre_lot'},
              {'label': 'house_size', 'value': 'house_size'},
    ],multi = False,placeholder="Select the numeric feature"),
    html.Br(),
    html.Div([dcc.Graph(id='outlier-graph-1')], style={'width': '48%', 'display': 'inline-block'}),
    html.Div([dcc.Graph(id='outlier-graph-2')], style={'width': '48%', 'display': 'inline-block'}),
    html.Br(),
    dcc.Loading(
        id = 'outlier-loading-id',
        type='circle',
        children = [
            dcc.Textarea(
                id='outlier-textarea',
                readOnly=True,
                style={'width': '100%', 'height': 200}),
                ]
        )
])

@my_app.callback(
    [Output('outlier-graph-1', 'figure'), Output('outlier-graph-2', 'figure'), Output('outlier-textarea', 'value')],
    Input('outlier-feature-dropdown', 'value'),
)

def return_boxplot(feature):
    print('feature chosen', feature)

    if feature is not None:
        fig1 = px.box(original_df, y=feature, title=f"Boxplot for {feature} prior to outlier removal")
        fig2 = px.box(df_cleaned, y=feature,  title=f"Boxplot for {feature} after outlier removal")
        output = "In order to remove outliers, the following lower and upper bounds were selected using IQR:\n"
        output += f"Lower bound for {feature}: {lower_bounds[feature]}\n"
        output += f"Upper bound for {feature}: {upper_bounds[feature]}\n"

        return fig1, fig2, output

# dimensionality reduction
dimred_layout = html.Div([
    html.H1('Principal Component Analysis'),
    html.H6('Reducing our numerical features to fewer components in order to visualize them. Please choose the type of visualization:'),
    dcc.RadioItems(
        id='radio-options-for-visualization',
        options=[
            {'label': '2D visualization', 'value': 'option-2d'},
            {'label': '3D visualization', 'value': 'option-3d'}
        ],
        value='option-2d'
    ),
    html.Div(id='dynamic-image', style={'width': '300px', 'height': 'auto'})

])

@my_app.callback(
    Output('dynamic-image', 'children'),
    Input('radio-options-for-visualization', 'value')
)

def return_image(selected_option):
    if selected_option == 'option-2d':
        return html.Img(id='2d-image', src=f'assets/2DPCA.jpeg')
    elif selected_option == 'option-3d':
        return html.Img(id='3d-image', src=f'assets/3DPCA.jpeg')

# normality tests
normality_layout = html.Div([
    html.H2("Normality Tests"),
    html.Br(),
    html.H5("Select the normality test"),
    dcc.Dropdown(id='normality-test-dropdown',options=[
              {'label': 'Shapiro-Wilk test', 'value': 'shapiro_test'},
              {'label':'Kolmogorov-Smirnov test','value':'ks_test'},
              {'label':'D\'Agostino\'s K^2 test', 'value':'da_k2_test'},
    ],multi = False,placeholder="Select the normality test"),
    html.Br(),
    html.H5("Select the numeric feature"),
    dcc.Dropdown(id='normality-feature-dropdown',options=[
              {'label': 'price','value':'price'},
              {'label': 'bed', 'value': 'bed'},
              {'label': 'bath', 'value': 'bath'},
              {'label': 'acre_lot', 'value': 'acre_lot'},
              {'label': 'house_size', 'value': 'house_size'},
    ],multi = False,placeholder="Select the numeric feature"),

    dcc.Loading(
        id = 'normality-loading-id',
        type='circle',
        children = [
            dcc.Textarea(
                id='normality-textarea',
                readOnly=True,
                style={'width': '100%', 'height': 200}),
                ]
        )
])

@my_app.callback(
    Output('normality-textarea', 'value'),
    [
    Input('normality-test-dropdown', 'value'),  
    Input('normality-feature-dropdown', 'value'),
    ]
)


def update_output(test_type, feature):
    print(test_type, feature)
    output = ""
    if test_type == None or feature == None:
         output = f" Please select a normality test type and a feature"
         return output
    time.sleep(2)

    if test_type == "shapiro_test":
        output = shapiro_test(df[feature], feature)
    elif test_type == "ks_test":
        output = ks_test(df[feature], feature)
    elif test_type == "da_k2_test":
        output = da_k_squared_test(df[feature], feature)
    print('look at output', output)
    return output

# data transformation

data_transformation_layout = html.Div([
    
    html.H4("Showing the normalized and standardized versions of the Price feature"),


    html.Div([
        dcc.Graph(id='data-transformation-graph-1', figure={
            'data': [
                go.Scatter(x=df['prev_sold_date'][:200], y=df['price'][:200].values, mode='lines', name='Original Feature'),
            ],
            'layout': go.Layout(
                title='Original Feature',
                xaxis=dict(title='Year'),
                yaxis=dict(title='Price ($)'),
            )
        })], style={'width': '48%', 'display': 'inline-block'}),

    html.Div([
        dcc.Graph(id='data-transformation-graph-2', figure={
            'data': [
                go.Scatter(x=df['prev_sold_date'][:200], y=standardized_df['price'][:200].values, mode='lines', name='Standardized Frature', line=dict(color='orange')),
                go.Scatter(x=df['prev_sold_date'][:200], y=normalized_df['price'][:200].values, mode='lines', name='Normalized and Standardized Feature', line=dict(color='green')),
            ],
            'layout': go.Layout(
                title='Standardized and Normalized Price feature',
                xaxis=dict(title='Year'),
                yaxis=dict(title='Price ($)'),
                legend=dict(
                    x=1,            
                    y=1,           
                    xanchor='right',
                    yanchor='top'  
                )
            )
        })], style={'width': '48%', 'display': 'inline-block'})
])

# numerical feature distribution
numerical_feature_distribution_layout = html.Div([
    html.H1('Showing the feature distribution for numerical features'),
    html.Br(),
    html.H5("Select the type of graph to be plotted"),
    dcc.Dropdown(id='numerical-graph-type',options=[
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
    ],multi = False,placeholder="Select the type of graph to be plotted"),
    html.H5("Select the numeric feature"),
    dcc.Dropdown(id='numerical-feature-distribution-dropdown',options=[
              {'label': 'price','value':'price'},
              {'label': 'bed', 'value': 'bed'},
              {'label': 'bath', 'value': 'bath'},
              {'label': 'acre_lot', 'value': 'acre_lot'},
              {'label': 'house_size', 'value': 'house_size'},
    ],multi = False,placeholder="Select the numeric feature"),
    dcc.Graph(id='numerical-feature-graph')
])

# feature relationships
correlation_layout = html.Div([
    html.H1("Correlation Matrix Visualization"),
    dcc.Graph(figure=px.imshow(
        correlation_matrix,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Correlation Heatmap"
    ))
])

# price analysis
price_analysis_layout = html.Div([
    html.H3('Choose the type of graph to be plotted.'),
    dcc.Dropdown(id='graph-dropdown',options=[
              {'label':'Lineplot showing trends in average house prices','value':'lineplot'},
              {'label': 'Histogram displaying distribution of price values', 'value': 'histogram'},
              {'label':'Bar chart showing house price changes across the years', 'value':'bar'},
              {'label':'Grouped bar chart showing price and status for states', 'value':'grouped_bar'},
              {'label':'Stacked bar chart showing price and status for states', 'value':'stacked_bar'},
    ],multi = False,placeholder="Select the type of graph to be plotted for the Price feature"),
    dcc.Graph(id='numerical-graph')
])

@my_app.callback(
    Output('numerical-graph', 'figure'),
        [Input('graph-dropdown', 'value')]

)

def update_graph(graph_type):
    print('inside!!')
    
    if graph_type == 'lineplot':
        fig = px.line(avg_price_per_year, x=avg_price_per_year.index, y='price', title=f'Lineplot showing trends in house prices', markers=True)
    elif graph_type == 'histogram':
        fig = px.histogram(df, x='price', title='Histogram displaying distribution of price values', color_discrete_sequence=['#636EFA'])
        fig.update_layout(
            plot_bgcolor='white',  
            paper_bgcolor='lightgray',  
            title_font=dict(size=20, color='black'),
            xaxis=dict(title='Price', color='black'),
            yaxis=dict(title='Frequency', color='black')
        )
        # fig.update_layout(
        #     plot_bgcolor='white',  # Change plot background
        #     paper_bgcolor='white',  # Change entire figure background
        # )

    elif graph_type == 'bar':
        fig = px.bar(df, x='year', y='price', title='Bar chart showing house price changes across the years')
        fig.update_layout(
            plot_bgcolor='white',  
            paper_bgcolor='lightgray',  
            title_font=dict(size=20, color='black'),
            xaxis=dict(title='Price', color='black'),
            yaxis=dict(title='Frequency', color='black')
        )
    elif graph_type == 'stacked_bar':
        fig = px.bar(df, x='state', y='price', color = 'variable',barmode='stack', title='Stacked bar chart showing price and status for states')  # Example palette)
        fig.update_layout(
            plot_bgcolor='white',  
            paper_bgcolor='lightgray',  
            title_font=dict(size=20, color='black'),
            xaxis=dict(title='Price', color='black'),
            yaxis=dict(title='Frequency', color='black')
        )
    elif graph_type == 'grouped_bar':
        fig = px.bar(df, x='state', y='price',color = 'variable' , barmode='group', height=400, title='Grouped bar chart showing price and status for states' )
        fig.update_layout(
            plot_bgcolor='white',  
            paper_bgcolor='lightgray',  
            title_font=dict(size=20, color='black'),
            xaxis=dict(title='Price', color='black'),
            yaxis=dict(title='Frequency', color='black')
        )
    return fig


# Categorical feature analysis

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


# Statistics 

statistics_layout = html.Div([
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


def update_output(checklist, variable):
    output = ""
    if checklist == None or variable == None:
         output = f" Click on a mode of operation,select a feature and click the button"
         return output

    time.sleep(2)
    if 'mean' in checklist:
        output += f"The mean value is {np.mean(df[variable])}.\n"
    if 'median' in checklist:
        output += f" The median value is {np.median(df[variable])}.\n"
    if 'mode' in checklist:
        output += f" The mode value is {statistics.mode(df[variable])}.\n"
    if 'std_dev' in checklist:
        output += f" The std_dev value is {np.std(df[variable])}.\n"
    if 'variance' in checklist:
        output += f" The variance value is {np.var(df[variable])}.\n"
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

# question6_layout = html.Div([html.Img(id='img', src='assets/image.png'),
# html.H3("b1"),
# dcc.Slider(
#         id='bed',
#         min=0,
#         max=10,
#         step=1,
#         value=3,
#         marks={i: str(i) for i in range(0, 11)}),
#         html.H3("bedroom"),
# dcc.Slider(
#         id='bath',
#         min=0,
#         max=10,
#         step=1,
#         value=2,
#         marks={i: str(i) for i in range(0, 11)}
#     ),
#     html.H3("bathroom"),
#         dcc.Graph(id='graph-q6')
# ],style={'width': '30%', 'display': 'inline-block', 'vertical-align':'middle'})
#
# @my_app.callback(
#         Output('graph-q6', 'figure'),
#         [Input('bed', 'value'),
#          Input('bath', 'value'),
#          ]
#
# )
#
# def update_output(bed,bath):
#         print(bed,bath)
#
#         p = np.linspace(-5,5,1000)
#         x1 = p*w1 + b1
#         x2 = p*w2 + b2
#         sigmoid = lambda x: 1 / (1 + np.exp(-x))
#         a1 = sigmoid(x1)
#         a2 = sigmoid(x2)
#         a3 = w3 * a1 + w4 * a2 +b3
#         fig = px.line(x =p,y =a3)
#         return fig

#

@my_app.callback(
    Output('layout', 'children'),
    [Input('questions', 'value')
     ]
)
def update_layout(tab):
    if tab == 'outlier_removal_tab':
        return outlier_layout
    elif tab == 'dimred_tab':
        return dimred_layout
    elif tab == 'correlation_tab':
        return correlation_layout
    elif tab == 'normality_test_tab':
        return normality_layout
    elif tab == 'data_transformation_tab':
        return data_transformation_layout
    elif tab == 'price_analysis_tab':
        return price_analysis_layout
    elif tab == 'statistics_tab':
        return statistics_layout
    else:
        return tab2_layout


my_app.run_server(debug=True,port=8083)