import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


df = pd.read_csv('realtor-data.csv')
print(df.columns)
print(df.shape)
print(f"The dataset has {df.shape[0]} rows ")
print(f"The dataset has {df.shape[1]} columns")
print("First 5 rows of the dataset:\n",df.head().to_string())

#Handling Missing Values

print("Initial missing values :\n",df.isnull().sum())
df.dropna(inplace=True)
print("Missing values after cleaning:\n",df.isnull().sum())
print(df.shape)
print(f"The dataset has {df.shape[0]} rows ")
print(f"The dataset has {df.shape[1]} columns")
if df.isnull().sum().sum() == 0:
    print("The dataset has been cleaned")
print(df.shape)
print(df.head().to_string())

#Creating two new features
df["price_per_sqft"] = df["price"]/df["house_size"]
df["total_rooms"] = df["bed"]+df["bath"]

#Box Plot for outlier removal

Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5*IQR
upper_bound = Q3 + 1.5*IQR
outliers = df[(df['price'] < lower_bound) | (df['price'] > upper_bound)]
df_cleaned = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]
sns.boxplot(x = df_cleaned['price'])
plt.title("Boxplot of Prices after Outlier Removal")
plt.show()






#numerical features only
df['price'] = df['price'].astype('float64')
df['bed'] = df['bed'].astype('float64')
df['bath'] = df['bath'].astype('float64')
df['acre_lot'] = df['acre_lot'].astype('float64')
df['house_size'] = df['house_size'].astype('float64')
df['price_per_sqft'] = df['price_per_sqft'].astype('float64')
df['total_rooms'] = df['total_rooms'].astype('float64')
df_numerical = df[['price','bed','bath','acre_lot','house_size','price_per_sqft','total_rooms']]

#Normalization
def normalize(features):
    df_numerical = features
    new_df = pd.DataFrame()
    for i in df_numerical.columns:
        print("i",i)
        max_number = np.max(df_numerical[i])
        min_number = np.min(df_numerical[i])
        normalized = (df_numerical[i] - min_number) / (max_number - min_number)
        new_df[i] = normalized
    return new_df

#Standardization
def standardize(features):
    df_numerical = features
    new_df1 = pd.DataFrame()
    for i in df_numerical.columns:
        standardized = (df_numerical[i] - np.mean(df_numerical[i])) / np.std(df_numerical[i])
        new_df1[i] = standardized
    return new_df1

normalized_df =normalize(df_numerical)
df_preprocessed = standardize(normalized_df)


#PCA
df_preprocessed_complete = df_preprocessed.copy()

df["brokered_by"] = df["brokered_by"].astype('category')
df["status"] = df["status"].astype('category')
df["street"] = df["street"].astype('category')
df["city"]= df["city"].astype('category')
df['state'] = df['state'].astype('category')
df['zip_code'] = df['zip_code'].astype('category')
df["prev_sold_date"] = df["prev_sold_date"].astype('category')



df_preprocessed_complete['brokered_by'] = df['brokered_by'].cat.codes
df_preprocessed_complete['status'] = df['status'].cat.codes
df_preprocessed_complete['street'] = df['street'].cat.codes
df_preprocessed_complete['city'] = df['city'].cat.codes
df_preprocessed_complete['state'] = df['state'].cat.codes
df_preprocessed_complete['zip_code'] = df['zip_code'].cat.codes
df_preprocessed_complete['prev_sold_date'] = df['prev_sold_date'].cat.codes


X = df_preprocessed_complete.drop(columns = ['price'])
y = df_preprocessed_complete['price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5764)


pca1 = PCA()
pca1.fit(X_train,y_train)
explained_variance_ratio = pca1.explained_variance_ratio_
sum = 0
flag = 0
sum_explained_variance_ratio = []
for i in range(len(explained_variance_ratio)):
    sum += explained_variance_ratio[i]
    sum_explained_variance_ratio.append(sum)
    if sum > 0.95 and flag == 0:
        print(str(i+1), "features are needed to explain 95% of the dependent variance")
        flag += 1
print(sum_explained_variance_ratio)

n = len(sum_explained_variance_ratio)
num_features = np.arange(1,n+1)

plt.plot(num_features,sum_explained_variance_ratio)
plt.axhline(y = 0.95,ls = "-.")
#print(np.interp(0.95,sum_explained_variance_ratio,num_features))
plt.axvline(x = np.interp(0.95,sum_explained_variance_ratio,num_features),ls ="-.")
plt.xlabel("Number of Features")
plt.ylabel("Cumulative Explained Variance")
plt.grid()
plt.title("Cumulative Explained Variance VS Number of Features")
plt.show()

components = pca1.components_
print("Components",components)



#Bar Plot
#Line Plot-Numerical

plt.figure(figsize=(10,6))

df['bed'].value_counts().sort_index().plot(kind='bar')

plt.title('Distribution of Bedrooms in Real Estate Listings')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.xlim([0,10])
plt.show()

plt.figure(figsize=(12,8))
df['state'].value_counts().sort_values(ascending=False).plot(kind='bar')

plt.title('Distribution of Properties by State')
plt.xlabel('State')
plt.ylabel('Number of Properties')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
#Line Plot-Numerical

sns.lineplot(df,x = 'prev_sold_date', y = 'price')
plt.title("Line plot of price")
plt.show()


#Bar Plot
sns.barplot(y =df['state'].groupby(by= df['state']) , x = df['state'].index())
plt.title("Barplot of state")
plt.show()


#df.drop(upper_array.index)
#df.drop(index=lower_array, inplace=True)

#plt.show()

#df["price_per_sqft"] = df["price"]/df["house_size"]
#df["total_rooms"] = df["bed"]+df["bath"]
#sns.boxplot(data=df)
#plt.show()

#sns.lineplot(data=df, x='house_size', y='price')
#plt.show()

#sns.lineplot(data=df, x='acre_lot', y='price')
#plt.show()

#plt.scatter(x='house_size', y='price')
#plt.show()

#plt.scatter(x='acre_lot', y='price')
#plt.show()

#plt.hist(df['bed'])
#plt.show()
#plt.hist(df['bath'])
#plt.show()

#plt.bar(df['state'])
#plt.show()
#Bar plot

#Comment due to dash
#sns.countplot(data=df, x='brokered_by', hue='status', dodge=False)
#plt.title('Stacked Bar Plot of Brokered By vs Status')
#plt.show()


#Count Plot
#sns.countplot(data=df, x='status')
#plt.title('Count Plot of Status')
#plt.show()

#Pie chart
#status_counts = df['status'].value_counts()
#status_counts.plot.pie(autopct='%1.1f%%',figsize=(6, 6))
#plt.title('Pie Chart of Status')
#plt.ylabel('')
#plt.show()

#Line Plot
#sns.lineplot(data=df, x='house_size', y='price')
#plt.title('Line Plot of House Size vs Price')
#plt.xlim([0,100000])
#plt.ylim([0,150000000])
#plt.show()



#Pairplot
#sns.pairplot(df[['price', 'bed', 'bath', 'acre_lot', 'house_size']])
#plt.show()

#Dash

# #import dash
# from click import option
# from dash import dcc,html
# from dash.dependencies import Input, Output
# import plotly.express as px
# import numpy as np
# import pandas as pd
# from scipy import fft
#
# external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css']
#
# my_app = dash.Dash('my_app')
# my_app.layout = html.Div([
# dcc.Dropdown(id='numerical-variable-dropdown',options=[
#               {'label':'price','value':'price'},
#               {'label': 'bed', 'value': 'bed'},
#               {'label': 'bath', 'value': 'bath'},
#               {'label': 'acre_lot', 'value': 'acre_lot'},
#                {'label': 'house_size', 'value': 'house_size'},
#
#     ],multi = True,placeholder="Pick the Numerical Feature"),
# dcc.Dropdown(id='plot-type-dropdown',options=[
#               {'label':'histogram','value':'histogram'},
#               {'label': 'scatter', 'value': 'scatter'},
#               {'label': 'boxplot', 'value': 'boxplot'},
#     ],multi = False,placeholder="Pick the Type of Plot"),
#
#         html.Div(id = 'div-id'),
#         dcc.Graph(id='numerical-features-plot-graph')
#
# ])
#
# @my_app.callback(
#     Output('numerical-features-plot-graph', 'figure'),
#         [Input('numerical-variable-dropdown', 'value'),
#          Input('plot-type-dropdown', 'value'),]
# )
# def update_graph(numerical_variable,plot_type):
#         if plot_type == 'line':
#             return px.line(numerical_variable)
#         elif plot_type == 'scatter':
#             return px.scatter(numerical_variable)
#         elif plot_type == 'boxplot':
#             return px.box(numerical_variable)
#         else:
#             print("Invalid plot type")
#
#
# my_app.run_server(debug=True,port=8017)