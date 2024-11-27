import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


df = pd.read_csv('realtor-data.csv')

df.dropna(inplace=True)
df["price_per_sqft"] = df["price"]/df["house_size"]
df["total_rooms"] = df["bed"]+df["bath"]
df['year'] = df['prev_sold_date'].str.strip().str[:4]
df['year'] = df['year'].astype(int)
df = df[df['year']<= 2024]
print(max(df['year']), min(df['year']))
print(type(df['year']))
df_recent_10_years = df[df['year'] >= max(df['year']) - 10]
print(df.head().to_string())
df_numerical = df[['price','bed','bath','acre_lot','house_size','price_per_sqft','total_rooms','year']]


#Outlier Removal
for feature in df_numerical.columns:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5*IQR
    upper_bound = Q3 + 1.5*IQR
    outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
    df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
    df_numerical = df[['price', 'bed', 'bath', 'acre_lot', 'house_size', 'price_per_sqft', 'total_rooms', 'year']]




#Line Plot
avg_price_per_year = df.groupby('year')['price'].mean()
avg_price_per_sqft_year = df.groupby('year')['price_per_sqft'].mean()
fig,axes = plt.subplots(2,figsize=(10,8))
axes[0].plot(avg_price_per_year.index, avg_price_per_year.values,label = "Price")
axes[0].set_title('Average Property Price Over Time')
#plt.title('Average Property Price Over Time')
axes[0].set_xlabel('Year')
axes[0].legend()
axes[0].set_ylabel('($)')
# plt.xlabel('Year')
# plt.ylabel('($)')
# plt.grid()
# plt.legend()
plt.plot(avg_price_per_sqft_year.index, avg_price_per_sqft_year.values,label = "Price_per_sqft")
axes[1].set_title('Average Property Price per sqft Over Time')
axes[1].set_xlabel('Year')
axes[1].legend()
axes[1].set_ylabel('($)')
#plt.title('Average Property Price per sqft Over Time')
#plt.xlabel('Year')
#plt.ylabel('($)')
#plt.grid()
#plt.legend()
plt.tight_layout()
plt.show()


#Stacked Bar plot
avg_bed_per_state = df.groupby('state')['bed'].mean()
avg_bath_per_state = df.groupby('state')['bath'].mean()
print("avg bed state",avg_bed_per_state)
plt.figure(figsize=(16, 8))
plt.bar(avg_bed_per_state.index, avg_bed_per_state.values, color='r',label='Bed')
plt.bar(avg_bath_per_state.index, avg_bath_per_state.values, bottom=avg_bed_per_state.values, color='b',label='Bath')
plt.tight_layout()
plt.xlabel("State")
plt.ylabel("Bedrooms and Bathrooms")
plt.title("Average Beds and Bath Per State")
plt.xticks(rotation=45, ha='right')
plt.subplots_adjust(bottom=0.2,left=0.2,top=0.8,right=0.8)
plt.legend()
plt.show()

#Group Bar Plot
plt.figure(figsize=(20, 10))
x = np.arange(len(avg_bed_per_state.index))
width = 0.40
plt.bar(x - 0.2, avg_bed_per_state.values, width,label='Bed')
plt.bar(x + 0.2, avg_bath_per_state, width,label='Bath')
plt.tight_layout()
plt.xlabel("State")
plt.ylabel("Bedrooms and Bathrooms")
plt.title("Average Beds and Bath Per State")
plt.xticks(x,avg_bed_per_state.index,rotation=45, ha='right')
plt.subplots_adjust(bottom=0.2,left=0.2,top=0.8,right=0.8)

plt.legend()
plt.show()

#Bar Plot-Numerical
plt.figure(figsize=(10,6))
df['bed'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribution of Bedrooms in Real Estate Listings')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.xlim([0,10])
plt.show()

#Bar Plot
plt.figure(figsize=(12,8))
df['state'].value_counts().sort_values(ascending=False).plot(kind='bar')
plt.title('Distribution of Properties by State')
plt.xlabel('State')
plt.ylabel('Number of Properties')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#DO subplots of top 3 states with top cities
#Count Plot
plt.figure(figsize=(12,6))
sns.countplot(x='year', data=df_recent_10_years)

plt.title('Distribution of Properties Sold in the last 10 years')
plt.xlabel('Year')
plt.ylabel('Number of Properties')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


#Piechart

plt.figure(figsize=(12,6))
explode = [0.1,0]
plt.pie(df['status'].value_counts(),labels=df['status'].unique(),autopct='%1.1f%%',explode=explode)
plt.legend()
plt.title('Distribution of Properties by Status')
plt.show()

#Displot
print("hi")
sample_df = df.sample(n=50000)
sns.displot(data = sample_df,x = 'total_rooms',hue = 'status')
plt.title("Displot")
plt.tight_layout()
plt.xlim([0,20])
plt.show()

#Displot for
sample_df = df.sample(n=5000)
sns.displot(data = sample_df,x = 'price',kde = True)
plt.title("Displot for Acre Lot")
plt.xlim([0,5000000])
plt.tight_layout()
plt.show()

#Bivariate plot
sns.displot(data = sample_df,x = 'price',y = 'house_size')
plt.title("Bivariate plot")
plt.tight_layout()
plt.xlim([0,5000000])
plt.ylim(0,7500)
plt.show()

#Bivariate kde plot
sns.displot(data = sample_df,x = 'price',y = 'house_size',kind='kde')
plt.title("Bivariate kde plot")
plt.tight_layout()
plt.xlim([0,5000000])
plt.ylim(0,7500)
plt.show()


#1)
# sns.displot(data=sample_df, x="price", hue="state", kind="kde")
# plt.title("Bivariate plot with hue and kde")
# plt.tight_layout()
# plt.show()

#2)
#sns.displot(data=sample_df, x="flipper_length_mm", hue="species", multiple="stack")
#Pair Plot

df_pairplot = df.drop(columns=['brokered_by','street','price_per_sqft','total_rooms','year','zip_code','acre_lot'])
ten_df = df_pairplot.sample(n=10000)
sns.pairplot(ten_df,hue = 'status')
plt.suptitle("Pairplot")
plt.tight_layout()
plt.show()
#Diagonals nee x lim

#Heatmap with Cbar
plt.figure(figsize=(20,10))
import seaborn as sns
df = pd.DataFrame(df)
corr = df_numerical.corr()
sns.heatmap(corr,annot=True,cbar=True)
plt.title("Heatmap With CbAr")
plt.show()

#Histogram plot with KDE

sns.displot(data=df, x='price_per_sqft',kde=True,label = "Price per sqft")
plt.xlim([0,1250])
plt.title("Histogram with KDE")
plt.show()

print("describe\n",df['price_per_sqft'].describe())
sns.histplot(data=df, x='price_per_sqft',kde=True,label = "Price per sqft",bins=500)
plt.xlim([0,100000])
plt.title("Histogram with KDE")
plt.legend()
plt.show()
#CHeck this


#Histogram Plot with KDE
sns.histplot(df['price'], kde=True, bins=30, color='blue')
plt.title('Distribution of Price with KDE')
plt.xlabel('Price')
plt.xlim([0,20000000])
plt.ylabel('Frequency')
plt.show()
#CHeck this

#QQplot
sm.qqplot(df_numerical['house_size'], line ='45')
plt.title('QQ Plot for Price')
plt.show()
#CHeck this

#KDE plot will fill, alpha = 0.6, chose a palette, chose a linewidth

sns.kdeplot(data=df['price'],fill=True,alpha=0.6,palette='Blues',linewidth=2)

plt.title('KDE Plot for Price with Fill')
plt.xlabel('Price')
plt.ylabel('Density')
plt.xlim([0,10000000])
plt.show()
#CHeck this
sns.kdeplot(data=df['total_rooms'],fill=True,alpha=0.6,palette='Blues',linewidth=2)

print("TOTAL ROOMS\n",df['total_rooms'].describe())
print("BED ROOMS\n",df['bed'].describe())
print("bath ROOMS\n",df['bath'].describe())

plt.title('KDE Plot for total_rooms with Fill')
plt.xlabel('total_rooms')
plt.ylabel('Density')
#plt.xlim([0,10000000])
plt.show()
#Im or reg plot with scatter representation and regression line

sample_df = df.sample(n=500)

sns.lmplot(x = "house_size", y = "price",ci = None, data = sample_df)
plt.title('Price vs. House Size with Regression Line', fontsize=14)
plt.xlabel('House Size (sqft)', fontsize=12)
plt.ylabel('Price ($)', fontsize=12)
plt.tight_layout()
plt.show()
sample_df = df.sample(n=500)

sns.regplot(x='house_size', y='price', data=sample_df, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})

plt.title('Price vs. House Size with Regression Line', fontsize=14)
plt.xlabel('House Size (sqft)', fontsize=12)
plt.ylabel('Price ($)', fontsize=12)
plt.tight_layout()
plt.xlim([0,4000])
plt.ylim([0,2000000])
plt.show()
#CHeck this


# #Multivariate box plot
#Make subplots
plt.figure(figsize=(12, 6))
df_numerical.boxplot()
#sns.boxplot(x='state', y='price', data=df)
plt.title('Price Distribution by state')
plt.xlabel('City')
plt.ylabel('Price ($)')
plt.ylim([0,10])
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Area plot
df_chicago = df[df['city'] == 'Chicago']
df_chicago_yearly = df_chicago.groupby('year')['price'].mean()
df_chicago_yearly.plot(kind='area', figsize=(12, 6), color='skyblue', alpha=0.6)

plt.title('Total Price Over Year in Chicago')
plt.xlabel('Year')
plt.ylabel('Total Price ($)')
plt.tight_layout()

plt.show()

# Violin plot
df_virginia = df[(df['state'] == 'Virginia') & ((df['city'] == 'Falls Church') | (df['city'] == 'Alexandria') | (df['city'] == 'Fairfax'))]
print(df_virginia['state'].unique())
print("va",df_virginia.shape)
plt.figure(figsize=(12,6))

sns.violinplot(data = df_virginia,x = 'state',y = 'price',hue = 'city')
plt.title('Violin Plot')
plt.xlabel('City')
plt.ylabel('Price ($)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# Joint plot with KDE and scatter representation
sns.jointplot(x='house_size', y='price', data=df, kind='scatter', color='blue')
plt.suptitle('Price vs. House Size with KDE and Scatter')
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()

# Rug plot
plt.figure(figsize=(10, 2))
sns.rugplot(data = df,x = 'price', color='blue')

plt.title('Rug Plot of Price', fontsize=16)
plt.xlabel('Price ($)', fontsize=12)
plt.tight_layout()

plt.show()

#check this
# 3D plot and contour plot

# Cluster map

# Hexbin

# Strip plot
# Swarm plot




# df_numerical.boxplot(column=df['price'])
# df_numerical.boxplot(column=df['bed'])
# #sns.boxplot(data=df_numerical, orient="h", palette="Set2")
# plt.title("Box Plot for df_numerical")
# plt.tight_layout()
# plt.show()

#Area Plot
ax = df_numerical.plot.area(y='total_rooms',rot=0)
plt.title('Area plot of total Rooms')
plt.xlabel('Area')
plt.ylabel('Total Rooms')
plt.show()

#


