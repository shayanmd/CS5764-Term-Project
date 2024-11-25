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
print(max(df['year']), min(df['year']))
print(type(df['year']))
df_recent_10_years = df[df['year'] >= max(df['year']) - 10]
print(df.head().to_string())
df_numerical = df[['price','bed','bath','acre_lot','house_size','price_per_sqft','total_rooms','year']]

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

#Count plot
#df['city'].value_counts().sort_values(ascending=False).plot(kind='count')


# df['year'].value_counts().sort_values(ascending=False).plot(kind='count')
#
# sns.countplot(df['year'].value_counts())
# plt.title('Distribution of Properties by Year')
# plt.xlabel('Year')
# plt.ylabel('Number of Properties')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.show()

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
plt.pie(df['status'].value_counts(),labels=df['status'].unique(),autopct='%1.1f%%')
plt.legend()
plt.title('Distribution of Properties by Status')
plt.show()

#Displot
print("hi")
sample_df = df.sample(n=5000)
# sns.displot(sample_df, x='house_size', y='price')
# plt.title('Displot of Properties by Acre Lot')
# seventy_five_percentile_y = df['price'].quantile(0.75)
# seventy_five_percentile_x = df['house_size'].quantile(0.75)
# plt.xlim([0,seventy_five_percentile_x])
# plt.ylim([0,seventy_five_percentile_y])
# plt.show()

sns.displot(data = sample_df,x = 'total_rooms',hue = 'status')
plt.title("Displot")
plt.tight_layout()
plt.xlim([0,20])
plt.show()


#Heatmap Correlation
plt.figure(figsize=(20,10))
import seaborn as sns
df = pd.DataFrame(df)
corr = df_numerical.corr()
sns.heatmap(corr,annot=True)
plt.title("Heatmap Correlation")
plt.show()

#Histogram plot with KDE

sns.displot(data=df, x='price_per_sqft',kind='kde')
plt.xlim([0,10000])
plt.title("Histogram with KDE")
plt.show()

#Pairplot
#sns.pairplot(df[['price', 'house_size']])
#sns.pairplot(df,hue='status')
#ns.pairplot(df_numerical)
#lt.title("Pairplot")
#plt.show()

#Histogram Plot with KDE
sns.histplot(df['price'], kde=True, bins=30, color='blue')
plt.title('Distribution of Price with KDE')
plt.xlabel('Price')
plt.xlim([0,20000000])
plt.ylabel('Frequency')
plt.show()

#QQplot
sm.qqplot(df_numerical, line ='45')
plt.title('QQ Plot for Price')
plt.show()

#kde plot

sns.kdeplot(data=df['price'],fill=True,alpha=0.6,palette='Blues',linewidth=2)

plt.title('KDE Plot for Price with Fill')
plt.xlabel('Price')
plt.ylabel('Density')
plt.xlim([0,10000000])
plt.show()

#lmplot(Scatter representation)
#slowing code down
# sns.lmplot(data = df,x = 'house_size',y='price')
# plt.title("Question 20")
# plt.tight_layout()
# plt.legend()
# plt.show()

#Multivariate box plot
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


