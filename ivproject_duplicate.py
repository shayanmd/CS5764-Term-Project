import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.ndimage import label

df = pd.read_csv('realtor-data.csv')

df.dropna(inplace=True)
df["price_per_sqft"] = df["price"]/df["house_size"]
df["total_rooms"] = df["bed"]+df["bath"]
df['year'] = df['prev_sold_date'].str.strip().str[:4]
df['year'] = df['year'].astype(int)
df = df[df['year']<= 2023]
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


print(df.head().to_string())
df1 = df[:30000]
df2 = df[-30000:]
df_mixed = pd.concat([df1,df2])
df_mixed_new = df.sample(n=60000)
df_mixed_new.to_csv("new-realtor-data-50K-part2.csv", index=False)

font_title = {'family' : 'serif','color' : 'blue','size' : 20}
font_label = {'family' : 'serif','color' : 'darkred','size' : 15}

#Line Plot
avg_price_per_year = df.groupby('year')['price'].mean()
avg_price_per_sqft_year = df.groupby('year')['price_per_sqft'].mean()
fig,axes = plt.subplots(2,figsize=(10,8))
axes[0].plot(avg_price_per_year.index, avg_price_per_year.values,label = "Price")
axes[0].set_title('Average Property Price Over Time',fontdict=font_title)
axes[0].set_xlabel('Year',fontdict=font_label)
axes[0].legend()
axes[0].set_ylabel('($)',fontdict=font_label)
axes[0].grid(True)

plt.plot(avg_price_per_sqft_year.index, avg_price_per_sqft_year.values,label = "Price_per_sqft")
axes[1].set_title('Average Property Price per sqft Over Time',fontdict=font_title)
axes[1].set_xlabel('Year',fontdict=font_label)
axes[1].legend()
axes[1].set_ylabel('($)',fontdict=font_label)
axes[1].grid(True)
plt.tight_layout()
plt.show()

#Stacked Bar plot
avg_bed_per_state = df.groupby('state')['bed'].mean()
avg_bath_per_state = df.groupby('state')['bath'].mean()
print("avg bed state",avg_bed_per_state)
plt.figure(figsize=(16, 8))
plt.bar(avg_bed_per_state.index, avg_bed_per_state.values, color='r',label='Bed')
plt.bar(avg_bath_per_state.index,avg_bath_per_state.values, bottom=np.round(avg_bed_per_state.values,2), color='b',label='Bath')
plt.tight_layout()
plt.xlabel("State",fontdict=font_label)
plt.ylabel("Bedrooms and Bathrooms",fontdict=font_label)
plt.title("Average Beds and Bath Per State",fontdict=font_title)
plt.xticks(rotation=45, ha='right')
plt.subplots_adjust(bottom=0.2,left=0.2,top=0.8,right=0.8)
plt.legend()
plt.grid()
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
plt.grid()
plt.legend()
plt.show()

#Bar Plot-Numerical
plt.figure(figsize=(10,6))
df['bed'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribution of Bedrooms in Real Estate Listings',fontdict=font_title)
plt.xlabel('Number of Bedrooms',fontdict=font_label)
plt.ylabel('Count',fontdict=font_label)
plt.xticks(rotation=0)
plt.xlim([0,10])
plt.grid()
plt.legend()
plt.show()

#Bar Plot
plt.figure(figsize=(12,8))
df['state'].value_counts().sort_values(ascending=False).plot(kind='bar')
plt.title('Distribution of Properties by State',fontdict=font_title)
plt.xlabel('State',fontdict=font_label)
plt.ylabel('Number of Properties',fontdict=font_label)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.grid()
plt.legend()
plt.show()

#Histogram
plt.figure(figsize=(10,6))
plt.hist(df['price'],label="Count")
plt.title('Distribution of Price',fontdict=font_title)
plt.xlabel('price',fontdict=font_label)
plt.ylabel('count',fontdict=font_label)
plt.grid()
plt.legend()
plt.show()

#DO subplots of top 3 states with top cities

#Count Plot
plt.figure(figsize=(12,6))
sns.countplot(x='year', data=df_recent_10_years,label="Number of Properties")
plt.title('Distribution of Properties Sold in the last 10 years',fontdict=font_title)
plt.xlabel('Year',fontdict=font_label)
plt.ylabel('Number of Properties',fontdict=font_label)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.grid()
plt.legend()
plt.show()


#Piechart

plt.figure(figsize=(12,6))
explode = [0.1,0]
plt.pie(df['status'].value_counts(),labels=df['status'].unique(),autopct='%1.2f%%',explode=explode)
plt.legend()
plt.title('Distribution of Properties by Status')
plt.show()

#Displot

sample_df = df.sample(n=50000)
sns.displot(data = sample_df,x = 'bed',hue = 'status')
plt.title("Displot",fontdict=font_title)
plt.xlabel("Total Beds",fontdict=font_label)
plt.ylabel("Count",fontdict=font_label)
plt.tight_layout()
plt.xlim([0,6])
plt.grid()
plt.legend()
plt.show()

#Displot or Histogram with KDE
sample_df = df.sample(n=5000)
sns.displot(data = sample_df,x = 'price',kde = True,label = 'Count')
plt.title("Displot With KDE for Price",fontdict=font_title)
plt.xlabel("Price",fontdict=font_label)
plt.ylabel("Count",fontdict=font_label)
plt.xlim([0,2000000])
plt.tight_layout()
plt.grid()
plt.legend()
plt.show()

#Pairplot

df_pairplot = df.drop(columns=['brokered_by','street','price_per_sqft','total_rooms','year','zip_code','acre_lot'])
ten_df = df_pairplot.sample(n=10000)
sns.pairplot(ten_df,hue = 'status')
plt.suptitle("Pairplot")
plt.tight_layout()
plt.grid()
plt.legend()
plt.show()

#Heatmap with Cbar
plt.figure(figsize=(20,10))
import seaborn as sns
df = pd.DataFrame(df)
corr = df_numerical.corr()
sns.heatmap(corr,annot=True,cbar=True)
plt.title("Heatmap With Cbar",fontdict=font_title)
plt.xlabel("Features",fontdict=font_label)
plt.ylabel("Features",fontdict=font_label)
plt.legend()
plt.show()


#Histogram plot with KDE

sns.histplot(data=df, x='price_per_sqft',kde=True,label = "Price per sqft")
plt.xlim([0,500])
plt.title("Histogram with KDE",fontdict=font_title)
plt.xlabel("Price per sqft",fontdict=font_label)
plt.ylabel("Count",fontdict=font_label)
plt.grid()
plt.legend()
plt.show()

# print("describe\n",df['price_per_sqft'].describe())
# sns.histplot(data=df, x='price_per_sqft',kde=True,label = "Price per sqft",bins=500)
# plt.xlim([0,100000])
# plt.title("Histogram with KDE")
# plt.legend()
# plt.show()
# #CHeck this
# #Histogram Plot with KDE
# sns.histplot(df['price'], kde=True, bins=30, color='blue')
# plt.title('Distribution of Price with KDE')
# plt.xlabel('Price')
# plt.xlim([0,20000000])
# plt.ylabel('Frequency')
# plt.show()
#CHeck this

#QQplot
sm.qqplot(df_numerical['house_size'], line ='s',label = "QQplot")
plt.title('QQ Plot for Price',fontdict=font_title)
plt.xlabel("Theoretical Quantities",fontdict=font_label)
plt.ylabel("Sample Quantities",fontdict=font_label)
plt.grid()
plt.legend()
plt.show()
#CHeck this

#KDE plot will fill, alpha = 0.6, chose a palette, chose a linewidth
sns.kdeplot(data=df['price'],fill=True,alpha=0.6,palette='Blues',linewidth=2,label = "KDE plot")
plt.title('KDE Plot for Price with Fill',fontdict=font_title)
plt.xlabel('Price',fontdict=font_label)
plt.ylabel('Density',fontdict=font_label)
plt.xlim([0,2000000])
plt.grid()
plt.legend()
plt.show()

#CHeck this

sns.kdeplot(data=df['total_rooms'],fill=True,alpha=0.6,palette='Blues',linewidth=2,label = "KDE plot")
plt.title('KDE Plot for total_rooms with Fill',fontdict=font_title)
plt.xlabel('total_rooms',fontdict=font_label)
plt.ylabel('Density',fontdict=font_label)
plt.grid()
plt.legend()
plt.show()


#Im or reg plot with scatter representation and regression line
sample_df = df.sample(n=5000)
plt.figure(figsize=(20,10))
sns.lmplot(data=sample_df, x="house_size", y="price",col="status", row="bath", height=3,
    facet_kws=dict(sharex=False, sharey=False))
plt.suptitle('Price vs. House Size with Regression Line')
plt.tight_layout()
plt.subplots_adjust(top=0.85,right=0.95,bottom=0.15,left=0.15)
plt.grid()
plt.legend()
plt.show()

sample_df = df.sample(n=5000)
plt.figure(figsize=(20,10))
sns.regplot(x='house_size', y='price', data=sample_df, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'},label = "Regplot")
plt.title('Price vs. House Size with Regression Line', fontdict=font_title)
plt.xlabel('House Size (sqft)', fontdict=font_label)
plt.ylabel('Price ($)', fontdict=font_label)
plt.tight_layout()
plt.xlim([0,4000])
plt.ylim([0,2000000])
plt.grid()
plt.legend()
plt.show()
#CHeck this

#Multivariate box or Boxen Plot
plt.figure(figsize=(10, 6))
df_box = df[["bed","bath","acre_lot","total_rooms"]]
sns.boxplot(data=df_box)
plt.title('Multivariate Box Plot',fontdict=font_title)
plt.xlabel('Variables',fontdict=font_label)
plt.ylabel('Values',fontdict=font_label)
plt.grid()
plt.legend(['Bed', 'Bath','Acre_Lot','Total_Rooms'], loc='upper left')
plt.subplots_adjust(top=0.85,right=0.95,bottom=0.15,left=0.15)
plt.show()



# Area plot
df_chicago = df[df['city'] == 'Chicago']
df_chicago_yearly = df_chicago.groupby('year')['price'].mean()
df_chicago_yearly.plot(kind='area', figsize=(12, 6), color='skyblue', alpha=0.6)

plt.title('Total Price Over Year in Chicago',fontdict=font_title)
plt.xlabel('Year',fontdict=font_label)
plt.ylabel('Total Price ($)',fontdict=font_label)
plt.tight_layout()
plt.grid()
plt.legend()
plt.show()

# Violin plot

df_virginia = df[(df['state'] == 'Virginia') & ((df['city'] == 'Falls Church') | (df['city'] == 'Alexandria') | (df['city'] == 'Fairfax'))]
plt.figure(figsize=(12,6))
sns.violinplot(data = df_virginia,x = 'state',y = 'price',hue = 'city')
plt.title('Violin Plot',fontdict=font_title)
plt.xlabel('State',fontdict=font_label)
plt.ylabel('Price ($)',fontdict=font_label)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.grid()
plt.legend()
plt.show()

#Joint plot with KDE and scatter representation
df_joint = sns.JointGrid(x='house_size', y='price', data=sample_df)

g = df_joint.plot_joint(sns.scatterplot, hue='status', data=sample_df)
sns.kdeplot(sample_df.loc[sample_df['status']=='for_sale', 'house_size'], ax=g.ax_marg_x, legend=False)
sns.kdeplot(sample_df.loc[sample_df['status']=='sold', 'house_size'], ax=g.ax_marg_x, legend=False)
sns.kdeplot(sample_df.loc[sample_df['status']=='for_sale', 'price'], ax=g.ax_marg_y, vertical=True, legend=False)
sns.kdeplot(sample_df.loc[sample_df['status']=='sold', 'price'], ax=g.ax_marg_y, vertical=True, legend=False)
plt.suptitle('Joint Plot with Scatter and KDE', fontsize=20, family='serif', color='blue')
plt.subplots_adjust(top=0.95)
plt.legend()
plt.show()

#Rug Plot
sample_df = df.sample(n=10000)

sns.scatterplot(data=sample_df, x="house_size", y="price",label = "Scatter Plot")
sns.rugplot(data=sample_df, x="house_size", y="price", height=.1)
plt.title('Rug Plot of Price vs House Size', fontdict=font_title)
plt.xlabel('House Size', fontdict=font_label)
plt.ylabel('Price ($)',fontdict=font_label)
plt.tight_layout()
plt.grid()
plt.legend()
plt.show()

#3D + Contour

# x = sample_df['house_size']
# y = sample_df['bed']
# X,Y = np.meshgrid(x,y)
# #Z = sample_df['price']
# Z = np.sin(np.sqrt(X**2 + Y**2))
# fig = plt.figure(figsize=(10,10))
# # sin_1 = np.sin(np.sqrt(x**2))
# # sin_2 = np.sin(np.sqrt(y**2))
# ax = fig.add_subplot(111,projection='3d')
# ax.plot_surface(X, Y, Z, cmap='coolwarm', linewidth=1,alpha = 0.6)
# #zero= np.zeroes(800)
# ax.contour(X, Y, Z, zdir='z', offset=-6, cmap='coolwarm', linewidths=1)
# ax.contour( X,Y,Z, zdir='x', offset=-5, cmap='coolwarm', linewidths=1)
# ax.contour(X,Y,Z, zdir='y', offset=5, cmap='coolwarm', linewidths=1)
#
#
# ax.set_box_aspect([1,1,1])
#
# ax.set_xlabel('X label',fontdict = font_label)
# ax.set_ylabel('Y label',fontdict = font_label)
# ax.set_zlabel('Z label',fontdict = font_label)
# ax.set_title(' 3D + Contour Plot',fontdict = font_title)
# # ax.set_zlim([-6,2])
# plt.tight_layout()
# plt.show()

#Cluster Map
sample_df = df_numerical[["bed","bath","acre_lot","total_rooms"]].sample(n=10000)
sns.clustermap(sample_df)
plt.legend()
plt.show()

#Hexbin
plt.figure(figsize=(10, 6))
sns.scatterplot(x='house_size', y='price', data=df, color='blue', alpha=0.5)
plt.hexbin(df['house_size'], df['price'], gridsize=30, cmap='coolwarm', mincnt=1)
plt.colorbar(label='Color Bar')
plt.title('Hexbin Plot of House Size vs Price', fontdict=font_title)
plt.xlabel('House Size (sq ft)', fontdict=font_label)
plt.ylabel('Price ($)', fontdict=font_label)
plt.grid()
plt.show()


# Strip plot
df_strip = df[['price', 'bed']]
plt.figure(figsize=(10, 6))
sns.stripplot(x='bed', y='price', data=df_strip, jitter=True, palette='viridis', alpha=0.7)
plt.xlabel('Number of Bedrooms',fontdict=font_label)
plt.ylabel('Price ($)',fontdict=font_label)
plt.title('Strip Plot of Price vs. Number of Bedrooms',fontdict=font_title)
plt.grid()
plt.show()

#Swarm Plot
sample_df = df.sample(n=1000)
sns.swarmplot(data=sample_df, x="status", y="price")
plt.title("Swarm Plot",fontdict=font_title)
plt.xlabel('Status', fontdict=font_label)
plt.ylabel('Price ($)',fontdict=font_label)
plt.grid()
plt.show()

#Subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
status_counts = df['status'].value_counts()
axes[0, 0].pie(status_counts, labels=status_counts.index, autopct='%1.1f%%')
axes[0, 0].set_title('Distribution of Property Status', fontdict=font_title)
axes[0,0].legend()
state_counts = df['state'].value_counts().head(6)
sns.barplot(x=state_counts.index, y=state_counts.values, ax=axes[0, 1])
axes[0, 1].set_title('Number of Properties per City',fontdict=font_title)
axes[0, 1].set_xlabel('State', fontdict=font_label)
axes[0, 1].set_ylabel('Number of Properties', fontdict=font_label)
axes[0, 1].legend()
avg_price_by_bed = df.groupby('bed')['price'].mean().sort_values(ascending=False)
sns.barplot(x=avg_price_by_bed.index, y=avg_price_by_bed.values, ax=axes[1, 0])
axes[1, 0].set_title('Average Price by Number of Bedrooms', fontdict=font_title)
axes[1, 0].set_xlabel('Number of Bedrooms',fontdict=font_label)
axes[1, 0].set_ylabel('Average Price ($)', fontdict=font_label)
axes[1, 0].legend()
state_counts = df['state'].value_counts().head(12)
axes[1, 1].pie(state_counts, labels=state_counts.index, autopct='%1.1f%%')
axes[1, 1].set_title('Property Distribution by State', fontdict=font_title)
plt.tight_layout()
axes[1, 1].legend()
plt.show()


#Bivariate plot
hi
sns.displot(data = sample_df,x = 'price',y = 'house_size')
plt.title("Bivariate plot",fontdict=font_title)
plt.xlabel("Price",fontdict=font_label)
plt.ylabel("House Size",fontdict=font_label)
plt.tight_layout()
plt.xlim([0,2000000])
plt.ylim(0,4000)
plt.grid()
plt.show()

#Bivariate kde plot
sns.displot(data = sample_df,x = 'price',y = 'house_size',kind='kde',fontdict=font_label)
plt.title("Bivariate kde plot")
plt.tight_layout()

plt.xlim([0,5000000])
plt.ylim(0,7500)
plt.show()




#Diagonals nee x lim












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



#Area Plot
ax = df_numerical.plot.area(y='total_rooms',rot=0)
plt.title('Area plot of total Rooms')
plt.xlabel('Area')
plt.ylabel('Total Rooms')
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
sns.clustermap(data=df_numerical,cmap='viridis')
plt.title('Cluster map Plot')
plt.show()
# Hexbin
plt.figure(figsize=(10, 6))
df_hex = df[['price', 'bed']].dropna()
sns.jointplot(x=df_hex['bed'], y=df_hex['price'], kind="hex", color="#4CB391")
plt.colorbar(label='Number of Properties')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Price ($)')
plt.title('Hexbin Plot of Price vs. Number of Bedrooms')
plt.show()

plt.hexbin(df_hex['bed'], df_hex['price'], gridsize=15, cmap='plasma')
plt.colorbar(label='Number of Properties')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Price ($)')
plt.title('Hexbin Plot of Price vs. Number of Bedrooms')
plt.show()


# Strip plot
df_strip = df[['price', 'bed']].dropna()
plt.figure(figsize=(10, 6))
sns.stripplot(x='bed', y='price', data=df_strip, jitter=True, palette='viridis', alpha=0.7)
plt.xlabel('Number of Bedrooms')
plt.ylabel('Price ($)')
plt.title('Strip Plot of Price vs. Number of Bedrooms')
plt.show()
# Swarm plot
plt.figure(figsize=(10, 6))
sns.swarmplot(x='bed', y='price', data=df_strip)
plt.xlabel('Number of Bedrooms')
plt.ylabel('Price ($)')
plt.title('Swarm Plot of Price vs. Number of Bedrooms')
plt.show()


# df_numerical.boxplot(column=df['price'])
# df_numerical.boxplot(column=df['bed'])
# #sns.boxplot(data=df_numerical, orient="h", palette="Set2")
# plt.title("Box Plot for df_numerical")
# plt.tight_layout()
# plt.show()



#


