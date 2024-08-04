#Getting the data and cleaning it

import  pandas as pd
import numpy as np

#Read the csv file
order_df = pd.read_csv("https://raw.githubusercontent.com/swapnilsaurav/OnlineRetail/master/order_items.csv")

#Display all the column names
print(list(order_df.columns))

#Required columns
order_df = order_df[['order_id','product_id','price']]
#print(order_df)

prod_df = pd.read_csv("https://raw.githubusercontent.com/swapnilsaurav/OnlineRetail/master/products.csv")
#Display all the column names
print(list(prod_df.columns))

#Required columns
prod_df = prod_df[['product_id','product_category_name']]
#print(prod_df)

#Read category translation file
cat_df = pd.read_csv("https://raw.githubusercontent.com/swapnilsaurav/OnlineRetail/master/product_category_name.csv")

#Display all the column names
print(list(cat_df.columns))
#Output : ['1 product_category_name', '2 product_category_name_english']

#Let's Rename the column names
cat_df = cat_df.rename(columns={'1 product_category_name':'product_category_name','2 product_category_name_english':'product_category'})
print(list(cat_df.columns))

#Final dataset - merge tab1 and tab2
data = pd.merge(order_df,prod_df, on='product_id',how='left')

#Now merge with category to get English category
data = pd.merge(data, cat_df, on='product_category_name',how='left')

#Check for Missing Data Percentage List
for col in data.columns:
    pct_missing = np.mean(data[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))
print("\n")

#product_category_name - 1% - lets create a new category called Unknown
data['product_category'] = data['product_category'].fillna(("Unknown"))

#Check if all rows have been accounted for
#if not then merge didnt happen correctly

print("Number of rows: \n\n order_items[{}], \n\n MergedData[{}]".format(order_df.count(),data.count()))

#Note : Number of rows in order_items and MergedData should be same

#if you want to push the content to a csv file and
#perform manual test, then uncomment the below line
#data.to_csv("TestingMerge.csv")

#We are now ready to perform the Pareto analysis

#Code 2 : Analyzing using Pareto

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
df=data[['price','product_category']]
df.set_index(data['product_category'])

#Initially test with small dataset to see what you get
#df = df.head(100) #review with the smaller dataset
#print(df)

#Analysis 1 : What is the most in demand product category?
sns.countplot(df['product_category'], order=df['product_category'].value_counts().index)
plt.title('Product Categories based on Demand'.title(),fontsize=20)
plt.ylabel('count'.title(),fontsize=14)
plt.xlabel('product category'.title(),fontsize=14)
plt.xticks(rotation=90, fontsize=10)
plt.yticks(fontsize=12)
plt.show()

#2 : Which categories generates high sales-Pareto
#Sort the values in the descending order
quant_variable = df['price']
by_variable = df['product_category']

column = 'price'
group_by = 'product_category'

df = df.groupby(group_by)[column].sum().reset_index()
df = df.sort_values(by=column,ascending=False)
df["cumpercentage"] = df[column].cumsum()/df[column].sum()*100
fig, ax = plt.subplots(figsize=(20,5))
ax.bar(df[group_by], df[column], color="C0")
ax2 = ax.twinx()
ax2.plot(df[group_by], df["cumpercentage"], color="C1",
marker="D", ms=7)
ax2.yaxis.set_major_formatter(PercentFormatter())
ax.tick_params(axis="x", rotation=90)
ax.tick_params(axis="y", colors="C0")
ax2.tick_params(axis="y", colors="C1")
plt.title('Product Categories based on Sales'.title(),fontsize=20)

plt.show()

#Variation 2
#Plotting above graph with only top 40 categories, rest as other categories
total=quant_variable.sum()

df = df.groupby(group_by)[column].sum().reset_index()
df = df.sort_values(by=column,ascending=False)
df["cumpercentage"] = df[column].cumsum()/df[column].sum()*100
threshold = df[column].cumsum() /5 #20%

df_above_threshold = df[df['cumpercentage']< threshold]
df=df_above_threshold
df_below_threshold = df[df['cumpercentage'] >= threshold]
sum = total - df[column].sum()
restbarcumsum = 100 - df_above_threshold['cumpercentage'].max()
rest = pd.Series(['OTHERS', sum, restbarcumsum], index=[group_by,column, 'cumpercentage'])
df = df._append(rest,ignore_index=True)
df.index = df[group_by]
df = df.sort_values(by='cumpercentage',ascending=True)
fig, ax = plt.subplots()
ax.bar(df.index, df[column], color="C0")
ax2 = ax.twinx()
ax2.plot(df.index, df["cumpercentage"], color="C1", marker="D", ms=7)
ax2.yaxis.set_major_formatter(PercentFormatter())
ax.tick_params(axis="x", colors="C0", labelrotation=90)
ax.tick_params(axis="y", colors="C0")
ax2.tick_params(axis="y", colors="C1")
plt.title('Product Categories based on Sales-2'.title(),fontsize=20)
plt.show()



