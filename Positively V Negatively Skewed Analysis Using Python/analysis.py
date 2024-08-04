#Reading Data from Github
import statistics

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
#Read the csv file
order_df = pd.read_csv('https://raw.githubusercontent.com/swapnilsaurav/OnlineRetail/master/orders.csv')
#Display all the column names
print(list(order_df.columns))
X=pd.to_datetime(order_df['order_delivered_customer_date']) - pd.to_datetime(order_df['order_purchase_timestamp'])

for i in range(0,len(X)):
    X[i]=X[i].days
plt.figure(figsize=(10,5))

sns.barplot(x=X.value_counts().sort_values(ascending=False).head(30).index,y=X.value_counts().sort_values(ascending=False).head(30).values)
plt.xlabel('Delivery Days')
plt.ylabel('Frequency')
plt.show()
info=X.describe()

print("Mean Value of Delivery Days: {:0.1f}".format(np.mean(X)))
print("Median Value of Delivery Days: " , np.median(X))
print("Mode Value of Delivery Days: " , statistics.mode (X))
print("Standard Deviation of Delivery Days: {:0.1f}".format(X.std()))

