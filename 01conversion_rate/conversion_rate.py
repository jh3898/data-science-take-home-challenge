"""
test conversion rate for a company's website
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassfier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


### import csv file
df = pd.read_csv('conversion_data.csv')

df.dropna(inplace= True)
print(df.head())
y = df['converted']
x = df.drop(['converted'], axis =1)

pd_source = pd.get_dummies(x['source'])
pd_country = pd.get_dummies(x['country'])
new_col = ['age','new_user','total_pages_visited']
df2= pd.concat([x[new_col],pd_source,pd_country],axis = 1)
#df2.sort_values(by= ['country'], ascending = False).sum()
#df3 =df2.groupby(['country']).agg(['count'])
df2.head()

x_train, x_test, y_train, y_test = train_test_split(df2, y, random_state = 42)
rf_fit = RandomForestClassfier(n_estimators = 1000, criterion = 'gini', max_depth = 300, min_sample_split = 3, min_samples_leaf = 1)
rf_fit.fit(x_train, y_train)
#### importance plotting 
importance = rf_fit.feature_importances_
indices = np.argsort(importances)[::-1]

colnames = list(x_train.columns)
print('\nFeature ranking:\n')
for f in range(x_train.shape[1]):
    print('Feature',indices[f],',', colnames[indices[f]], round(importance[indices[f]],4))
    
plt.figure()
plt.bar(range(x_train.shape[1]), importances[indices], color = 'r', yerr = std[indices],align = 'center')
plt.xticks(range(x_train.shape[1]),indices)
plt.xlim([-1, x_train.shape[1]])
plt.show()

y_predict = rf.predict(x_train)
print(pd.crosstab(y_train,y_predict ))

