## fraud predicting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bisect
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import auc
from sklearn.metrics import precision_score
from sklearn.ensemble import RandomForestClassifier
df1 = pd.read_csv('Fraud_Data.csv')
#print(df1.head())
df1.columns = [x.strip('" "') for x in df1.columns]

df1['country'] = ''
df2 = pd.read_csv('IpAddress_to_Country.csv')
#print(df2.head())
L2 = len(df2)
"""
def matching_IP_address(ip =None):
    #type(ip)
    mact = (ip <= df2['upper_bound_ip_address']) & (ip >= df2['upper_bound_ip_address'])
    if mact.any() :
        return df2['country'][ip].to_string(index=False)
    else:
        return 'unknown'


"""



class IpLookupTable(object):
    def __init__(self, df):
        """
        input:
            df: DataFrame read from 'IpAddress_to_Country.csv'
        """
        self._nrows = df.shape[0]
        # add two more slots, one is for the ipaddress < minimum ipaddress from the file
        # another is for the ipaddress > maximum ipaddress from the file
        self._ip_lowbounds = [0 for _ in range(self._nrows + 2)]
        self._countries = ["Unknown" for _ in range(self._nrows + 2)]

        # start from 1, because 0-th cell is for ipaddress < minimal known ipaddress
        for r in range(1, self._nrows + 1):
            self._ip_lowbounds[r] = df.iloc[r - 1, 0]
            self._countries[r] = df.iloc[r - 1, 2]
            # assume the file is in ascending order
            assert self._ip_lowbounds[r] > self._ip_lowbounds[r - 1]


        # we cannot assign all ip> last low boundary to be that country
        # so I create a dummy ipaddress which larger than maximal known ipaddress
        self._ip_lowbounds[self._nrows + 1] = df.iloc[self._nrows - 1, 1] + 1

    def find_country(self, ip):
        index = bisect.bisect(self._ip_lowbounds, ip) - 1
        # within the range, or in last cell which doesn't have next cell
        assert ip >= self._ip_lowbounds[index] and (index == self._nrows + 1 or ip < self._ip_lowbounds[index + 1])
        return self._countries[index]


ip2country = pd.read_csv("IpAddress_to_Country.csv")
iplookuptable = IpLookupTable(ip2country)
df1["country"] = df1['ip_address'].apply(iplookuptable.find_country)
#print(df1.head())
df1['total_time'] = (pd.to_datetime(df1['purchase_time'])- pd.to_datetime(df1['signup_time'])).dt.total_seconds()
df1.drop(['signup_time','purchase_time'],axis =1, inplace= True)
#print(df1.head())
# how many times the same ip address is used
#print(df1['device_id'].value_counts())
n_dev_shared = df1['device_id'].value_counts()
df1['n_dev_shared'] = df1['device_id'].map(n_dev_shared)
df1.drop(['device_id'], axis =1, inplace= True)
#print(df1.head())
## how many user are from the same country
n_country_shared = df1['country'].value_counts()
df1['n_country_shared'] = df1['country'].map(n_country_shared )
df1.drop(['country'], axis =1, inplace= True)
#print(df1.head())

# convert sex to 1 or 0
df1.loc[df1['sex'] == 'F',['sex']] =1
df1.loc[df1['sex'] == 'M',['sex']] =0
df1 = pd.get_dummies(df1, columns= ['source','browser'])
df1.drop(['source_Direct','browser_Opera'], axis =1, inplace= True)

df1.rename(columns = {'class' : 'is_fraud'}, inplace= True)
print(df1.head())
df1.to_csv('fraud_cleaned.csv', index = 'user_id')

##### Train the model
seed = 999
X= df1.loc[:, df1.columns != 'is_fraud']
y= df1['is_fraud']

x_train, x_test, y_train, y_test =train_test_split(X,y, random_state = 42)

rnd_f = RandomForestClassifier()
rnd_f.fit(x_train, y_train)
print('\nRandom_forest -  \n\n',pd.crosstab(y_train, rnd_f.predict(x_train), rownames= ['Actuall'],
                                            colnames=['Predicted']))
print('\nRandom_forest - prediction report', precision_score(y_test, rnd_f.predict(x_test)))
importances = rnd_f.feature_importances_
print(importances )
#importances.sort(ascending = 'False')
model_ranks= pd.Series(rnd_f.feature_importances_, index = x_train.columns, name = 'Importance').sort_values(
    ascending= False, inplace= False)
model_ranks.index.name = 'Variable'
top_features = model_ranks.iloc[:5].sort_values(ascending= True, inplace= False)
#
plt.figure(figsize = (10,8))
ax = top_features.plot(kind = 'barh')
_ = ax.set_title('Variable Importance Plot')
_ = ax.set_xlabel('Mean decrease in Variance')
#plt.show()
#both = pd.concat([y_train, rnd_f.predict(x_train)], axis =1)
fpr, tpr, thresholds = metrics.roc_curve(y_test,rnd_f.predict(x_test), pos_label = 1)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color = 'darkorange', label= 'ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0,1], [0,1], color = 'navy', linestyle = '--')
plt.xlim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc= 'lower right')
plt.show()