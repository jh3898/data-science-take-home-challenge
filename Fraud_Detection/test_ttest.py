import numpy as np
from scipy import stats
xbar = 990; mu0 = 1000; s = 12.5; n= 30
### test statistic
t_sample = (xbar -mu0)/(s/np.sqrt(float(n)))
print('Test statistic:', round(t_sample, 2))

### critical value from t-table
alpha = 0.05
t_alpha = stats.t.pdf(alpha, n-1)
print(t_alpha)

##3 lower tail p-value from t-table
p_val = stats.t.sf(np.abs(t_sample),n-1)
print(p_val)