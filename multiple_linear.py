import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

from sklearn import linear_model
from sklearn.metrics import r2_score

df = pd.read_csv("data/epl2020.csv")

cdf = df[['npxGD','xpts','xG','xGA','deep']]
print(cdf.head(9))

msk = np.random.rand(len(df)) < 0.5
train = cdf[msk]
test = cdf[~msk]

regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['xG','xGA','deep']])
train_y = np.asanyarray(train[['xpts']])
regr.fit (train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)

test_x = np.asanyarray(test[['xG','xGA','deep']])
test_y = np.asanyarray(test[['xpts']])
test_y_ = regr.predict(test_x)

plt.scatter(train.deep, train.xpts,  color='blue')
#plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("deep")
plt.ylabel("xpts")
plt.show()

print("Residual sum of squares: %.2f"
      % np.mean((test_y_ - test_y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(test_x, test_y))
