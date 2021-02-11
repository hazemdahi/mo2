 import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("Final Database.csv")


# take a look at the dataset
#df.head()

#use required features
cdf = df[['Temp','DO','pH','BOD','NITRATE_N_NITRITE_N', 'WQI']]

#Training Data and Predictor Variable
# Use all data for training (tarin-test-split not used)
cols = cdf.shape[1]
x = cdf.iloc[:,0:cols-1]
y = cdf.iloc[:,cols-1:cols]

regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(x, y)

# Saving model to disk
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
pickle.dump(regressor, open('model.pkl','wb'))

'''
#Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2.6, 8, 10.1]]))
'''