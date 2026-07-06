import model_simple as model
import pandas as pd
import numpy as np

train_data = pd.read_csv('linear_reg_data/train.csv')
test_data = pd.read_csv('linear_reg_data/test.csv')

# regression_model = model.linear_regression(train_data, test_data)
# m,c = regression_model.train(train_data['x'], train_data['y'])
# regression_model.test(m, c, test_data['x'], test_data['y'])

# data = pd.read_csv('linear_reg_data/data.csv')
# train_data, test_data = model.train_test_split(data, test_size=0.2)
model = model.polynomial_regression(train_data, test_data, degree=2)
x,b = model.train(train_data['x'], train_data['y'])
print("Coefficients:", b)


print(data.head())