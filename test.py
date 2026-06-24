import model_simple as model
import pandas as pd
import numpy as np

train_data = pd.read_csv('archive(1)/train.csv')
test_data = pd.read_csv('archive(1)/test.csv')

regression_model = model.linear_regression(train_data, test_data)
m,c = regression_model.train(train_data['x'], train_data['y'])
regression_model.test(m, c, test_data['x'], test_data['y'])

data = pd.read_csv('/home/user/Projects/AI/Ice_cream selling data.csv')
train_data, test_data = model.train_test_split(data, test_size=0.2)
test_data = test_data.reset_index(drop=True)
regression_model_poly = model.polynomial_regression(train_data, test_data, degree=2)
b, e = regression_model_poly.train(train_data['Temperature (°C)'], train_data['Ice Cream Sales (units)'])
regression_model_poly.test(b, test_data['Temperature (°C)'], test_data['Ice Cream Sales (units)'])
