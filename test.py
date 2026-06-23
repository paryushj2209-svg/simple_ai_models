import model_simple as model
import pandas as pd
import numpy as np

train_data = pd.read_csv('archive(1)/train.csv')
test_data = pd.read_csv('archive(1)/test.csv')

regression_model = model.linear_regression(train_data, test_data)
m,c = regression_model.train(train_data['x'], train_data['y'])
regression_model.test(m, c, test_data['x'], test_data['y'])
