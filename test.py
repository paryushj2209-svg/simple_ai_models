import pandas as pd
import model_simple as ms
import numpy as np
print("\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
print("\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")

test = pd.read_csv('/home/user/Projects/simple_ai_models/linear_reg_data/test.csv')
train = pd.read_csv('/home/user/Projects/simple_ai_models/linear_reg_data/train.csv')
train = ms.handle_nan(train)
test = ms.handle_nan(test)
lr = ms.linear_regression()
x_train = train["x"]
y_train = train["y"]
lr = ms.linear_regression()
m, c = lr.train(x_train, y_train)
print("Slope (m):", m)
print("Intercept (c):", c)
x_test = test["x"]
y_test = test["y"]
lr.test(m, c, x_test, y_test)

print("\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
print("\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")

pr = ms.polynomial_regression(degree=100)
coeffs = pr.train(x_train, y_train)
print("Polynomial Coefficients:", coeffs)
pr.test(coeffs, x_test, y_test)

print("\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
print("\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
