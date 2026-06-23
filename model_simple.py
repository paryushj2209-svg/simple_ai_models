import numpy as np

class linear_regression:
    def __init__(self,train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def train(self,x_train,y_train):
        sum_x = np.sum(x_train)
        sum_y = np.sum(y_train)
        sum_x_squared = np.sum(x_train**2)
        sum_xy = np.sum(x_train*y_train)
        n = len(x_train)
        m = (n*sum_xy - sum_x*sum_y) / (n*sum_x_squared - sum_x**2)
        c = (sum_y - m*sum_x) / n
        return m, c
    
    def test(self,m,c, x_test, y_test):
        y_pred = self.predict(x_test,m,c)
        mse = np.mean((y_test - y_pred)**2)
        rmse = np.sqrt(mse)
        r2 = 1 - (np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2))
        print("Mean Squared Error:", mse)
        print("Root Mean Squared Error:", rmse)
        print("R-squared:", r2)

    def predict(self,x,m,c):
        return m*x + c


