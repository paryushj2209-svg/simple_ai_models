import numpy as np

def inverse(matrix):
    return np.linalg.inv(matrix)

class method:
    def __init__(self):
        pass

    def step(self, X):
        for x in X:
            if x<=0:
                return 0
        return 1
        
    def relu(self, X):
        for x in X:
            return max(0,x)
    
    def linear(self,X):
        return X
    
def train_test_split(data, test_size=0.2):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    train_x = data.iloc[train_indices, :-1]
    train_y = data.iloc[train_indices, -1]
    test_x = data.iloc[test_indices, :-1]
    test_y = data.iloc[test_indices, -1]
    return train_x, train_y, test_x, test_y

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
        accuracy = np.mean((y_test - y_pred)**2 < 0.01)
        print("Accuracy:", accuracy)
        print("Mean Squared Error:", mse)
        print("Root Mean Squared Error:", rmse)
        print("R-squared:", r2)

    def predict(self,x,m,c):
        return m*x + c

class logistic_regression:
    def __init__(self,train_data, test_data, threshold=0.5):
        self.train_data = train_data
        self.test_data = test_data
        self.threshold = threshold

    def pred(self,x_train,Z):
        z = Z(x_train)
        y_pred = 1 / (1 + np.exp(-z))
        return y_pred
    
    def test(self,x_test, y_test):
        y_pred = self.pred(x_test, self.Z)
        y_pred_class = (y_pred >= self.threshold).astype(int)
        accuracy = np.mean(y_pred_class == y_test)
        mse = np.mean((y_test - y_pred)**2)
        rmse = np.sqrt(mse)
        r2 = 1 - (np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2))
        print("Accuracy:", accuracy)
        print("Mean Squared Error:", mse)
        print("Root Mean Squared Error:", rmse)
        print("R-squared:", r2)
