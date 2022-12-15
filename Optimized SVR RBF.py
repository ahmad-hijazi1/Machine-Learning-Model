from skopt import BayesSearchCV
import pandas as pd
from sklearn.svm import SVR
import sklearn.metrics as metrics
col_list_x = ['Ni', 'Ti', 'Cu', 'Fe', 'Pd', 'cs', 'arc', 'mr', 'en', 'ven', 'dor']
col_list_y = ['Delta T']
X_train = pd.read_csv("C:/Users/ahmad/Thesis Results November 2022/Training Set Iteration 13.csv", usecols = col_list_x)
y_train = pd.read_csv("C:/Users/ahmad/Thesis Results November 2022/Training Set Iteration 13.csv", usecols = col_list_y)
X_test = pd.read_csv ("C:/Users/ahmad/Thesis Results November 2022/Search Space Iteration 13.csv", usecols = col_list_x)
opt = BayesSearchCV(
    SVR(),
    {
        'C': (0.01, 0.01000001, 'log-uniform'),
        'gamma': (0.1, 0.1000001, 'log-uniform'),
        'kernel': ['rbf'],  # categorical parameter
    },
    n_iter=2,
    cv=5
)
opt.fit(X_train, y_train)
y_predicted = opt.predict(X_test)


#SVR Linear
opt2 = BayesSearchCV(
    SVR(),
    {
        'C': (0.01, 0.01000001, 'log-uniform'),
        'gamma': (0.1, 0.1000001, 'log-uniform'),
        'kernel': ['linear'],  # categorical parameter
    },
    n_iter=2,
    cv=5
)

opt2.fit(X_train, y_train)
y_predicted2 = opt2.predict(X_test)
#print(opt)
print(min(y_predicted2))
print(min(y_predicted))
print('SVR Model| MSE on test set: %.4f'%metrics.mean_squared_error(y_predicted2, y_predicted))
print('SVR Model| MAE on test set: %.4f'%metrics.mean_absolute_error(y_predicted2, y_predicted))
df = pd.read_csv("C:/Users/ahmad/Thesis Results November 2022/Search Space Iteration 13.csv")
df["Delta T"] = y_predicted
df.to_csv("C:/Users/ahmad/Thesis Results November 2022/Search Space SVR RBF Iteration 13.csv", index=True)
df2 = pd.read_csv("C:/Users/ahmad/Thesis Results November 2022/Search Space Iteration 13.csv")
df2["Delta T"] = y_predicted2
df2.to_csv("C:/Users/ahmad/Thesis Results November 2022/Search Space SVR Linear Iteration 13.csv", index=True)
