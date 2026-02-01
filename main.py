import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

df = pd.read_csv("train.csv")
print("head: \n", df.head())
print("info: \n", df.info())
print("is na: \n", df.isna().sum())

df["city"] = df["city"].map({"sj": 0, "iq": 1})

df = df.fillna(df.mean(numeric_only=True))


df = df.fillna(0)

df = df.drop(columns=["week_start_date"])

X = df.drop(columns=["total_cases"])
y = df["total_cases"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, shuffle=False
)
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

#df["cases_lag_1"] = df.groupby("city")["total_cases"].shift(1)
#df["cases_lag_2"] = df.groupby("city")["total_cases"].shift(2)

test_df = pd.read_csv("test.csv")

test_df["city"] = test_df["city"].map({"sj": 0, "iq": 1})
test_df = test_df.drop(columns=["week_start_date"])
test_df = test_df.fillna(test_df.mean(numeric_only=True))

test_preds = model.predict(test_df)

# Output the train error rate
print("error rate for training set: ", model.score(X_train, y_train))
# Output the test error rate
print("error rate for validation set: ", model.score(X_val, y_val))

# turn test_preds into dataframe and produce the CSV file named "rf_prediction.csv"
rf = pd.DataFrame(
    {
        "id": test_df.iloc[:,0],
        "target_variable": test_preds
    }    
)
rf.to_csv("rf_prediction.csv", index=False)

# Multiple Linear Regression below:

# fit the multiple linear regression
linear = LinearRegression()
linear.fit(X_train, y_train)
test_preds = linear.predict(test_df)

# Evaluate the model
# Output the train error rate
print("error rate for training set: ", linear.score(X_train, y_train))
# Output the test error rate
print("error rate for validation set: ", linear.score(X_val, y_val))

# turn test_preds into dataframe and produce the CSV file named "linear_prediction.csv"
linear_df = pd.DataFrame(
    {
        "id": test_df.iloc[:,0],
        "target_variable": test_preds
    }    
)
linear_df.to_csv("linear_prediction.csv", index=False)


# Ridge regression below:

# Fit the ridge regression model
ridge = Ridge() 
ridge.fit(X_train, y_train)

# Output the train error rate
print("ridge regression:")
print("error rate for training set: ", ridge.score(X_train, y_train))
# Output the test error rate
print("error rate for validation set: ", ridge.score(X_val, y_val))

# get the final prediction
test_preds = ridge.predict(test_df)

# turn test_preds into dataframe and produce the CSV file named "ridge_prediction.csv"
ridge_df = pd.DataFrame(
    {
        "id": test_df.iloc[:,0],
        "target_variable": test_preds
    }    
)
ridge_df.to_csv("ridge_prediction.csv", index=False)


# LASSO regression below:

# Fit the LASSO regression model
lasso = Lasso() # AI suggested alpha=0.1
lasso.fit(X_train, y_train)

# Output the train error rate
print("error rate for training set: ", lasso.score(X_train, y_train))
# Output the test error rate
print("error rate for validation set: ", lasso.score(X_val, y_val))

test_preds = lasso.predict(test_df)

# turn test_preds into dataframe and produce the CSV file named "rf_prediction.csv"
lasso_df = pd.DataFrame(
    {
        "id": test_df.iloc[:,0],
        "target_variable": test_preds
    }    
)
lasso_df.to_csv("lasso_prediction.csv", index=False)
