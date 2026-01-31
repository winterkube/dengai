import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

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