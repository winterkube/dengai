import pandas as pd
import csv

# with open('train.csv', mode='r', newline='', encoding='utf-8') as file:
#     reader = csv.reader(file)
#     for row in reader:
#         print(row[0])
#         if (row == 1):
#             break

df = pd.read_csv("train.csv")
print(df)