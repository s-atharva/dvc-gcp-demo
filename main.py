import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris(as_frame=True)
df = iris.frame

X = df.drop(columns=['target'])
y = df['target']

iris_train, iris_test = train_test_split(df, test_size=0.2, random_state=42)

iris_train.to_csv("data/train.csv", index=False)
iris_test.to_csv("data/test.csv", index=False)

print("Train and Test data created successfully")
