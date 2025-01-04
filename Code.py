import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df=pd.read_csv('auto-mpg.csv')
df=df.drop_duplicates()
len(df)
df.columns
for col in df.columns:
    print(df[col].isnull().sum())
df.dtypes    
df['horsepower'].max()
print(df[df['horsepower'].str.contains(",")])
df['horsepower']=pd.to_numeric(df['horsepower'],errors='coerce')
df['horsepower'].max()
df=df.dropna()
len(df)
col=['cylinders','displacement','horsepower','weight','mpg']
cor=df[col].corr()
cor
sns.pairplot(df[["mpg", "cylinders", "displacement", "weight"]], diag_kind="kde")

X=df[[ "cylinders", "displacement", "weight"]]
y=df['mpg']
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

lr=LinearRegression()lr.fit(X_train,y_train)
y_pred= lr.predict(X_test) 
sns.distplot(y_test - y_pred, bins = 20)
plt.legend()
plt.title("Residuals")
plt.show()
