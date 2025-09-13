# Jupyter Notebook Walkthrough

## Load Dataset
```python
import pandas as pd
df = pd.read_csv('data/house_prices.csv')
df.head()
```
**Output:**
```
   Rooms  Area Location   Price
0      2   800        A  100000
1      3  1200        B  150000
2      4  1500        A  200000
3      3  1100        C  130000
4      5  2000        B  300000
```

## Summary Stats
```python
df.describe()
```
**Output:**
```
          Rooms         Area          Price
count  6.000000     6.000000       6.000000
mean   3.500000  1366.666667  188333.333333
std    1.048809   444.947829   75368.394060
min    2.000000   800.000000  100000.000000
25%    3.000000  1100.000000  137500.000000
50%    3.500000  1350.000000  175000.000000
75%    4.250000  1625.000000  237500.000000
max    5.000000  2000.000000  300000.000000
```

## Train simple regression model (RandomForest)
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

X = df[['Rooms', 'Area']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
preds
```
**Output:**
```
array([127000., 205000.])
```
