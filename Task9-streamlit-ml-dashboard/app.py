import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.title("House Price Dashboard")

df = pd.read_csv('data/house_prices.csv')
st.write("## Dataset Preview")
st.dataframe(df)

# Plot histogram
fig = px.histogram(df, x="Price", nbins=10, title="Price Distribution")
st.plotly_chart(fig)

# Modeling
X = df[['Rooms', 'Area']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

st.write("### Predictions")
preds = model.predict(X_test)
st.write(preds)
