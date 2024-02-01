"""
Model config
"""

from pydantic import BaseModel
import pandas as pd
from sklearn.linear_model import LinearRegression


def model(data: str, country: str):
	df = pd.read_csv(data)

	country_data = df[['Date', country]]

	X = country_data['Date'].apply(lambda x: int(x.split('-')[2])).values.reshape(-1, 1)
	y = country_data[country].values

	model = LinearRegression()
	model.fit(X, y)

	next_day = int(df['Date'].iloc[-1].split('-')[2]) + 1
	next_day_weather = model.predict([[next_day]])

	return round(next_day_weather[0], 2)


class WetherModel(BaseModel):
	"""
	Base model
	"""
	navigation: str
