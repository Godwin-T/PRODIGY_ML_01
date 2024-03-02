import requests

url = "http://127.0.0.1:9696/predict"
data_path = "../data/house-prices-advanced-regression-techniques/test.csv"
response = requests.post(url, json=data_path).json()
print(response["Status"])
