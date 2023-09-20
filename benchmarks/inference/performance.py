import requests

endpoint = 'http://localhost:8000'

response_time = []
for i in range(10):
    res = requests.get(endpoint + '/inference')
    print(res.headers)
    print(res.json())