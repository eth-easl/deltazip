import requests

endpoint = "http://localhost:8092"

res = requests.post(
    f"{endpoint}/v1/completions",
    json={
        "model": "delta-1",
        "prompt": "USER: What is the capital of France?\nASSISTANT:",
    },
)

print(res)
