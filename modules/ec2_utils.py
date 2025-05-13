import requests
import json

EC2_API_URL = "http://3.38.250.18:8000/ingest"

def send_to_ec2(user_data):
    try:
        res = requests.post(
            EC2_API_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(user_data, ensure_ascii=False).encode("utf-8")
        )
        return res.status_code == 200
    except:
        return False
