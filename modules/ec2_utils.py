import requests
import json
import os
from dotenv import load_dotenv

load_dotenv(override = True)

EC2_API_URL = os.getenv("EC2_PUBLIC_ADDR")

# 입력 받은 USER 데이터 ec2로 전송
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