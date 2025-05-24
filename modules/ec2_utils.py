import requests
import json
from config import EC2_API_URL

# 입력 받은 USER 데이터를 EC2로 전송
def send_to_ec2(user_data):
    try:
        response = requests.post(
            EC2_API_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(user_data, ensure_ascii=False).encode("utf-8")
        )
        return response.status_code == 200
    except Exception as e:
        print(f"[EC2 전송 실패] {e}")
        return False