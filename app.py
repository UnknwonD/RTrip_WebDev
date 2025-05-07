from flask import Flask, request, render_template, redirect, url_for
from dotenv import load_dotenv
from datetime import datetime
import os
import boto3
import uuid
import json
import requests

app = Flask(__name__)
load_dotenv()

# === EC2 전송 설정 ===
EC2_API_URL = "http://3.38.250.18:8000/ingest"

def send_to_ec2(user_data):
    try:
        res = requests.post(
            EC2_API_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(user_data, ensure_ascii=False).encode("utf-8")
        )
        if res.status_code == 200:
            print(f"[✓] EC2에 전송 성공: {user_data['uuid']}")
        else:
            print(f"[!] EC2 전송 실패 - 상태 코드: {res.status_code}")
    except Exception as e:
        print(f"[!] EC2 전송 중 오류 발생: {str(e)}")

# === AWS S3 설정 ===
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
REGION_NAME = os.getenv("AWS_REGION")

s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=REGION_NAME
)

# === 기본 페이지 라우팅 ===
@app.route("/")
def home():
    return render_template("app.html")

@app.route("/contactthanks")
def contactthanks():
    return render_template("contactthanks.html")

@app.route("/privacy")
def privacy():
    return render_template("privacy.html")

@app.route("/shortcodes")
def shortcodes():
    return render_template("shortcodes.html")

@app.route("/subscribe")
def subscribe():
    return render_template("subscribe.html")

@app.route("/video")
def video():
    return render_template("video.html")

@app.route("/download")
def download():
    return render_template("download.html")

@app.route("/index")
def index():
    return render_template("index.html")

# === 회원가입 ===
@app.route("/register")
def register_form():
    return render_template("register.html")

@app.route("/register", methods=["POST"])
def register():
    fields = [
        'USER_ID', 'PASSWORD', 'CONFIRM_PASSWORD', 'NAME', 'BIRTHDATE',
        'GENDER', 'EDU_NM', 'EDU_FNSH_SE', 'MARR_STTS', 'JOB_NM',
        'INCOME', 'HOUSE_INCOME', 'TRAVEL_TERM',
        'TRAVEL_LIKE_SIDO_1', 'TRAVEL_LIKE_SIDO_2', 'TRAVEL_LIKE_SIDO_3',
        'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4',
        'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
        'TRAVEL_MOTIVE_1', 'TRAVEL_MOTIVE_2',
        'FAMILY_MEMB', 'TRAVEL_NUM', 'TRAVEL_COMPANIONS_NUM'
    ]

    user_data = {field: request.form.get(field, "") for field in fields}
    user_data["uuid"] = str(uuid.uuid4())

    # 나이 계산
    birthdate_str = user_data.get("BIRTHDATE", "")
    try:
        birth_year = datetime.strptime(birthdate_str, "%Y-%m-%d").year
        age = datetime.now().year - birth_year
        age_group = (age // 10) * 10
        user_data["AGE_GRP"] = "90" if age_group >= 90 else str(max(10, age_group))
    except:
        user_data["AGE_GRP"] = ""

    # 전화번호 조합
    phone_prefix = request.form.get("phone_prefix", "")
    phone_middle = request.form.get("phone_middle", "")
    phone_last = request.form.get("phone_last", "")
    user_data["phone_number"] = f"{phone_prefix}{phone_middle}{phone_last}"

    # S3 업로드
    try:
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=f"travelers/{user_data['uuid']}.json",
            Body=json.dumps(user_data, ensure_ascii=False).encode('utf-8'),
            ContentType='application/json'
        )
        print(f"[✓] S3 저장 완료: {user_data['uuid']}")
    except Exception as e:
        print(f"[!] S3 저장 실패: {str(e)}")
        return f"S3 저장 실패: {str(e)}", 500

    # EC2 전송
    send_to_ec2(user_data)

    return render_template("app.html")

# === 앱 실행 ===
if __name__ == "__main__":
    app.run(debug=True)
