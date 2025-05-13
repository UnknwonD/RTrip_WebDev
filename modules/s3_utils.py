import boto3
import json
import botocore
import os
from dotenv import load_dotenv

load_dotenv(override=True)

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
REGION_NAME = os.getenv("AWS_REGION")

s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=REGION_NAME,
    config=botocore.client.Config(signature_version='s3v4')
)

def get_user_from_s3(username):
    try:
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix="users/")
        for obj in response.get('Contents', []):
            key = obj['Key']
            if not key.endswith('.json'):
                continue

            file_obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
            body = file_obj['Body'].read().decode('utf-8').strip()
            if not body:
                print(f"[!] 빈 파일: {key}")
                continue

            try:
                user_json = json.loads(body)
            except json.JSONDecodeError as e:
                print(f"[!] JSON 파싱 오류: {key} → {e}")
                continue

            if user_json.get("USER_ID") == username:
                return user_json
        return None
    except Exception as e:
        raise RuntimeError(f"S3 조회 오류: {str(e)}")

def update_user_in_s3(username, updated_data):
    try:
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix="users/")
        for obj in response.get('Contents', []):
            key = obj['Key']
            file_obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
            user_json = json.loads(file_obj['Body'].read().decode('utf-8'))

            if user_json.get("USER_ID") == username:
                user_json.update(updated_data)
                s3.put_object(
                    Bucket=BUCKET_NAME,
                    Key=key,
                    Body=json.dumps(user_json, ensure_ascii=False).encode('utf-8'),
                    ContentType='application/json'
                )
                return True
        return False
    except Exception as e:
        raise RuntimeError(f"S3 저장 오류: {str(e)}")


