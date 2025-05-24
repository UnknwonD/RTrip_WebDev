from flask import session
from modules.s3_utils import list_s3_objects, get_json_from_s3
import json
from config import s3, BUCKET_NAME

def authenticate(user_id, password):
    objects = list_s3_objects("users/")
    
    for obj in objects:
        user_json = get_json_from_s3(obj['Key'])
        if not user_json:
            continue  # JSON 파싱 실패나 빈 데이터 건너뜀

        if user_json.get("USER_ID") == user_id and user_json.get("PASSWORD") == password:
            session["username"] = user_id
            return True
            
    return False

def find_user_by_credentials(user_id, password):
    try:
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix="users/")
        for obj in response.get('Contents', []):
            key = obj['Key']
            if not key.endswith('.json'):
                continue

            file_obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
            body = file_obj['Body'].read().decode('utf-8').strip()
            if not body:
                continue

            try:
                user_json = json.loads(body)
            except json.JSONDecodeError:
                continue

            if user_json.get("USER_ID") == user_id and user_json.get("PASSWORD") == password:
                return user_json, key
        return None, None
    except Exception as e:
        raise RuntimeError(f"S3 조회 오류: {str(e)}")


def handle_login_success(user_json, session_styles):
    if session_styles:
        for i, val in enumerate(session_styles):
            user_json[f"TRAVEL_STYL_{i+1}"] = val

    uuid_key = user_json.get("uuid") or user_json.get("USER_ID")
    if not uuid_key:
        uuid_key = "missing-uuid"  # fallback

    user_json["uuid"] = uuid_key
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=f"users/{uuid_key}.json",
        Body=json.dumps(user_json, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json"
    )

    return True