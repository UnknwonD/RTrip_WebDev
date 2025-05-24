import uuid
from datetime import datetime
import json
from modules.s3_utils import (
    put_json_to_s3,
    list_s3_objects,
    get_json_from_s3
)


# 중복 필드 검사 (예: USER_ID, 전화번호 등)
def is_duplicate(field_name, value):
    objects = list_s3_objects("users/")
    for obj in objects:
        user_json = get_json_from_s3(obj['Key'])
        if not user_json:
            continue
        if user_json.get(field_name) == value:
            return True
    return False


# 사용자 등록 (회원가입)
def register_user(form_data):
    user_data = {field: form_data.get(field, "") for field in form_data.keys()}
    user_data["uuid"] = str(uuid.uuid4())
    user_data["AGE_GRP"] = calculate_age_group(user_data.get("BIRTHDATE", ""))
    put_json_to_s3(f"users/{user_data['uuid']}.json", user_data)
    return user_data["uuid"]

# 사용자 정보 업데이트 (마이페이지 등)
def update_user_info(username, updated_data):
    objects = list_s3_objects("users/")
    for obj in objects:
        key = obj['Key']
        user_json = get_json_from_s3(key)
        if not user_json:
            continue

        if user_json.get("USER_ID") == username:
            if "BIRTHDATE" in updated_data:
                updated_data["AGE_GRP"] = calculate_age_group(updated_data["BIRTHDATE"])

            user_json.update({
                k: v for k, v in updated_data.items() if k in user_json
            })

            put_json_to_s3(key, user_json)
            return True
    return False


# user 정보 반환 ( input = ID output = dict )
def get_user_info(username):
    objects = list_s3_objects("users/")
    json_objects = [obj for obj in objects if obj['Key'].endswith('.json')] # .json 파일 필터링링

    for obj in json_objects:
        key = obj['Key']
        try:
            user_json = get_json_from_s3(key)   # dict
        except Exception as e:
            print(f"[!] JSON 파싱 실패: {key} → {e}")
            continue
        if user_json.get("USER_ID") == username:    # USER_ID가 일치하면 해당 유저 정보 반환 
            return user_json                        # dict
    return None


def extract_user_data(form, session_styles):
    fields = [
        'USER_ID', 'PASSWORD', 'NAME', 'GENDER', 'BIRTHDATE',
        'TRAVEL_TERM', 'TRAVEL_NUM',
        'TRAVEL_LIKE_SIDO_1', 'TRAVEL_LIKE_SIDO_2', 'TRAVEL_LIKE_SIDO_3',
        'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4',
        'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
        'TRAVEL_MOTIVE_1', 'TRAVEL_MOTIVE_2',
    ]
    user_data = {field: form.get(field, "") for field in fields}

    if session_styles and isinstance(session_styles, list):
        for i, val in enumerate(session_styles):
            user_data[f'TRAVEL_STYL_{i+1}'] = val

    user_data['uuid'] = str(uuid.uuid4())
    user_data['AGE_GRP'] = calculate_age_group(user_data.get('BIRTHDATE', ''))

    return user_data


def calculate_age_group(birthdate_str):
    try:
        birth_year = datetime.strptime(birthdate_str, "%Y-%m-%d").year
        age = datetime.now().year - birth_year
        age_group = (age // 10) * 10
        return "90" if age_group >= 90 else str(max(10, age_group))
    except:
        return ""


def save_user_to_s3(s3, BUCKET_NAME, user_data):
    try:
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=f"users/{user_data['uuid']}.json",
            Body=json.dumps(user_data, ensure_ascii=False).encode('utf-8'),
            ContentType='application/json'
        )
        print(f"S3 저장 완료: {user_data['uuid']}")
        return True
    except Exception as e:
        print(f"S3 저장 실패: {str(e)}")
        return False