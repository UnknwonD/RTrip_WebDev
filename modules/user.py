import uuid
from datetime import datetime
from modules.s3_utils import put_json_to_s3, list_s3_objects, get_json_from_s3

# 생년월일로 나이대 계산산
def calculate_age_group(birthdate_str):
    try:
        birth_year = datetime.strptime(birthdate_str, "%Y-%m-%d").year
        age = datetime.now().year - birth_year
        age_group = (age // 10) * 10
        return "90" if age_group >= 90 else str(max(10, age_group))
    except:
        return ""

#
def is_duplicate(field_name, value):
    objects = list_s3_objects("users/")
    for obj in objects:
        user_json = get_json_from_s3(obj['Key'])
        if user_json.get(field_name) == value:
            return True
    return False


def register_user(form_data):
    user_data = {field: form_data.get(field, "") for field in form_data.keys()}
    user_data["uuid"] = str(uuid.uuid4())
    user_data["AGE_GRP"] = calculate_age_group(user_data.get("BIRTHDATE", ""))
    put_json_to_s3(f"users/{user_data['uuid']}.json", user_data)
    return user_data["uuid"]

def update_user_info(username, updated_data):
    objects = list_s3_objects("users/")
    for obj in objects:
        key = obj['Key']
        try:
            user_json = get_json_from_s3(key)
        except Exception as e:
            print(f"[!] JSON 파싱 실패: {key} → {e}")
            continue
        
        if user_json.get("USER_ID") == username:
            if 'BIRTHDATE' in updated_data:
                updated_data["AGE_GRP"] = calculate_age_group(updated_data["BIRTHDATE"])

            for k, v in updated_data.items():
                if k in user_json:
                    user_json[k] = v

            put_json_to_s3(key, user_json)
            return True
    return False