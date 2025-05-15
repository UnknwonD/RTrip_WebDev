from flask import session, redirect, url_for
from modules.s3_utils import list_s3_objects, get_json_from_s3

def authenticate(user_id, password): # 사용자가 입력한 ID, PW
    objects = list_s3_objects("users/") # 경로에 있는 객체 리스트 불러오기 ( JSON )
    for obj in objects:
        user_json = get_json_from_s3(obj['Key']) 
        if user_json.get("USER_ID") == user_id and user_json.get("PASSWORD") == password: # 입력 ID,PW와 S3의 ID,PW값 비교교
            session["username"] = user_id # session에 유저 아이디 저장장
            return True
    return False

# log out
def logout_user():
    session.pop("username", None) # session에서 아이디 제거 
    return redirect(url_for("home"))
