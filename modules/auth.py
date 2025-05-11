from flask import session, redirect, url_for
from modules.s3_utils import list_s3_objects, get_json_from_s3

def authenticate(user_id, password):
    objects = list_s3_objects("travelers/")
    for obj in objects:
        user_json = get_json_from_s3(obj['Key'])
        if user_json.get("USER_ID") == user_id and user_json.get("PASSWORD") == password:
            session["username"] = user_id
            return True
    return False

def logout_user():
    session.pop("username", None)
    return redirect(url_for("home"))
