from dependency import *
from config import s3, BUCKET_NAME

############## 초기 모델 로드 ##################
with open("./pickle/metadata_lite.pkl", "rb") as f:
    metadata = pickle.load(f)
with open("./pickle/visit_area_id_map.pkl", "rb") as f:
    visit_area_id_map = pickle.load(f)
    
with open("./pickle/dataset.pkl", "rb") as f:
    base_data = pickle.load(f)

## 삐삐꼬는 GNN용
# with open("./pickle/PPK/metadata_lite.pkl", "rb") as f:
#     metadata = pickle.load(f)
# with open("./pickle/PPK/visit_area_id_map.pkl", "rb") as f:
#     visit_area_id_map = pickle.load(f)
# with open("./pickle/PPK/dataset.pkl", "rb") as f:
#     base_data = pickle.load(f)

# model = PpiKkoTwistGNN(
#         metadata=metadata,
#         user_input_dim=25,
#         travel_input_dim=29,
#         hidden_dim=128,
#         num_layers=8
#     )


model = LiteTwistGNN(
        metadata=metadata,
        user_input_dim=25,
        travel_input_dim=29,
        hidden_dim=128,
        num_layers=8
    )
    
state = torch.load('./pickle/ppk_lite.pt', map_location='cpu')
model.load_state_dict(state)
model.eval()
#############################################

app = Flask(__name__)

app.secret_key = 'test'


# app.py
@app.route("/main", methods=["GET", "POST"])
def main():
    if request.method == "POST":
        session["travel_styles"] = extract_travel_styles(request.form)
        return redirect(url_for("main_register"))

    travel_styles = session.get("travel_styles")
    images = find_nearest_users(travel_styles, k=5) if travel_styles else []

    return render_template("main.html", images=images)

# 회원가입 페이지
@app.route("/main_register", methods=["GET", "POST"])
def main_register():
    if request.method == "POST":
        user_id = request.form.get("USER_ID")
        if is_duplicate("USER_ID", user_id):
            return render_template("main_register.html", error="이미 사용 중인 아이디입니다.")

        travel_styles = session.get("travel_styles", [])
        user_data = extract_user_data(request.form, travel_styles)

        if not save_user_to_s3(s3, BUCKET_NAME, user_data):
            return "S3 저장 실패", 500

        session["username"] = user_data["USER_ID"]  # ✅ 자동 로그인 처리 추가

        return redirect(url_for("main_recommended"))

    return render_template("main_register.html")




# 메인 페이지
@app.route("/", methods=["GET", "POST"])
def main_recommended():
    user_json = None    # user 정보
    travel_plans_data = [] # GNN 결과 여행 정보 
    
    if request.method == "POST":
        
        travel_input = request.form.to_dict()           # 설문 받은 유저 정보
        raw_user = get_user_info(session["username"])   # 유저 S3 기본 데이터 정보 

        # 필요 없는 정보 제거
        user_json = {
            k: v for k, v in raw_user.items() 
            if k not in {"BIRTHDATE", "uuid", "phone_number", "PASSWORD", "CONFIRM_PASSWORD"}
        }
####################################test########################################################################

    dummy_ids = [2308260002, 2308260003, 2308260005, 2308260006, 2308260007, 2308260008]


    route_lists = [dummy_ids[i:i+3] for i in range(0, len(dummy_ids), 3)]
    travel_plan_data = travel_plans(route_lists)


####################################test########################################################################
        
        # gnn 학습
        # user_input, travel_input = preprocess_gnn(user_json, travel_input)

        # 장소 추론 (output : list 형태의 visit_area_id )
        # visit_area_ids = recommend_from_input(model, user_input, travel_input, base_data, visit_area_id_map)

        # visit_area_ids 형태 가공 필요 -> 이중 배열 [[],[],...]
        # 추천 결과 가공
        # travel_plans = travel_plans(visit_area_ids)
    
    # Get 요청일 때 추천 데이터 제공 (입력 받기 전 예시 데이터)
    # else:
    #     travel_plans = default_travel_plans()

    return render_template(
        "main_recommended.html", 
        purpose_options=purpose_options,
        movement_options=movement_options,
        whowith_options=whowith_options,
        user_feature_keys=user_feature_keys,
        user_info=user_json,
        travel_plans = travel_plan_data
    )
    
# 로그인
@app.route("/login", methods=["POST"])
def login():
    input_id = request.form.get("USER_ID")
    input_pw = request.form.get("PASSWORD")

    try:
        user_json, s3_key = find_user_by_credentials(input_id, input_pw)
        if not user_json:
            return render_template("main_recommended.html", error="아이디 또는 비밀번호가 잘못되었습니다.")

        session["username"] = input_id
        travel_styles = session.get("travel_styles")
        handle_login_success(user_json, travel_styles)

        return redirect(url_for("main_recommended"))

    except RuntimeError as e:
        return str(e), 500

# logout
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("main"))

@app.route("/check_duplicate")
def check_duplicate():
    field = request.args.get("field")
    value = request.args.get("value")
    return jsonify({"duplicate": is_duplicate(field, value)})


@app.route("/preview_images")
def preview_images():
    travel_styles = session.get("travel_styles")
    if not travel_styles:
        return jsonify({"error": "No style data"}), 400

    photos = find_nearest_users(travel_styles)
    image_data = get_presigned_image_urls(photos)

    return render_template("main.html", images=image_data)

@app.route("/analyze_styles", methods=["POST"])
def analyze_styles():
    data = request.get_json()
    scores = data.get("scores", [])
    session["travel_styles"] = scores

    images = find_nearest_users(scores) 

    return jsonify({
        "images": images
    })

# 수정 해야함 -> 기존 정보 수정이 아닌 여행 동선 저장장
# @app.route("/mypage", methods=["GET", "POST"])
# def mypage():
#     if "username" not in session:
#         return redirect(url_for("home"))

#     username = session["username"]

#     if request.method == "GET":
#         try:
#             user_json = get_user_info(username)
            
#             if user_json:
#                 return render_template("mypage.html", user=user_json, today=datetime.today().strftime('%Y-%m-%d'))
#             return "사용자 정보를 찾을 수 없습니다.", 404
#         except RuntimeError as e:
#             return str(e), 500

#     elif request.method == "POST":
#         update_fields = [
#             'NAME', 'GENDER', 'BIRTHDATE', 'phone_number',
#             'EDU_NM', 'EDU_FNSH_SE', 'MARR_STTS', 'FAMILY_MEMB',
#             'JOB_NM', 'INCOME', 'HOUSE_INCOME', 'TRAVEL_TERM', 'TRAVEL_NUM',
#             'TRAVEL_LIKE_SIDO_1', 'TRAVEL_LIKE_SIDO_2', 'TRAVEL_LIKE_SIDO_3',
#             'TRAVEL_MOTIVE_1', 'TRAVEL_MOTIVE_2', 'TRAVEL_COMPANIONS_NUM'
#         ] + [f'TRAVEL_STYL_{i}' for i in range(1, 9)]

#         updated_data = {field: request.form.get(field, "") for field in update_fields}
#         print("[ 업데이트 데이터]", updated_data)
#         try:
#             success = update_user_info(username, updated_data)
            
#             if success:
#                 flash("회원 정보가 성공적으로 수정되었습니다.")
#                 return redirect(url_for("home"))
#             return "수정 대상 사용자를 찾을 수 없습니다.", 404
#         except RuntimeError as e:
#             return str(e), 500


if __name__ == "__main__":
    app.run(debug=True)

if __name__ == "__main__":
    style_vec = [5, 5, 3, 2, 4, 5, 3, 6]  # 테스트용 input
    ids = find_nearest_users(style_vec, k=5)
    print("유사한 유저 ID:", ids)
    