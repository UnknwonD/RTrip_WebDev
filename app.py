from dependency import *

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

load_dotenv(override=True)

app.secret_key = 'test'  # 세션을 위한 시크릿 키 설정

# AWS Setting 
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
REGION_NAME = os.getenv("AWS_REGION")

# EC2 Setting
EC2_PUBLIC_ADDR = os.getenv("EC2_PUBLIC_ADDR")


# S3 Client setting
s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=REGION_NAME
)

# === 중복 확인 함수 ===
def is_duplicate(field_name, value):
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

            if user_json.get(field_name) == value:
                return True
            if user_json.get(field_name) == value:
                return True
    except Exception as e:
        print(f"[!] 중복 확인 오류: {str(e)}")
    return False

def send_to_ec2(user_data):
    try:
        res = requests.post(
            EC2_PUBLIC_ADDR,
            headers={"Content-Type": "application/json"},
            data=json.dumps(user_data, ensure_ascii=False).encode("utf-8")
        )
        if res.status_code == 200:
            print(f"EC2에 전송 성공: {user_data['uuid']}")
        else:
            print(f"EC2 전송 실패 - 상태 코드: {res.status_code}")
    except Exception as e:
        print(f"[!] EC2 전송 중 오류 발생: {str(e)}")



# 첫 페이지, 성향 분석 데이터 받아옴, 로그인/회원가입으로 연결 (로그인의 경우에는 자동으로 성향 업데이트)
@app.route("/main", methods=["GET", "POST"])
def main():
    if request.method == "POST":
        travel_styles = {
            f"TRAVEL_STYL_{i}": request.form.get(f"TRAVEL_STYL_{i}", "4")
            for i in range(1, 9)
        }
        session["travel_styles"] = travel_styles

        return redirect(url_for("main_register"))

    return render_template("main.html")

# 회원가입 페이지
@app.route("/main_register", methods=["GET", "POST"])
def main_register():
    if request.method == "POST":
        user_id = request.form.get("USER_ID")

        if is_duplicate("USER_ID", user_id):
            return render_template("main_register.html", error="이미 사용 중인 아이디입니다.")  
      
        fields = [
            'USER_ID', 'PASSWORD', 'NAME', 'GENDER', 'BIRTHDATE',
            'TRAVEL_TERM', 'TRAVEL_NUM',
            'TRAVEL_LIKE_SIDO_1', 'TRAVEL_LIKE_SIDO_2', 'TRAVEL_LIKE_SIDO_3',
            'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4',
            'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
            'TRAVEL_MOTIVE_1', 'TRAVEL_MOTIVE_2',
        ]

        user_data = {field: request.form.get(field, "") for field in fields}

        travel_styles = session.get("travel_styles", {})
        user_data.update(travel_styles)

        user_data["uuid"] = str(uuid.uuid4())

        birthdate_str = user_data.get("BIRTHDATE", "")
        try:
            birth_year = datetime.strptime(birthdate_str, "%Y-%m-%d").year
            age = datetime.now().year - birth_year
            age_group = (age // 10) * 10
            user_data["AGE_GRP"] = "90" if age_group >= 90 else str(max(10, age_group))
        except:
            user_data["AGE_GRP"] = ""

        try:
            s3.put_object(
                Bucket=BUCKET_NAME,
                Key=f"users/{user_data['uuid']}.json",
                Body=json.dumps(user_data, ensure_ascii=False).encode('utf-8'),
                ContentType='application/json'
            )
            print(f"S3 저장 완료: {user_data['uuid']}")
        except Exception as e:
            print(f"S3 저장 실패: {str(e)}")
            return f"S3 저장 실패: {str(e)}", 500

        return redirect(url_for("main_recommended"))
    
    return render_template("main_register.html")

# 메인 페이지
@app.route("/", methods=["GET", "POST"])
def main_recommended():
    # 테스트용 임시 travel_plans
    travel_plans = [
        {
            "title": "제주 힐링 3일 코스",
            "description": "바다와 자연을 즐기는 여정",
            "main_image_url": "https://via.placeholder.com/800x300/89CFF0/ffffff?text=Jeju+Trip",
            "route": [
                {"name": "성산일출봉", "description": "일출로 하루를 시작"},
                {"name": "우도", "description": "자전거로 한 바퀴"},
                {"name": "용두암", "description": "돌하르방과 사진 한 컷"}
            ]
        },
        {
            "title": "서울 도심 속 하루 코스",
            "description": "도시의 매력을 느끼는 코스",
            "main_image_url": "https://via.placeholder.com/800x300/FFB6C1/ffffff?text=Seoul+Trip",
            "route": [
                {"name": "경복궁", "description": "한복 입고 투어"},
                {"name": "북촌한옥마을", "description": "전통과 현대의 조화"},
                {"name": "한강공원", "description": "야경 보며 피크닉"}
            ]
        }
    ]
    user_json = None
    if request.method == "POST":
        travel_input = request.form.to_dict()

        raw_user = get_user_info(session["username"])
        if raw_user:
            exclude_fields = {"BIRTHDATE", "uuid", 'phone_number', "PASSWORD", "CONFIRM_PASSWORD"} # user 정보에서 필요 없는 정보들 입력
            user_json = {k: v for k, v in raw_user.items() if k not in exclude_fields}

        user_input, travel_input = preprocess_gnn(user_json, travel_input)
        
        results = recommend_from_input(model, user_input, travel_input, base_data, visit_area_id_map)
        session["results"] = results
        # return redirect(url_for("main_recommended"))
        return redirect(url_for("recommend_result"))
    else:
        return render_template(
            "main_recommended.html", # 이대호 recommended_result인가가
            travel_plans=travel_plans,
            purpose_options=purpose_options,
            movement_options=movement_options,
            whowith_options=whowith_options,
            user_feature_keys=user_feature_keys,
            user_info=user_json
        )
    # return render_template("main_recommended.html")


# 로그인
@app.route("/login", methods=["POST"])
def login():
    input_id = request.form.get("USER_ID")
    input_pw = request.form.get("PASSWORD")
    
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
            except json.JSONDecodeError as e:
                continue

            if user_json.get("USER_ID") == input_id and user_json.get("PASSWORD") == input_pw:
                session["username"] = input_id

                travel_styles = session.get("travel_styles")

                if travel_styles:
                    user_json.update(travel_styles)

                uuid_key = user_json.get("uuid", str(uuid.uuid4()))

                s3.put_object(
                    Bucket=BUCKET_NAME,
                    Key=f"users/{uuid_key}.json",
                    Body=json.dumps(user_json, ensure_ascii=False).encode("utf-8"),
                    ContentType="application/json"
                )
                return redirect(url_for("main_recommended"))

        return render_template("main_recommended.html", error="아이디 또는 비밀번호가 잘못되었습니다.")
    
    except Exception as e:
        return f"S3 조회 오류: {str(e)}", 500
    
# logout
@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("main_recommended"))

@app.route("/check_duplicate")
def check_duplicate():
    field = request.args.get("field")
    value = request.args.get("value")
    return jsonify({"duplicate": is_duplicate(field, value)})


# 수정 해야함
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

# 추천 결과 페이지
# @app.route("/recommend_result")
# def recommend_result():
#     if not results:
#         return redirect(url_for("main_recommended"))
#     results = session.pop("results", None)
#     return render_template("recommend_result.html", results=results)

# 이미지 불러오기 위한 전역변수
# @app.context_processor
# def inject_images():
#     if "username" in session:
#         images = get_user_recommended_images_and_areas(session["username"])
#         print(images)
#     else:
#         images = get_random_images_from_rds()

#     return dict(images=images)

# @app.route("/register", methods=["POST"])
# def register():
#     user_id = request.form.get("USER_ID")
#     phone_number = f"{request.form.get('phone_prefix')}{request.form.get('phone_middle')}{request.form.get('phone_last')}"

#     if is_duplicate("USER_ID", user_id):
#         return render_template("register.html", error="이미 사용 중인 아이디입니다.")
#     if is_duplicate("phone_number", phone_number):
#         return render_template("register.html", error="이미 등록된 전화번호입니다.")

#     fields = [
#         'USER_ID', 'PASSWORD', 'CONFIRM_PASSWORD', 'NAME', 'BIRTHDATE',
#         'GENDER', 'EDU_NM', 'EDU_FNSH_SE', 'MARR_STTS', 'JOB_NM',
#         'INCOME', 'HOUSE_INCOME', 'TRAVEL_TERM',
#         'TRAVEL_LIKE_SIDO_1', 'TRAVEL_LIKE_SIDO_2', 'TRAVEL_LIKE_SIDO_3',
#         'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4',
#         'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
#         'TRAVEL_MOTIVE_1', 'TRAVEL_MOTIVE_2',
#         'FAMILY_MEMB', 'TRAVEL_NUM', 'TRAVEL_COMPANIONS_NUM'
#     ]

#     user_data = {field: request.form.get(field, "") for field in fields}
#     user_data["uuid"] = str(uuid.uuid4())

#     birthdate_str = user_data.get("BIRTHDATE", "")
#     try:
#         birth_year = datetime.strptime(birthdate_str, "%Y-%m-%d").year
#         age = datetime.now().year - birth_year
#         age_group = (age // 10) * 10
#         user_data["AGE_GRP"] = "90" if age_group >= 90 else str(max(10, age_group))
#     except:
#         user_data["AGE_GRP"] = ""

#     user_data["phone_number"] = phone_number

#     try:
#         s3.put_object(
#             Bucket=BUCKET_NAME,
#             Key=f"users/{user_data['uuid']}.json",
#             Body=json.dumps(user_data, ensure_ascii=False).encode('utf-8'),
#             ContentType='application/json'
#         )
#         print(f"[✓] S3 저장 완료: {user_data['uuid']}")
#     except Exception as e:
#         print(f"[!] S3 저장 실패: {str(e)}")
#         return f"S3 저장 실패: {str(e)}", 500

#     return redirect(url_for("home"))

# 추천 페이지
# @app.route("/recommended", methods=["GET", "POST"])
# def recommended():
#     user_json = None
#     if request.method == "POST":
#         travel_input = request.form.to_dict()

#         raw_user = get_user_info(session["username"])
#         if raw_user:
#             exclude_fields = {"BIRTHDATE", "uuid", 'phone_number', "PASSWORD", "CONFIRM_PASSWORD"} # user 정보에서 필요 없는 정보들 입력
#             user_json = {k: v for k, v in raw_user.items() if k not in exclude_fields}

#         user_input, travel_input = preprocess_gnn(user_json, travel_input)
        
#         results = recommend_from_input(model, user_input, travel_input, base_data, visit_area_id_map)
#         print(results)
#         return redirect(url_for("recommend_result"))
#     else:
#         return render_template(
#             "recommended.html",
#             purpose_options=purpose_options,
#             movement_options=movement_options,
#             whowith_options=whowith_options,
#             user_feature_keys=user_feature_keys,
#             user_info=user_json
#         )

# 회원가입 페이지
# @app.route("/register")
# def register_form():
#     return render_template("register.html")
    
# @app.route("/contactthanks")
# def contactthanks():
#     return render_template("contactthanks.html")

# @app.route("/privacy")
# def privacy():
#     return render_template("privacy.html")

# @app.route("/shortcodes")
# def shortcodes():
#     return render_template("shortcodes.html")

# @app.route("/subscribe")
# def subscribe():
#     return render_template("subscribe.html")

# # @app.route("/video")
# def video():
#     return render_template("video.html")

# @app.route("/download")
# def download():
#     return render_template("download.html")

# @app.route("/index")
# def index():
#     return render_template("index.html")





if __name__ == "__main__":
    app.run(debug=True)