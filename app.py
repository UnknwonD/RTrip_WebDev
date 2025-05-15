from dependency import *

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


# 이미지 정보 같이 불러오는 코드 추가할 것 (장소명, 주소, 방문 횟수 ...)
def get_s3_signed_urls(reverse = False):
    s3 = boto3.client('s3',
                      aws_access_key_id=AWS_ACCESS_KEY,
                      aws_secret_access_key=AWS_SECRET_KEY,
                      region_name= REGION_NAME,
                      config=botocore.client.Config(signature_version='s3v4')
                    )

    bucket = BUCKET_NAME
    prefix = 'data/resized_image/E/'

    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    all_keys = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.jpg')]
    all_keys = sorted(all_keys, reverse=reverse)[:10]  # 상위 10개
    
    signed_urls = [
        s3.generate_presigned_url('get_object', Params={'Bucket': bucket, 'Key': key}, ExpiresIn=3600)
        for key in all_keys
    ]
    return signed_urls

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

# 기본 페이지
@app.route("/")
def home():
    return render_template("app.html")

# 로그인 팝업
@app.route("/login", methods=["POST"])
def login():
    input_id = request.form.get("USER_ID")
    input_pw = request.form.get("PASSWORD")
    
    try:
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix="users/")
        for obj in response.get('Contents', []):
            key = obj['Key']
            if not key.endswith('.json'):
                continue  # 폴더 객체 등은 무시

            file_obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
            body = file_obj['Body'].read().decode('utf-8').strip()

            if not body:
                print(f"[!] 빈 파일: {key}")
                continue

            try:
                user_json = json.loads(body)
            except json.JSONDecodeError as e:
                print(f"[!] JSON 파싱 실패: {key} → {e}")
                continue

            if user_json.get("USER_ID") == input_id and user_json.get("PASSWORD") == input_pw:
                session["username"] = input_id
                return redirect(url_for("home"))
        return render_template("app.html", error="아이디 또는 비밀번호가 잘못되었습니다.")
    except Exception as e:
        return f"S3 조회 오류: {str(e)}", 500

@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("home"))

@app.route("/register", methods=["POST"])
def register():
    user_id = request.form.get("USER_ID")
    phone_number = f"{request.form.get('phone_prefix')}{request.form.get('phone_middle')}{request.form.get('phone_last')}"

    if is_duplicate("USER_ID", user_id):
        return render_template("register.html", error="이미 사용 중인 아이디입니다.")
    if is_duplicate("phone_number", phone_number):
        return render_template("register.html", error="이미 등록된 전화번호입니다.")

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

    birthdate_str = user_data.get("BIRTHDATE", "")
    try:
        birth_year = datetime.strptime(birthdate_str, "%Y-%m-%d").year
        age = datetime.now().year - birth_year
        age_group = (age // 10) * 10
        user_data["AGE_GRP"] = "90" if age_group >= 90 else str(max(10, age_group))
    except:
        user_data["AGE_GRP"] = ""

    user_data["phone_number"] = phone_number

    try:
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=f"users/{user_data['uuid']}.json",
            Body=json.dumps(user_data, ensure_ascii=False).encode('utf-8'),
            ContentType='application/json'
        )
        print(f"[✓] S3 저장 완료: {user_data['uuid']}")
    except Exception as e:
        print(f"[!] S3 저장 실패: {str(e)}")
        return f"S3 저장 실패: {str(e)}", 500

    return redirect(url_for("home"))

@app.route("/check_duplicate")
def check_duplicate():
    field = request.args.get("field")
    value = request.args.get("value")
    return jsonify({"duplicate": is_duplicate(field, value)})

@app.route("/mypage", methods=["GET", "POST"])
def mypage():
    if "username" not in session:
        return redirect(url_for("home"))

    username = session["username"]

    if request.method == "GET":
        try:
            user_json = get_user_info(username)
            if user_json:
                return render_template("mypage.html", user=user_json)
            return "사용자 정보를 찾을 수 없습니다.", 404
        except RuntimeError as e:
            return str(e), 500

    elif request.method == "POST":
        update_fields = ['NAME', 'GENDER', 'JOB_NM', 'INCOME', 'HOUSE_INCOME', 'TRAVEL_TERM']
        updated_data = {field: request.form.get(field, "") for field in update_fields}

        try:
            success = update_user_info(username, updated_data)
            if success:
                flash("회원 정보가 성공적으로 수정되었습니다.")
                return redirect(url_for("home"))
            return "수정 대상 사용자를 찾을 수 없습니다.", 404
        except RuntimeError as e:
            return str(e), 500
# 추천 결과 페이지
@app.route("/recommend_result")
def recommend_result():
    results = session.pop("results", None)
    return render_template("recommend_result.html", results=results)

# 추천 페이지
@app.route("/recommended", methods=["GET", "POST"])
def recommended():
    user_json = None
    if request.method == "POST":
        travel_input = request.form.to_dict()

        raw_user = get_user_info(session["username"])
        if raw_user:
            exclude_fields = {"BIRTHDATE", "uuid", 'phone_number', "PASSWORD", "CONFIRM_PASSWORD"} # user 정보에서 필요 없는 정보들 입력
            user_json = {k: v for k, v in raw_user.items() if k not in exclude_fields}

        user_input, travel_input = preprocess_gnn(user_json, travel_input)


        # user_input = get_user_feature_from_session(session["username"])
        # travel_input = preprocess_travel_form(form)

        # model = load_model('models/best_model.pt', metadata)
        # base_data = torch.load("data/base_data.pt")
        # visit_area_id_map = load_json("data/visit_area_id_map.json")

        # results = recommend_from_input(model, user_input, travel_input, base_data, visit_area_id_map)

        return redirect(url_for("recommend_result"))
    else:
        return render_template(
            "recommended.html",
            purpose_options=purpose_options,
            movement_options=movement_options,
            whowith_options=whowith_options,
            user_feature_keys=user_feature_keys,
            user_info=user_json
        )


# 지도 페이지
@app.route("/map")
def map():
    return render_template("map.html")

# 회원가입 페이지
@app.route("/register")
def register_form():
    return render_template("register.html")

# XAI Page
@app.route("/xai")
def xai():
    return render_template("xai.html")

# 이미지 불러오기 위한 전역변수
@app.context_processor
def inject_images():
    if "username" in session:
        images = get_user_recommended_images_and_areas(session["username"])
        print(images)
    else:
        images = get_s3_signed_urls()  # 랜덤 이미지
    return dict(images=images)

    
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