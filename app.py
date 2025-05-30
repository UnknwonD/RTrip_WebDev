import os
os.environ['PYTHONAUTOFLUSH'] = '1'
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module=".*resource_tracker.*")

from dependency import *
from config import s3, BUCKET_NAME

app = Flask(__name__)
app.secret_key = 'test'
recommendation_storage = {}

# app.py
@app.route("/main", methods=["GET", "POST"])
def main():
    if request.method == "POST":
        session["travel_styles"] = extract_travel_styles(request.form)
        return redirect(url_for("main_register"))

    travel_styles = session.get("travel_styles")
    images = find_nearest_users(travel_styles, k=5) if travel_styles else []

    show_step = request.args.get("show_step", default=None)

    return render_template("main.html", images=images, travel_styles=travel_styles,show_step=show_step)

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

        session["username"] = user_data["USER_ID"]  

        return redirect(url_for("main_recommended"))

    return render_template("main_register.html")

def serialize_routes(routes):
    new_serializable_routes = {}
    for day, spots in routes.items():
        new_spots = []
        for spot in spots:
            # coords를 list로 변환하고 x, y 키 추가
            if isinstance(spot.get('coords'), np.ndarray):
                spot['coords'] = spot['coords'].tolist()
            if 'coords' in spot and isinstance(spot['coords'], list) and len(spot['coords']) == 2:
                spot['x'], spot['y'] = spot['coords'][0], spot['coords'][1]
            
            # 모든 numpy.int64 같은 정수형을 int로 변환
            for k, v in spot.items():
                if isinstance(v, (np.integer, np.int64)):
                    spot[k] = int(v)
            new_spots.append(spot)
        new_serializable_routes[day] = new_spots
    return new_serializable_routes


def filter_disliked_places(new_routes, dislike_set):
    """
    새로운 추천 경로에서 dislike_set에 포함된 장소를 제거
    """
    filtered_routes = {}
    for day, spots in new_routes.items():
        filtered_spots = []
        for spot in spots:
            if str(spot['id']) not in dislike_set:
                filtered_spots.append(spot)
        filtered_routes[day] = filtered_spots
    return filtered_routes


@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.json
    plan_id = session.get('current_plan_id')

    stored_data = recommendation_storage.get(plan_id)
    if not stored_data:
        return jsonify({'success': False, 'message': '데이터 없음'})

    # 세션 dislike_set이 없으면 초기화
    if 'dislike_set' not in session:
        session['dislike_set'] = []

    # 이번에 새로 싫어요한 장소 누적
    for dislike_id in data['disliked_ids']:
        dislike_id = str(dislike_id)
        if dislike_id not in session['dislike_set']:
            session['dislike_set'].append(dislike_id)

    # 새로운 추천 생성
    new_routes = feedback_usage_user(
        recommender=stored_data['recommender'],
        origin_routes=stored_data['route'],
        unique_recommendations=stored_data['unique_recommendations'],
        travel_context_tensor=stored_data['travel_context_tensor'],
        disliked_ids=data['disliked_ids']
    )
    # dislike_set 기반으로 다시 필터링
    dislike_set = set(session['dislike_set'])
    dislike_set = dislike_set.union(stored_data['unique_recommendations'])
    filtered_routes = filter_disliked_places(new_routes, dislike_set)

    # 경로 업데이트 및 직렬화
    stored_data['route'] = filtered_routes
    new_routes_serializable = serialize_routes(filtered_routes)

    return jsonify({'success': True, 'newRoutes': new_routes_serializable})



def update_recommendations(route, recommender, unique_recommendations, travel_context_tensor):
    try:
        plan_id = str(uuid.uuid4())[:8]
        
        # 메모리에 저장
        recommendation_storage[plan_id] = {
            'route': route,
            'recommender': recommender,
            'unique_recommendations': unique_recommendations,
            'travel_context_tensor': travel_context_tensor,
            'timestamp': datetime.now()
        }
        
        # 세션에는 ID만 저장
        session['current_plan_id'] = plan_id
        
        return plan_id
    except Exception as e:
        print("저장 중 오류 발생:", e)
        return None

def get_recommendations(plan_id=None):
    if plan_id is None:
        plan_id = session.get('current_plan_id')
    
    return recommendation_storage.get(plan_id)


# 메인 페이지
@app.route("/", methods=["GET", "POST"])
def main_recommended():
    user_json = None  # 사용자 정보
    travel_plan_list = []

    if request.method == "POST":
        travel_input = request.form.to_dict()
        
        try:
          route, recommender, unique_recommendations, travel_context_tensor = main_optimized_test(travel_input)

          plan_id = update_recommendations(route, recommender, unique_recommendations, travel_context_tensor)

          dummy_ids = [[d['id'] for d in v] for k, v in route.items()] # 날짜별로, 순서대로 인덱스 갖고 있음
          print(f"[DEBUG] 🤖 GNN 추론 결과: {dummy_ids[:5]}")

          travel_plan_list = travel_plans_with_debug(dummy_ids, travel_input['date_range'])
          print(f"[DEBUG] 🤖 travel_plan_list 값 확인: {travel_plan_list[:5]}")

          travel_plan_list = fill_missing_coords_with_kakao(travel_plan_list)

        except Exception as e:
             print(f"[DEBUG] ❌ GNN 추론 실패, 기본 계획 사용: {e}")
             travel_plan_list = default_travel_plans()

    else:
        travel_plan_list = default_travel_plans()
        # dummy_ids = {1: [7858, 1869, 9863], 2: [9858, 9855, 8691, 8032], 3:[6478, 8580, 8729]}
        # dummy_ids = [[v] for k, v in dummy_ids.items()]
        # print(f"[DEBUG] 🧪 더미 ID로 테스트 중: {dummy_ids}")
        
        # travel_plan_list = travel_plans_with_debug(dummy_ids, '2025-06-18 ~ 2025-06-20')

    return render_template(
        "main_recommended.html",
        purpose_options=purpose_options,
        movement_options=movement_options,
        whowith_options=whowith_options,
        user_feature_keys=user_feature_keys,
        user_info=user_json,
        travel_plans=travel_plan_list
    )
    
# 로그인
@app.route("/login", methods=["POST"])
def login():
    input_id = request.form.get("USER_ID")
    input_pw = request.form.get("PASSWORD")

    try:
        
        user_json, s3_key = find_user_by_credentials(input_id, input_pw)
    
        if not user_json:
            travel_styles = session.get("travel_styles", [])
            print(f"travel_style{travel_styles}")
            images = find_nearest_users(travel_styles, k=5) if travel_styles else []

            return render_template(
                "main.html",
                images=images,
                travel_styles=session.get("travel_styles"),
                error="아이디 또는 비밀번호가 잘못되었습니다.",
                show_step=11
            )

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
    return redirect(url_for("main_recommended"))

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
    session["recommended_images"] = images
    
    return jsonify({
        "images": images
    })

# 마이페이지
@app.route("/main_mypage", methods=["GET", "POST"])
def main_mypage():

    username = session.get("username")
    if not username:
        flash("로그인이 필요합니다.")
        return redirect(url_for("main"))
    
    if request.method == "GET":
        try:
            user_json = get_user_info(username)
            if user_json:
                return render_template("main_mypage.html", user=user_json, today=datetime.today().strftime('%Y-%m-%d'))
            return "사용자 정보를 찾을 수 없습니다.", 404
        except RuntimeError as e:
            return str(e), 500
        
    elif request.method == "POST":
        update_fields = [
            'NAME', 'GENDER', 'BIRTHDATE',
            'TRAVEL_TERM', 'TRAVEL_NUM',
            'TRAVEL_LIKE_SIDO_1', 'TRAVEL_LIKE_SIDO_2', 'TRAVEL_LIKE_SIDO_3',
            'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3',
            'TRAVEL_STYL_4', 'TRAVEL_STYL_5', 'TRAVEL_STYL_6',
            'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
            'TRAVEL_MOTIVE_1', 'TRAVEL_MOTIVE_2'
        ]
        updated_data = {field: request.form.get(field, "") for field in update_fields}
        try:
            success = update_user_info(username, updated_data)
            
            if success:
                flash("회원 정보가 성공적으로 수정되었습니다.")
                return redirect(url_for("main_recommended"))
            return "수정 대상 사용자를 찾을 수 없습니다.", 404
        except RuntimeError as e:
            return str(e), 500
    
# 내 여행 페이지
@app.route("/main_mytravel")
def main_mytravel():
    if "username" not in session:
        return render_template("main_recommended.html")

    user_id = session["username"]
    all_plans = load_all_travel_plans(user_id)

    return render_template("main_mytravel.html", travel_list=all_plans)

@app.route("/view_travel_plan/<key>")
def view_travel_plan(key):
    if "username" not in session:
        return redirect(url_for("main_recommended"))

    user_id = session["username"]
    s3_key = f"user_travel_plans/{user_id}/{key}.json"
    try:
        content = s3.get_object(Bucket=BUCKET_NAME, Key=s3_key)['Body'].read().decode('utf-8')
        travel_data = json.loads(content)
    except Exception as e:
        print(f"[ERROR] 여행 상세 보기 실패: {e}")
        return "일정 로드 실패", 500

    travel_plans = []
    for day_index, (_, day_plan) in enumerate(travel_data.items(), start=1):
        title = day_plan.get("title", f"Day {day_index}")
        date_str = title.split("|")[-1].strip() if "|" in title else ""
        spots = day_plan.get("route", [])

        travel_plans.append({
            "day": day_index,
            "route_id": date_str,
            "spots": [
                {
                    "name": spot.get("name", "Unknown"),
                    "coords": [spot.get("x"), spot.get("y")],
                    "route_code": spot.get("route_code", "N/A")
                }
                for spot in spots
            ]
        })

    return render_template("view_travel_plan.html", travel_plans=travel_plans, plan_key=key)


@app.route("/save_plan", methods=["POST"])
def save_plan():
    if "username" not in session:
        return jsonify({"error": "로그인 필요"}), 401

    user_id = session["username"]
    
    try:
        data = request.get_json()
        print(f"[DEBUG] 받은 데이터: {data}")  # 디버깅용
        
        # 데이터가 직접 전달되는 경우와 plan 키로 감싸진 경우 모두 처리
        if isinstance(data, dict) and "plan" in data:
            plan_data = data.get("plan")
            custom_title = data.get("custom_title", "")
            
        else:
            plan_data = data  # 데이터가 직접 전달된 경우
            custom_title = ""
        
        if not plan_data:
            return jsonify({"error": "빈 데이터"}), 400

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        key = f"travel__{timestamp}"  # 경로 제거
        s3_key = f"user_travel_plans/{user_id}/{key}.json"

        # 메타 정보 포함
        if "meta" not in plan_data:
            plan_data["meta"] = {}
        plan_data["meta"]["custom_title"] = custom_title
        plan_data["meta"]["saved_at"] = datetime.now().isoformat()

        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=s3_key,
            Body=json.dumps(plan_data, ensure_ascii=False),
            ContentType="application/json"
        )
        
        print(f"[DEBUG] 저장 성공: {s3_key}")
        return jsonify({"status": "ok", "key": key}), 200
        
    except Exception as e:
        print(f"[ERROR] 여행 저장 실패: {e}")
        return jsonify({"error": "서버 에러"}), 500

@app.route("/delete_travel_plan", methods=["POST"])
def delete_travel_plan():
    if "username" not in session:
        return jsonify({"error": "로그인 필요"}), 401

    user_id = session["username"]
    key = request.json.get("key")
    s3_key = f"user_travel_plans/{user_id}/{key}.json"

    try:
        s3.delete_object(Bucket=BUCKET_NAME, Key=s3_key)
        return jsonify({"status": "삭제 완료"}), 200
    except Exception as e:
        print(f"[ERROR] 삭제 실패: {e}")
        return jsonify({"error": "삭제 실패"}), 500

@app.route('/update_travel_title', methods=['POST'])
def update_travel_title():
    if "username" not in session:
        return jsonify({"success": False, "error": "로그인 필요"}), 401
    
    data = request.get_json()
    key = data.get('key')
    new_title = data.get('title')
    
    if not key or not new_title:
        return jsonify({"success": False, "error": "키와 제목이 필요합니다"}), 400
    
    if len(new_title) > 50:
        return jsonify({"success": False, "error": "제목은 50자 이하여야 합니다"}), 400
    
    user_id = session["username"]
    s3_key = f"user_travel_plans/{user_id}/{key}.json"
    
    try:
        # 기존 여행 데이터 가져오기
        content = s3.get_object(Bucket=BUCKET_NAME, Key=s3_key)['Body'].read().decode('utf-8')
        travel_data = json.loads(content)
        
        # 각 일차의 title을 새로운 제목으로 업데이트
        for day_key, day_data in travel_data.items():
            if day_key != "meta":  # meta 정보는 제외
                # 기존 title에서 날짜 부분 추출 (| 뒤의 부분)
                old_title = day_data.get("title", "")
                if "|" in old_title:
                    date_part = old_title.split("|")[-1].strip()
                    day_data["title"] = f"{new_title} | {date_part}"
                else:
                    day_data["title"] = new_title
        
        # meta 정보에도 custom_title 업데이트
        if "meta" not in travel_data:
            travel_data["meta"] = {}
        travel_data["meta"]["custom_title"] = new_title
        
        # S3에 업데이트된 데이터 저장
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=s3_key,
            Body=json.dumps(travel_data, ensure_ascii=False),
            ContentType="application/json"
        )
        
        return jsonify({"success": True}), 200
        
    except s3.exceptions.NoSuchKey:
        return jsonify({"success": False, "error": "여행 계획을 찾을 수 없습니다"}), 404
    except Exception as e:
        print(f"[ERROR] 제목 업데이트 실패: {e}")
        return jsonify({"success": False, "error": "서버 오류가 발생했습니다"}), 500


@app.template_filter("regex_replace")
def regex_replace(s, pattern, repl):
    return re.sub(pattern, repl, s)

if __name__ == "__main__":
    app.run(debug=True, threaded=False)

if __name__ == "__main__":
    style_vec = [5, 5, 3, 2, 4, 5, 3, 6]  # 테스트용 input
    ids = find_nearest_users(style_vec, k=5)
    print("유사한 유저 ID:", ids)
    