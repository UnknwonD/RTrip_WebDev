from dependency import *
from config import s3, BUCKET_NAME
from sqlalchemy import text

import torch
import torch.multiprocessing as mp
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

############## 초기 모델 로드 ##################
with open('./pickle/visit_area_id_to_index.pkl', 'rb') as f:
    visit_area_id_to_index = pickle.load(f)

with open('./pickle/dataset.pkl', 'rb') as f:
    data = pickle.load(f)

visit_area_df = pd.read_pickle('./pickle/visit_area_df.pkl')

# 모델 로드 - 안전한 방식으로 수정
model = RouteGNN(data.metadata())
model.load_state_dict(torch.load('./pickle/routegnn_model.pt', 
                                map_location='cpu',
                                weights_only=True))
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

        session["username"] = user_data["USER_ID"]  

        return redirect(url_for("main_recommended"))

    return render_template("main_register.html")


# 메인 페이지 - 수정된 버전
@app.route("/", methods=["GET", "POST"])
def main_recommended():
    user_json = None    # user 정보
    travel_plans_data = [] # GNN 결과 여행 정보 
    print(f"[DEBUG] 🛰️ 요청 진입: {request.method}")

    if request.method == "POST":
        
        travel_input = request.form.to_dict()           # 설문 받은 유저 정보
        raw_user = get_user_info(session["username"])   # 유저 S3 기본 데이터 정보 

        # 필요 없는 정보 제거
        user_json = {
            k: v for k, v in raw_user.items() 
            if k not in {"BIRTHDATE", "uuid", "phone_number", "PASSWORD", "CONFIRM_PASSWORD"}
        }

        try:
            # GNN 모델 추론 실행
            dummy_ids, area_names = run_inference(raw_user, travel_input, model, data, visit_area_id_to_index, visit_area_df)
            print(f"[DEBUG] 🤖 GNN 추론 결과 - IDs: {dummy_ids[:10]}...")  # 처음 10개만 출력
            print(f"[DEBUG] 🤖 GNN 추론 결과 - Names: {area_names[:10]}...")  # 처음 10개만 출력
            
            # 디버깅이 강화된 travel_plans 함수 사용
            travel_plan_list = travel_plans_with_debug(dummy_ids)
            
        except Exception as e:
            print(f"[DEBUG] ❌ GNN 추론 또는 여행 계획 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            # 오류 발생 시 기본 계획 사용
            travel_plan_list = default_travel_plans()
            
    # Get 요청일 때 추천 데이터 제공 (입력 받기 전 예시 데이터)
    else:
        travel_plan_list = default_travel_plans()

    return render_template(
        "main_recommended.html", 
        purpose_options=purpose_options,
        movement_options=movement_options,
        whowith_options=whowith_options,
        user_feature_keys=user_feature_keys,
        user_info=user_json,
        travel_plans = travel_plan_list
    )
    
    
# 로그인
@app.route("/login", methods=["POST"])
def login():
    input_id = request.form.get("USER_ID")
    input_pw = request.form.get("PASSWORD")

    try:
        
        user_json, s3_key = find_user_by_credentials(input_id, input_pw)
    
        if not user_json:
            return render_template("main.html", error="아이디 또는 비밀번호가 잘못되었습니다.", show_step = 11)

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

# 추가: 데이터베이스 상태 확인용 디버깅 엔드포인트
@app.route("/debug/db_status")
def debug_db_status():
    """
    데이터베이스 상태와 ID 매핑 확인용 엔드포인트
    """
    try:
        with engine.connect() as conn:
            # 각 테이블의 레코드 수 확인
            meta_count = conn.execute(text("SELECT COUNT(*) FROM meta_photo_new")).fetchone()[0]
            place_count = conn.execute(text("SELECT COUNT(*) FROM place_info_new")).fetchone()[0]
            
            # ID 범위 확인
            meta_range = conn.execute(text("SELECT MIN(NEW_VISIT_AREA_ID), MAX(NEW_VISIT_AREA_ID) FROM meta_photo_new")).fetchone()
            place_range = conn.execute(text("SELECT MIN(NEW_VISIT_AREA_ID), MAX(NEW_VISIT_AREA_ID) FROM place_info_new")).fetchone()
            
            # 샘플 데이터
            meta_samples = conn.execute(text("SELECT NEW_VISIT_AREA_ID FROM meta_photo_new WHERE PHOTO_FILE_NM IS NOT NULL LIMIT 10")).fetchall()
            place_samples = conn.execute(text("SELECT NEW_VISIT_AREA_ID FROM place_info_new WHERE NEW_VISIT_AREA_ID IS NOT NULL LIMIT 10")).fetchall()
            
            debug_info = {
                "meta_photo_new": {
                    "count": meta_count,
                    "id_range": meta_range,
                    "samples": [row[0] for row in meta_samples]
                },
                "place_info_new": {
                    "count": place_count,
                    "id_range": place_range,
                    "samples": [row[0] for row in place_samples]
                }
            }
            
            return jsonify(debug_info)
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# visit_area_id_to_index 딕셔너리 확인용 엔드포인트  
@app.route("/debug/gnn_mapping")
def debug_gnn_mapping():
    """
    GNN 모델의 ID 매핑 확인용 엔드포인트
    """
    try:
        # visit_area_id_to_index에서 샘플 확인
        sample_mappings = dict(list(visit_area_id_to_index.items())[:10])
        
        # 역매핑도 확인
        index_to_visit_area_id = {v: k for k, v in visit_area_id_to_index.items()}
        reverse_samples = dict(list(index_to_visit_area_id.items())[:10])
        
        mapping_info = {
            "total_mappings": len(visit_area_id_to_index),
            "id_to_index_samples": sample_mappings,
            "index_to_id_samples": reverse_samples,
            "max_index": max(visit_area_id_to_index.values()) if visit_area_id_to_index else 0,
            "min_id": min(visit_area_id_to_index.keys()) if visit_area_id_to_index else 0,
            "max_id": max(visit_area_id_to_index.keys()) if visit_area_id_to_index else 0
        }
        
        return jsonify(mapping_info)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/map_test")
def map_test():
    return render_template("map_test.html")
# 통합된 메인 실행 블록
if __name__ == "__main__":
    # multiprocessing 설정
    mp.set_start_method('spawn', force=True)
    
    # Flask 앱 실행
    app.run(debug=True, use_reloader=False, threaded=True)