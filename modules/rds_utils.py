from config import engine, s3, BUCKET_NAME, EC2_API_URL
import pymysql
import requests
import pandas as pd
import numpy as np
import faiss
from sqlalchemy import text
import json
import random

style_cols = [
    'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4',
    'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8'
]

def get_images_by_travel_ids(travel_ids):
    conn = engine.raw_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            placeholders = ','.join(['%s'] * len(travel_ids))
            sql = f"""
            WITH filtered_place AS (
                SELECT *
                FROM place_info_new
                WHERE TRAVEL_ID IN ({placeholders})
            ),
            joined_data AS (
                SELECT
                    p.TRAVEL_ID,
                    p.NEW_VISIT_AREA_ID,
                    p.VISIT_AREA_NM,
                    m.PHOTO_FILE_NM,
                    ROW_NUMBER() OVER (
                        PARTITION BY p.TRAVEL_ID, p.NEW_VISIT_AREA_ID
                        ORDER BY RAND()
                    ) AS rn
                FROM filtered_place p
                JOIN meta_photo_new m
                ON p.NEW_VISIT_AREA_ID = m.NEW_VISIT_AREA_ID
                WHERE m.PHOTO_FILE_NM IS NOT NULL
            )
            SELECT
                travel_id,
                NEW_VISIT_AREA_ID AS visit_area_id,
                VISIT_AREA_NM,
                PHOTO_FILE_NM
            FROM joined_data
            WHERE rn = 1
            ORDER BY travel_id, NEW_VISIT_AREA_ID
            """
            cursor.execute(sql, tuple(travel_ids))
            results = cursor.fetchall()
            
            prefix = "data/resized_image/E/"
            image_infos = []
            for row in results:
                file_name = row.get("PHOTO_FILE_NM")
                if not file_name:
                    continue
                    
                url = s3.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': BUCKET_NAME, 'Key': f"{prefix}{file_name}"},
                    ExpiresIn=3600
                )
                image_infos.append({
                    "url": url,
                    "area": row["VISIT_AREA_NM"],
                    "area_id": row["visit_area_id"]
                })
            return image_infos
    finally:
        conn.close()

def find_nearest_users(input_vec, k=5):
    try:
        user_df = pd.read_sql("SELECT * FROM users", con=engine)
        style_df = user_df[style_cols]
        style_array = style_df.to_numpy().astype('float32')
        
        input_vec = np.array(input_vec, dtype='float32').reshape(1, -1)
        
        d = style_array.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(style_array)
        
        D, I = index.search(input_vec, k)
        similar_users = user_df.iloc[I[0]]
        
        id_col = "TRAVELER_ID" if "TRAVELER_ID" in user_df.columns else "USER_ID"
        traveler_ids = similar_users[id_col].tolist() if k > 1 else [similar_users[id_col]]
        
        sql = f"SELECT * FROM travel WHERE TRAVELER_ID IN ({','.join(['%s']*len(traveler_ids))})"
        travel_df = pd.read_sql(sql, con=engine, params=tuple(traveler_ids))
        travel_ids = travel_df['TRAVEL_ID'].tolist()
        
        return get_images_by_travel_ids(travel_ids)
    except Exception as e:
        print("[ERROR] find_nearest_users 실패:", e)
        return []

def get_user_recommended_images_and_areas(username):
    from modules.s3_utils import get_user_info
    try:
        user_data = get_user_info(username)
        if not user_data:
            raise Exception("사용자 정보 없음")
            
        res = requests.post(EC2_API_URL, json=user_data)
        if res.status_code != 200:
            raise Exception("EC2 요청 실패")
            
        recommended_ids = res.json().get("recommended_travel_ids", [])
        if not recommended_ids:
            raise Exception("추천 travel_id 없음")
            
        return get_images_by_travel_ids(recommended_ids)
    except Exception as e:
        print("추천 이미지 처리 오류:", e)
        return []

def get_meta_photo_info(new_visit_area_id):
    """
    NEW_VISIT_AREA_ID로 메타 포토 정보를 가져오는 함수
    수정: 예외 처리 강화 및 디버깅 개선
    """
    print(f"[DEBUG] 🔍 요청된 NEW_VISIT_AREA_ID: {new_visit_area_id}")
    
    # 입력값 검증
    if not new_visit_area_id:
        print(f"[DEBUG] ⚠️ 빈 NEW_VISIT_AREA_ID 입력")
        return None
        
    try:
        query = """
        WITH combined_data AS (
            SELECT
                pi.NEW_VISIT_AREA_ID,
                pi.VISIT_AREA_NM,
                mp.PHOTO_FILE_NM,
                mp.PHOTO_FILE_X_COORD,
                mp.PHOTO_FILE_Y_COORD,
                pi.TRAVEL_ID,
                ROW_NUMBER() OVER (
                    PARTITION BY pi.NEW_VISIT_AREA_ID
                    ORDER BY pi.TRAVEL_ID, mp.TOUR_PHOTO_SEQ
                ) AS rn
            FROM place_info_new pi
            JOIN meta_photo_new mp ON pi.NEW_VISIT_AREA_ID = mp.NEW_VISIT_AREA_ID
            AND pi.TRAVEL_ID = mp.TRAVEL_ID
            WHERE pi.NEW_VISIT_AREA_ID = :id
            AND mp.PHOTO_FILE_NM IS NOT NULL
        )
        SELECT
            NEW_VISIT_AREA_ID AS visit_area_id,
            VISIT_AREA_NM,
            PHOTO_FILE_NM,
            PHOTO_FILE_X_COORD,
            PHOTO_FILE_Y_COORD
        FROM combined_data
        WHERE rn = 1
        """
        
        with engine.connect() as conn:
            result = conn.execute(text(query), {"id": new_visit_area_id})
            row = result.fetchone()
            
            if not row:
                print(f"[DEBUG] ⚠️ 결과 없음 for NEW_VISIT_AREA_ID: {new_visit_area_id}")
                return None
                
            try:
                # SQLAlchemy 2.0 호환 방식으로 수정
                photo = dict(row._mapping)
                print(f"[DEBUG] ✅ 결과 행: {photo}")
            except Exception as e:
                print(f"[DEBUG] ❌ 행 변환 실패: {e}")
                return None
                
            # S3 URL 생성
            try:
                url = s3.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": BUCKET_NAME, "Key": f"data/resized_image/E/{photo['PHOTO_FILE_NM']}"},
                    ExpiresIn=3600
                )
            except Exception as e:
                print(f"[DEBUG] ❌ S3 URL 생성 실패: {e}")
                url = ""
            
            return {
                "file_name": photo["PHOTO_FILE_NM"],
                "url": url,
                "x": photo["PHOTO_FILE_X_COORD"],
                "y": photo["PHOTO_FILE_Y_COORD"],
                "area": photo["VISIT_AREA_NM"] or "[이름없음]"
            }
            
    except Exception as e:
        print(f"[DEBUG] ❌ get_meta_photo_info 전체 오류: {e}")
        return None

def travel_plans(area_ids):
    """
    area_ids 리스트를 받아서 여행 계획을 생성하는 함수
    수정: 예외 처리 강화 및 빈 결과 처리 개선
    """
    print(f"[DEBUG] 🔁 travel_plans() 호출됨. area_ids: {area_ids}")
    
    # 입력값 검증
    if not area_ids or len(area_ids) == 0:
        print(f"[DEBUG] ⚠️ 빈 area_ids 입력, 기본 계획 반환")
        return default_travel_plans()
    
    plans = []
    route_lists = [area_ids[i:i+3] for i in range(0, len(area_ids), 3)]
    
    for i, route in enumerate(route_lists):
        print(f"[DEBUG] 📍 처리 중인 루트 {i+1}: {route}")
        
        route_infos = []
        main_img_url = ""
        
        for idx, area_id in enumerate(route):
            try:
                photo = get_meta_photo_info(area_id)
                if photo:
                    if idx == 0:  # 첫 번째 이미지를 메인 이미지로 설정
                        main_img_url = photo["url"]
                    route_infos.append({
                        "name": photo["area"],
                        "x": photo["x"],
                        "y": photo["y"],
                        "url": photo["url"]
                    })
                    print(f"[DEBUG] ✅ 장소 정보 추가: {photo['area']}")
                else:
                    print(f"[DEBUG] ⚠️ area_id {area_id}에 대한 사진 정보 없음")
            except Exception as e:
                print(f"[DEBUG] ❌ area_id {area_id} 처리 중 오류: {e}")
                continue
        
        # 루트에 최소 하나의 장소 정보가 있는 경우에만 계획에 추가
        if route_infos:
            plans.append({
                "main_image_url": main_img_url or "https://rtrip.s3.amazonaws.com/data/resized_image/E/default.jpg",
                "title": f"추천 루트 {i+1}",
                "description": f"{route_infos[0]['name']}을(를) 포함한 여행 경로입니다.",
                "route": route_infos
            })
            print(f"[DEBUG] ✅ 루트 {i+1} 생성 완료, {len(route_infos)}개 장소")
        else:
            print(f"[DEBUG] ⚠️ 루트 {i+1}에서 유효한 장소 정보 없음")
    
    # 결과가 없으면 기본 계획 반환
    if not plans:
        print(f"[DEBUG] ⚠️ 생성된 계획이 없음, 기본 계획 반환")
        return default_travel_plans()
    
    print(f"[DEBUG] 🎯 총 {len(plans)}개의 여행 계획 생성 완료")
    return plans

def debug_area_ids(area_ids_sample):
    """
    디버깅용: 실제 데이터베이스에 존재하는 NEW_VISIT_AREA_ID들을 확인
    """
    print(f"[DEBUG] 🔬 데이터베이스 존재 여부 확인 중...")
    
    try:
        with engine.connect() as conn:
            # meta_photo_new에서 사용 가능한 ID들 샘플 확인
            sample_query = """
            SELECT DISTINCT NEW_VISIT_AREA_ID 
            FROM meta_photo_new 
            WHERE PHOTO_FILE_NM IS NOT NULL 
            LIMIT 10
            """
            result = conn.execute(text(sample_query))
            available_ids = [row[0] for row in result.fetchall()]
            print(f"[DEBUG] 📋 meta_photo_new에서 사용 가능한 ID 샘플: {available_ids}")
            
            # place_info_new에서 사용 가능한 ID들 샘플 확인  
            place_query = """
            SELECT DISTINCT NEW_VISIT_AREA_ID 
            FROM place_info_new 
            WHERE NEW_VISIT_AREA_ID IS NOT NULL 
            LIMIT 10
            """
            result = conn.execute(text(place_query))
            place_ids = [row[0] for row in result.fetchall()]
            print(f"[DEBUG] 📋 place_info_new에서 사용 가능한 ID 샘플: {place_ids}")
            
            # GNN이 추천한 ID들과 실제 DB의 ID 범위 비교
            if area_ids_sample:
                print(f"[DEBUG] 🤖 GNN 추천 ID 샘플: {area_ids_sample[:5]}")
                
                # ID 범위 확인
                max_meta_id_query = "SELECT MAX(NEW_VISIT_AREA_ID) FROM meta_photo_new"
                max_place_id_query = "SELECT MAX(NEW_VISIT_AREA_ID) FROM place_info_new"
                
                max_meta = conn.execute(text(max_meta_id_query)).fetchone()[0]
                max_place = conn.execute(text(max_place_id_query)).fetchone()[0]
                
                print(f"[DEBUG] 📊 DB ID 범위 - meta_photo_new 최대: {max_meta}, place_info_new 최대: {max_place}")
                print(f"[DEBUG] 📊 GNN ID 범위 - 최소: {min(area_ids_sample)}, 최대: {max(area_ids_sample)}")
                
    except Exception as e:
        print(f"[DEBUG] ❌ 디버깅 쿼리 실패: {e}")

def travel_plans_with_debug(area_ids):
    """
    디버깅이 강화된 travel_plans 함수
    """
    print(f"[DEBUG] 🔁 travel_plans_with_debug() 호출됨. area_ids: {area_ids}")
    
    # 입력값 검증
    if not area_ids or len(area_ids) == 0:
        print(f"[DEBUG] ⚠️ 빈 area_ids 입력, 기본 계획 반환")
        return default_travel_plans()
    
    # 디버깅 정보 출력
    debug_area_ids(area_ids)
    
    plans = []
    route_lists = [area_ids[i:i+3] for i in range(0, len(area_ids), 3)]
    
    for i, route in enumerate(route_lists):
        print(f"[DEBUG] 📍 처리 중인 루트 {i+1}: {route}")
        
        route_infos = []
        main_img_url = ""
        
        for idx, area_id in enumerate(route):
            try:
                photo = get_meta_photo_info(area_id)
                if photo:
                    if idx == 0:  # 첫 번째 이미지를 메인 이미지로 설정
                        main_img_url = photo["url"]
                    route_infos.append({
                        "name": photo["area"],
                        "x": photo["x"],
                        "y": photo["y"],
                        "url": photo["url"]
                    })
                    print(f"[DEBUG] ✅ 장소 정보 추가: {photo['area']}")
                else:
                    print(f"[DEBUG] ⚠️ area_id {area_id}에 대한 사진 정보 없음")
            except Exception as e:
                print(f"[DEBUG] ❌ area_id {area_id} 처리 중 오류: {e}")
                continue
        
        # 루트에 최소 하나의 장소 정보가 있는 경우에만 계획에 추가
        if route_infos:
            plans.append({
                "main_image_url": main_img_url or "https://rtrip.s3.amazonaws.com/data/resized_image/E/default.jpg",
                "title": f"추천 루트 {i+1}",
                "description": f"{route_infos[0]['name']}을(를) 포함한 여행 경로입니다.",
                "route": route_infos
            })
            print(f"[DEBUG] ✅ 루트 {i+1} 생성 완료, {len(route_infos)}개 장소")
        else:
            print(f"[DEBUG] ⚠️ 루트 {i+1}에서 유효한 장소 정보 없음")
    
    # 결과가 없으면 기본 계획 반환
    if not plans:
        print(f"[DEBUG] ⚠️ 생성된 계획이 없음, 기본 계획 반환")
        return default_travel_plans()
    
    print(f"[DEBUG] 🎯 총 {len(plans)}개의 여행 계획 생성 완료")
    return plans

def default_travel_plans():
    """
    기본 여행 계획을 반환하는 함수
    """
    return [
        {
            "title": "인기 여행 루트",
            "main_image_url": "https://rtrip.s3.amazonaws.com/data/resized_image/E/sample.jpg",
            "description": "많은 사람들이 방문한 인기 루트예요!",
            "route": [
                {"name": "청계천", "description": "산책하기 좋은 도심 속 힐링 장소", "x": 0, "y": 0, "url": ""},
                {"name": "경복궁", "description": "조선의 중심, 서울의 상징", "x": 0, "y": 0, "url": ""},
                {"name": "남산타워", "description": "서울의 전경을 한눈에", "x": 0, "y": 0, "url": ""}
            ]
        }
    ]
