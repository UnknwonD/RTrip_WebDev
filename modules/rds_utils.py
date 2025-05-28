from config import engine, s3, BUCKET_NAME, EC2_API_URL
import pymysql
import requests
import pandas as pd
import numpy as np
import faiss
from sqlalchemy import text
import json
import random
import numbers

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
        print(f"user_df : {user_df}")
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

# GNN 

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
    print(f"[DEBUG] 요청된 NEW_VISIT_AREA_ID: {new_visit_area_id}")
    
    if not new_visit_area_id:
        print(f"[DEBUG] ⚠️ 빈 NEW_VISIT_AREA_ID 입력")
        return None
    
    if isinstance(new_visit_area_id, (np.integer,)):
        new_visit_area_id = int(new_visit_area_id)
    
    if not new_visit_area_id:
        print(f"[DEBUG] ⚠️ 빈 NEW_VISIT_AREA_ID 입력")
        return None
    
    if isinstance(new_visit_area_id, numbers.Integral):
        new_visit_area_id = (new_visit_area_id,)
    elif isinstance(new_visit_area_id, list):
        new_visit_area_id = tuple(new_visit_area_id)
        
    try:
        query = """
        (
            SELECT 
                NEW_VISIT_AREA_ID,
                VISIT_AREA_NM,
                X_COORD,
                Y_COORD
            FROM place_info_new
            WHERE NEW_VISIT_AREA_ID IN :ids
        )
        UNION
        (
            SELECT 
                NEW_VISIT_AREA_ID,
                NULL AS VISIT_AREA_NM,
                NULL AS X_COORD,
                NULL AS Y_COORD
            FROM meta_photo_new
            WHERE NEW_VISIT_AREA_ID IN :ids
            AND NEW_VISIT_AREA_ID NOT IN (
                SELECT NEW_VISIT_AREA_ID FROM place_info_new WHERE NEW_VISIT_AREA_ID IN :ids
            )
        )
        """

        with engine.connect() as conn:
            result = conn.execute(text(query), {"ids": tuple(new_visit_area_id)})
            rows = result.fetchall()

            if not rows:
                print(f"[DEBUG] ⚠️ 결과 없음 for NEW_VISIT_AREA_ID: {new_visit_area_id}")
                return None

            data = {}
            print(f"[DEBUG] ✅ 결과: {data}")

            for row in rows:
                data[row.NEW_VISIT_AREA_ID] = {
                    "area_id": row.NEW_VISIT_AREA_ID,
                    "area": row.VISIT_AREA_NM or "[이름없음]",
                    "x": row.X_COORD,
                    "y": row.Y_COORD
                }
        
        return data
            
    except Exception as e:
        print(f"[DEBUG] ❌ get_meta_photo_info 전체 오류: {e}")
        return None

def pares_dates(travel_date):
    from datetime import datetime, timedelta
    # 시작일과 종료일 파싱
    start_str, end_str = travel_date.split(' ~ ')
    start_date = datetime.strptime(start_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_str, '%Y-%m-%d')

    # 시작일 ~ 종료일 리스트 생성
    date_list = []
    current_date = start_date
    while current_date <= end_date:
        formatted_date = f'{current_date.month}월 {current_date.day}일'
        date_list.append(formatted_date)
        current_date += timedelta(days=1)

    return date_list


def travel_plans_with_debug(area_ids, travel_date:str):
    # travel_date = '2025-06-18 ~ 2025-06-21' 이런식으로 들어옴
    
    print(f"[DEBUG] 🔁 travel_plans_with_debug() 호출됨. area_ids: {area_ids}")
    
    if not area_ids or len(area_ids) == 0:
        print(f"[DEBUG] ⚠️ 빈 area_ids 입력, 기본 계획 반환")
        return default_travel_plans()
 
    plans = []
    date_list = pares_dates(travel_date) # 6월 8일, 6월 10일 ....

    for i, route in enumerate(area_ids):
        print(f"[DEBUG] 📍 처리 중인 루트 {i+1}: {route}")
        
        route_infos = []
        
        # 🚀 Batch로 한 번에 가져오기
        area_info_dict = get_meta_photo_info(route)
        route_infos = []
        
        for area_id in route:
            photo = area_info_dict.get(area_id)
            if photo:
                route_infos.append({
                    "name": photo["area"],
                    "x": photo["x"],
                    "y": photo["y"]
                })
                print(f"[DEBUG] ✅ 장소 정보 추가: {photo['area']}")
            else:
                print(f"[DEBUG] ⚠️ area_id {area_id}에 대한 장소 정보 없음")
        
        if route_infos:
            plans.append({
                "title": f"{i+1}일차 추천루트 | {date_list[i]}",
                "description": f"{route_infos[0]['name']}을(를) 포함한 여행 경로입니다.",
                "route": route_infos
            })
            print(f"[DEBUG] ✅ 루트 {i+1} 생성 완료, {len(route_infos)}개 장소")
        else:
            print(f"[DEBUG] ⚠️ 루트 {i+1}에서 유효한 장소 정보 없음")
    
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
                        {
                            "name": "청계천",
                            "description": "산책하기 좋은 도심 속 힐링 장소",
                            "x": 126.9784,
                            "y": 37.5703
                        },
                        {
                            "name": "경복궁",
                            "description": "조선의 중심, 서울의 상징",
                            "x": 126.9770,
                            "y": 37.5788
                        },
                        {
                            "name": "남산타워",
                            "description": "서울의 전경을 한눈에",
                            "x": 126.9882,
                            "y": 37.5512
                        }
                    ]
        }
    ]

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
                "title": f"{i+1} 일차",
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



# 추천 받은 여행 Route Save to S3
def save_fixed_day_route(username, day, route):
    key = f"user_travel_plans/{username}.json"
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        existing = json.loads(obj['Body'].read().decode('utf-8'))
    except s3.exceptions.NoSuchKey:
        existing = {}

    existing[str(day)] = route  # day는 문자열로 key 사용
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=json.dumps(existing, ensure_ascii=False),
        ContentType='application/json'
    )
