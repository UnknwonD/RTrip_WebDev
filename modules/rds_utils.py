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
                    FROM place_info
                    WHERE travel_id IN ({placeholders})
                ),
                joined_data AS (
                    SELECT 
                        p.travel_id,
                        p.visit_area_id,
                        p.visit_area_nm,
                        m.photo_file_nm,
                        ROW_NUMBER() OVER (
                            PARTITION BY p.travel_id, p.visit_area_id
                            ORDER BY RAND()
                        ) AS rn
                    FROM filtered_place p
                    JOIN meta_photo m
                    ON p.travel_id = m.travel_id
                    AND p.visit_area_id = m.visit_area_id
                    WHERE m.photo_file_nm IS NOT NULL
                )
                SELECT
                    travel_id,
                    visit_area_id,
                    visit_area_nm,
                    photo_file_nm
                FROM joined_data
                WHERE rn = 1
                ORDER BY travel_id, visit_area_id
            """
            cursor.execute(sql, tuple(travel_ids))
            results = cursor.fetchall()

        prefix = "data/resized_image/E/"
        image_infos = []
        for row in results:
            file_name = row.get("photo_file_nm")
            if not file_name:
                continue
            url = s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': BUCKET_NAME, 'Key': f"{prefix}{file_name}"},
                ExpiresIn=3600
            )
            image_infos.append({
                "url": url,
                "area": row["visit_area_nm"],
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


def get_meta_photo_info(visit_area_id):
    query = """
        WITH ranked_places AS (
            SELECT 
                pi.*,
                ROW_NUMBER() OVER (PARTITION BY pi.VISIT_AREA_ID ORDER BY pi.VISIT_AREA_NM) AS rn
            FROM place_info pi
            WHERE pi.VISIT_AREA_ID = :id
        )
        SELECT 
            pi.VISIT_AREA_ID,
            pi.VISIT_AREA_NM,
            mp.PHOTO_FILE_NM,
            mp.PHOTO_FILE_X_COORD,
            mp.PHOTO_FILE_Y_COORD
        FROM ranked_places pi
        JOIN meta_photo mp ON pi.VISIT_AREA_ID = mp.VISIT_AREA_ID
        WHERE rn = 1
    """

    with engine.connect() as conn:
        result = conn.execute(text(query), {"id": visit_area_id})
        row = result.fetchone()
        # print(row)
        # print("=" * 100)
        try:
            photo = dict(row._mapping)
        except:
            return None

    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET_NAME, "Key": f"data/resized_image/E/{photo['PHOTO_FILE_NM']}"},
        ExpiresIn=3600
    )

    return {
        "file_name": photo["PHOTO_FILE_NM"],
        "url": url,
        "x": photo["PHOTO_FILE_X_COORD"],
        "y": photo["PHOTO_FILE_Y_COORD"],
        "area": photo["VISIT_AREA_NM"] or "[이름없음]" 
    }


# 여러 여행 경로 리스트를 받아서 여행 추천 정보를 구성

def travel_plans(area_ids):
    plans = []

    # 3개씩 나누기
    route_lists = [area_ids[i:i+3] for i in range(0, len(area_ids), 3)]

    for i, route in enumerate(route_lists):
        route_infos = []
        main_img_url = ""

        for idx, area_id in enumerate(route):
            photo = get_meta_photo_info(area_id)
            if photo:
                if idx == 0:
                    main_img_url = photo["url"]

                route_infos.append({
                    "name": photo["area"],
                    "x": photo["x"],
                    "y": photo["y"],
                    "url": photo["url"]
                })

        if route_infos:
            plans.append({
                "main_image_url": main_img_url,
                "title": f"추천 루트 {i+1}",
                "description": f"{route_infos[0]['name']}을(를) 포함한 여행 경로입니다.",
                "route": route_infos
            })

    return plans



# 사용자 입력이 없을 경우 사용할 기본 여행 경로 예시 반환

def default_travel_plans():
    return [
        {
            "title": "인기 여행 루트",
            "main_image_url": "https://rtrip.s3.amazonaws.com/data/resized_image/E/sample.jpg",
            "description": "많은 사람들이 방문한 인기 루트예요!",
            "route": [
                {"name": "청계천", "description": "산책하기 좋은 도심 속 힐링 장소"},
                {"name": "경복궁", "description": "조선의 중심, 서울의 상징"},
                {"name": "남산타워", "description": "서울의 전경을 한눈에"}
            ]
        }
    ]




# def get_user_recommended_images_and_areas(travel_ids):
#     try:
        
#         conn = pymysql.connect(
#             host=DB_HOST,
#             user=DB_USER,
#             password=DB_PASSWORD,
#             db=DB_NAME,
#             port = DB_PORT,
#             charset='utf8mb4',
#             cursorclass=pymysql.cursors.DictCursor
#         )

#         with conn.cursor() as cursor:
            
#             format_str = ','.join(['%s'] * len(travel_ids))
#             sql = """
#                 WITH filtered_place AS (
#                 SELECT *
#                 FROM place_info
#                 WHERE travel_id IN (
#                     {}
#                 ) 
#             ),
#             joined_data AS (
#                 SELECT 
#                     p.travel_id,
#                     p.visit_area_id,
#                     p.visit_area_nm,
#                     m.photo_file_nm,
#                     ROW_NUMBER() OVER (
#                         PARTITION BY p.travel_id, p.visit_area_id
#                         ORDER BY RAND()
#                     ) AS rn
#                 FROM filtered_place p
#                 JOIN meta_photo m
#                 ON p.travel_id = m.travel_id
#                 AND p.visit_area_id = m.visit_area_id
#                 WHERE m.photo_file_nm IS NOT NULL
#             )
#             SELECT
#                 travel_id,
#                 visit_area_id,
#                 visit_area_nm,
#                 photo_file_nm
#             FROM joined_data
#             WHERE rn = 1
#             ORDER BY travel_id, visit_area_id;

#             """.format(format_str)
#             cursor.execute(sql, travel_ids)
#             results = cursor.fetchall()
#         conn.close()

        
#         prefix = "data/resized_image/E/"
#         image_infos = []
#         for row in results:
#             file_name = row.get("photo_file_nm")
#             area_name = row.get("visit_area_nm")
#             area_id = row.get("visit_area_id")

#             if file_name:
#                 url = s3.generate_presigned_url(
#                     'get_object',
#                     Params={'Bucket': BUCKET_NAME, 'Key': f"{prefix}{file_name}"},
#                     ExpiresIn=3600
#                 )
#                 image_infos.append({"url": url, "area": area_name, "area_id": area_id})
#         print(file_name)
#         print(url)
#         return image_infos
    
#     except Exception as e:
#         print("DB_HOST:", DB_HOST)

#         print(f" 추천 이미지 처리 오류: {str(e)}")
#         return []











# def get_user_recommended_images_and_areas(username):
#     try:
#         # 1. 사용자 JSON 불러오기
#         user_data = get_user_info(username)
#         if not user_data:
#             raise Exception("사용자 정보 없음")

#         # 2. EC2에 유저 정보 전송 → 추천 travel_id 리스트 응답
#         res = requests.post(EC2_API_URL, json=user_data)
#         if res.status_code != 200:
#             raise Exception("EC2 추천 요청 실패")

#         recommended_ids = res.json().get("recommended_travel_ids", [])
#         if not recommended_ids:
#             raise Exception("추천 travel_id 없음")

#         # 3. RDS 연결
#         conn = pymysql.connect(
#             host=DB_HOST,
#             user=DB_USER,
#             password=DB_PASSWORD,
#             db=DB_NAME,
#             port = DB_PORT,
#             charset='utf8mb4',
#             cursorclass=pymysql.cursors.DictCursor
#         )

#         with conn.cursor() as cursor:
#             # 4. place_info → meta_photo 조인 쿼리
#             format_str = ','.join(['%s'] * len(recommended_ids))
#             sql = """
#                 WITH filtered_place AS (
#                 SELECT *
#                 FROM place_info
#                 WHERE travel_id IN (
#                     {}
#                 )
#             ),
#             joined_data AS (
#                 SELECT 
#                     p.travel_id,
#                     p.visit_area_id,
#                     p.visit_area_nm,
#                     m.photo_file_nm,
#                     ROW_NUMBER() OVER (
#                         PARTITION BY p.travel_id, p.visit_area_id
#                         ORDER BY RAND()
#                     ) AS rn
#                 FROM filtered_place p
#                 JOIN meta_photo m
#                 ON p.travel_id = m.travel_id
#                 AND p.visit_area_id = m.visit_area_id
#                 WHERE m.photo_file_nm IS NOT NULL
#             )
#             SELECT
#                 travel_id,
#                 visit_area_id,
#                 visit_area_nm,
#                 photo_file_nm
#             FROM joined_data
#             WHERE rn = 1
#             ORDER BY travel_id, visit_area_id;

#             """.format(format_str)
#             cursor.execute(sql, recommended_ids)
#             results = cursor.fetchall()
#         conn.close()

#         # 5. presigned URL 생성
#         prefix = "data/resized_image/E/"
#         image_infos = []
#         for row in results:
#             file_name = row.get("photo_file_nm")
#             area_name = row.get("visit_area_nm")
#             area_id = row.get("visit_area_id")

#             if file_name:
#                 url = s3.generate_presigned_url(
#                     'get_object',
#                     Params={'Bucket': BUCKET_NAME, 'Key': f"{prefix}{file_name}"},
#                     ExpiresIn=3600
#                 )
#                 image_infos.append({"url": url, "area": area_name, "area_id": area_id})
          
#         return image_infos
    
#     except Exception as e:
#         print("DB_HOST:", DB_HOST)

#         print(f" 추천 이미지 처리 오류: {str(e)}")
#         return []
