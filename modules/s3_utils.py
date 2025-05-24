import boto3
import json
import botocore
import os
from dotenv import load_dotenv
import pymysql
import random
import requests

import faiss
import joblib
import numpy as np
import pandas as pd

from sqlalchemy import create_engine,text



load_dotenv(override=True)

# AWS Setting
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
REGION_NAME = os.getenv("AWS_REGION")

# EC2 URL
EC2_API_URL = os.getenv("EC2_PUBLIC_ADDR")

 # RDS Setting
DB_HOST = os.getenv("RDB_HOST")
DB_USER = os.getenv("RDB_USER")
DB_PASSWORD = os.getenv("RDB_PASSWORD")
DB_NAME = os.getenv("RDB_NAME")
DB_PORT = int(os.getenv("RDB_PORT"))

# S3 setting
s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=REGION_NAME,
    config=botocore.client.Config(signature_version='s3v4')
)

# S3에서 key로 파일을 읽고 json으로 반환환
def get_json_from_s3(key):
    try:
        file_obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        body = file_obj['Body'].read().decode('utf-8').strip()

        print(f"[DEBUG] S3 body for key '{key}': {repr(body)}")  # 핵심 디버깅

        if not body:
            print(f"[SKIP] 빈 JSON 파일: {key}")
            return None

        return json.loads(body)

    except Exception as e:
        print(f"[ERROR] JSON 파싱 실패: {key} → {str(e)}")
        return None

# prefix로 시작하는 S3 객체 리스트 반환환
def list_s3_objects(prefix):
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
    return [obj for obj in response.get('Contents', []) if obj['Key'].endswith(".json")]


# # key에 대해 1시간짜리 presigned URL 생성성
def get_s3_signed_urls(reverse=False):
    prefix = 'data/resized_image/E/'
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)

    all_keys = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.jpg')]
    selected_keys = random.sample(all_keys, k=min(10, len(all_keys)))

    signed_urls = [
        s3.generate_presigned_url('get_object', Params={'Bucket': BUCKET_NAME, 'Key': key}, ExpiresIn=3600)
        for key in selected_keys
    ]

    return [{"url": url, "area": ""} for url in signed_urls]


# user 정보 반환 ( input = ID output = dict )
def get_user_info(username):
    objects = list_s3_objects("users/")
    json_objects = [obj for obj in objects if obj['Key'].endswith('.json')] # .json 파일 필터링링

    for obj in json_objects:
        key = obj['Key']
        try:
            user_json = get_json_from_s3(key)   # dict
        except Exception as e:
            print(f"[!] JSON 파싱 실패: {key} → {e}")
            continue
        if user_json.get("USER_ID") == username:    # USER_ID가 일치하면 해당 유저 정보 반환 
            return user_json                        # dict
    return None

# S3에 Json 저장
def put_json_to_s3(key, data):
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=json.dumps(data, ensure_ascii=False).encode('utf-8'),
        ContentType='application/json'
    )

# 
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



def get_images_by_travel_ids(travel_ids):
    conn = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            db=DB_NAME,
            port = DB_PORT,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
    try:
        with conn.cursor() as cursor:
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
        
        print(f"최종이미지: {image_infos}")
        return image_infos

    finally:
        conn.close()

style_cols = ['TRAVEL_STYL_1',
            'TRAVEL_STYL_2',
            'TRAVEL_STYL_3',
            'TRAVEL_STYL_4',
            'TRAVEL_STYL_5',
            'TRAVEL_STYL_6',
            'TRAVEL_STYL_7',
            'TRAVEL_STYL_8']

# 로그인 전 사용자용
# input : 8개의 Style 점수 Output : Travel_ID -> get_images_by_travel_ids(Travel_ID)
def find_nearest_users(input_vec, k=5):

      # SQLAlhemy Engine
    engine = create_engine(
        f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    try:
        # 유저 데이터 가져오기
        user_df = pd.read_sql("SELECT * FROM users", con=engine)

        
        style_df = user_df[style_cols]
        style_array = style_df.to_numpy().astype('float32')
        
        input_vec = np.array(input_vec, dtype='float32').reshape(1, -1)
        
        d = style_array.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(style_array)
        D, I = index.search(input_vec, k)
    
        # 유사 유저 추출
        similar_users = user_df.iloc[I[0]]

        id_col = "TRAVELER_ID" if "TRAVELER_ID" in user_df.columns else "USER_ID"
        traveler_ids = similar_users[id_col].tolist() if k > 1 else [similar_users[id_col]]

        print(f"user ids: {traveler_ids}")
        # 여행 정보 필터링
        format_strings = ','.join(['%s'] * len(traveler_ids))
        sql = f"SELECT * FROM travel WHERE TRAVELER_ID IN ({format_strings})"
        travel_df = pd.read_sql(sql, con=engine, params=tuple(traveler_ids))
        print(f"travel_df: {travel_df}")

        travel_ids = travel_df['TRAVEL_ID'].tolist()
        print(f"travel_ids:{travel_ids}")
        return get_images_by_travel_ids(travel_ids)
        
    except Exception as e:
        print("[ERROR] find_nearest_users 실패:", e)
        return []

# 로그인 사용자용
def get_user_recommended_images_and_areas(username):

    try:
        user_data = get_user_info(username)
        if not user_data:
            raise Exception("사용자 정보 없음")

        # EC2 서버 호출
        res = requests.post(EC2_API_URL, json=user_data)
        if res.status_code != 200:
            raise Exception("EC2 요청 실패")
        
        recommended_ids = res.json().get("recommended_travel_ids", [])
        if not recommended_ids:
            raise Exception("추천 travel_id 없음")

        # RDS 연결 및 조회
        conn = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            db=DB_NAME,
            port=DB_PORT,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )

        with conn.cursor() as cursor:
            placeholders = ','.join(['%s'] * len(recommended_ids))
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
            cursor.execute(sql, tuple(recommended_ids))
            results = cursor.fetchall()

        conn.close()
        print(results)
        # presigned URL 생성
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

    except Exception as e:
        print("추천 이미지 처리 오류:", e)
        return []






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