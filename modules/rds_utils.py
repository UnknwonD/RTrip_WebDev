import pymysql
import random
import os
from dotenv import load_dotenv
import random
import boto3
import botocore


load_dotenv(override=True)

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
REGION_NAME = os.getenv("AWS_REGION")

DB_HOST = os.getenv("RDB_HOST")
DB_USER = os.getenv("RDB_USER")
DB_PASSWORD = os.getenv("RDB_PASSWORD")
DB_NAME = os.getenv("RDB_NAME")
DB_PORT = int(os.getenv("RDB_PORT"))

s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=REGION_NAME,
    config=botocore.client.Config(signature_version='s3v4')
)

def get_random_photo_filename(travel_id):
    try:
        conn = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            port = DB_PORT,
            password=DB_PASSWORD,
            db=DB_NAME,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        with conn.cursor() as cursor:
            sql = "SELECT PHOTO_FILE_NM FROM photo WHERE TRAVEL_ID = %s"
            cursor.execute(sql, (travel_id,))
            results = cursor.fetchall()

            if not results:
                print(f"[!] TRAVEL_ID에 해당하는 이미지 없음: {travel_id}")
                return None

            photo_file = random.choice(results)
            return photo_file['PHOTO_FILE_NM']
    except Exception as e:
        print(f"[!] RDS 조회 오류: {e}")
        return None
    finally:
        if 'conn' in locals():
            conn.close()



def get_images_by_travel_ids(travel_ids):
    try:
        # 1. RDS 연결
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
            # 2. place_info → meta_photo 조인 쿼리
            format_str = ','.join(['%s'] * len(travel_ids))
            sql = f"""
                WITH filtered_place AS (
                    SELECT *
                    FROM place_info
                    WHERE travel_id IN ({format_str})
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
                ORDER BY travel_id, visit_area_id;
            """
            cursor.execute(sql, travel_ids)
            results = cursor.fetchall()
        conn.close()

        # 3. presigned URL 생성
        prefix = "data/resized_image/E/"
        image_infos = []
        for row in results:
            file_name = row.get("photo_file_nm")
            area_name = row.get("visit_area_nm")
            area_id = row.get("visit_area_id")

            if file_name:
                url = s3.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': BUCKET_NAME, 'Key': f"{prefix}{file_name}"},
                    ExpiresIn=3600
                )
                image_infos.append({
                    "url": url,
                    "area": area_name,
                    "area_id": area_id
                })
        return image_infos

    except Exception as e:
        print(f"[!] 이미지 로딩 오류: {str(e)}")
        return []
    
def get_random_images_from_rds(k=10):
    try:
        # RDS에서 travel_id 전체 중 일부만 랜덤으로 추출
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
            cursor.execute("SELECT DISTINCT travel_id FROM place_info")
            all_ids = [row['travel_id'] for row in cursor.fetchall()]
        conn.close()

        sampled_ids = random.sample(all_ids, k=min(k, len(all_ids)))
        return get_images_by_travel_ids(sampled_ids)
    
    except Exception as e:
        print(f"랜덤 이미지 오류: {str(e)}")
        return []
