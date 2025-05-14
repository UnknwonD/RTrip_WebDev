import boto3
import json
import botocore
import os
from dotenv import load_dotenv
import pymysql
import requests


load_dotenv(override=True)

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
REGION_NAME = os.getenv("AWS_REGION")

EC2_API_URL = os.getenv("EC2_PUBLIC_ADDR")

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
def get_json_from_s3(key):
    file_obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
    return json.loads(file_obj['Body'].read().decode('utf-8'))

def get_user_info(username):
    objects = list_s3_objects("users/")

    json_objects = [obj for obj in objects if obj['Key'].endswith('.json')]

    for obj in json_objects:
        key = obj['Key']
        try:
            user_json = get_json_from_s3(key)
        except Exception as e:
            print(f"[!] JSON 파싱 실패: {key} → {e}")
            continue
        if user_json.get("USER_ID") == username:
            return user_json
    return None


def generate_signed_url(key, expires_in=3600):
    return s3.generate_presigned_url(
        'get_object',
        Params={'Bucket': BUCKET_NAME, 'Key': key},
        ExpiresIn=expires_in
    )

def put_json_to_s3(key, data):
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=json.dumps(data, ensure_ascii=False).encode('utf-8'),
        ContentType='application/json'
    )

def list_s3_objects(prefix):
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
    return response.get('Contents', [])

def get_user_recommended_images_and_areas(username):
    try:
        # # 1. 사용자 JSON 불러오기
        # user_data = get_user_info(username)
        # if not user_data:
        #     raise Exception("사용자 정보 없음")

        # # 2. EC2에 유저 정보 전송 → 추천 travel_id 리스트 응답
        # res = requests.post(EC2_API_URL, json=user_data)
        # if res.status_code != 200:
        #     raise Exception("EC2 추천 요청 실패")

        # recommended_ids = res.json().get("recommended_travel_ids", [])
        # if not recommended_ids:
        #     raise Exception("추천 travel_id 없음")

        # test
        recommended_ids = [
            'e_e000004', 'e_e000021', 'e_e000033',
            'e_e000057', 'e_e000060', 'e_e000065',
            'e_e000088', 'e_e000092', 'e_e000099',
            'e_e000102'
        ]

        # 3. RDS 연결
        conn = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            db=DB_NAME,
            port = DB_PORT,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )

        with conn.cursor() as cursor:
            # 4. place_info → meta_photo 조인 쿼리
            format_str = ','.join(['%s'] * len(recommended_ids))
            sql = f"""                
                SELECT mp.PHOTO_FILE_NM, pi.VISIT_AREA_NM
                FROM place_info pi
                JOIN meta_photo mp
                ON pi.VISIT_AREA_ID = mp.VISIT_AREA_ID AND pi.TRAVEL_ID = mp.TRAVEL_ID
                WHERE (pi.TRAVEL_ID, pi.VISIT_AREA_ID) IN (
                    SELECT TRAVEL_ID, MIN(VISIT_AREA_ID)
                    FROM place_info
                    WHERE TRAVEL_ID IN (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    GROUP BY TRAVEL_ID
                );
            """
            cursor.execute(sql, recommended_ids)
            results = cursor.fetchall()
        conn.close()

        # 5. presigned URL 생성
        prefix = "data/resized_image/E/"
        image_infos = []
        for row in results:
            file_name = row["PHOTO_FILE_NM"]            
            area_name = row["VISIT_AREA_NM"]
            url = s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': BUCKET_NAME, 'Key': f"{prefix}{file_name}"},
                ExpiresIn=3600
            )
            image_infos.append({"url": url, "area": area_name})
          
        return image_infos
    
    except Exception as e:
        print("DB_HOST:", DB_HOST)

        print(f"[!] 추천 이미지 처리 오류: {str(e)}")
        return []

