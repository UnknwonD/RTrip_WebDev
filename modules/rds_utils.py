import pymysql
import random
import os
from dotenv import load_dotenv

load_dotenv(override=True)

DB_HOST = os.getenv("RDS_HOST")
DB_USER = os.getenv("RDS_USER")
DB_PASSWORD = os.getenv("RDS_PASSWORD")
DB_NAME = os.getenv("RDS_DB_NAME")
DB_PORT = int(os.getenv("RDS_PORT"))

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
