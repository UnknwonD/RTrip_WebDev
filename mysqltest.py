import pymysql
import os
from dotenv import load_dotenv

load_dotenv(override=True)

conn = None
try:
    conn = pymysql.connect(
        host=os.getenv("RDS_HOST"),
        user=os.getenv("RDS_USER"),
        password=os.getenv("RDS_PASSWORD"),
        db=os.getenv("RDS_DB_NAME"),
        charset='utf8mb4'
    )
    with conn.cursor() as cursor:
        cursor.execute("SELECT VERSION();")
        result = cursor.fetchone()
        print(f"MySQL 연결 성공 ✅ - 버전: {result[0]}")

except Exception as e:
    print(f"[!] MySQL 연결 실패 ❌ : {e}")

finally:
    if conn:
        conn.close()
