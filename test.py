# import pymysql
# from dotenv import load_dotenv
# import os
# import pandas as pd

# load_dotenv(override=True)

# DB_HOST = os.getenv("RDB_HOST")
# DB_USER = os.getenv("RDB_USER")
# DB_PASSWORD = os.getenv("RDB_PASSWORD")
# DB_NAME = os.getenv("RDB_NAME")
# DB_PORT = int(os.getenv("RDB_PORT"))

# style_cols = ['TRAVEL_STYL_1',
#             'TRAVEL_STYL_2',
#             'TRAVEL_STYL_3',
#             'TRAVEL_STYL_4',
#             'TRAVEL_STYL_5',
#             'TRAVEL_STYL_6',
#             'TRAVEL_STYL_7',
#             'TRAVEL_STYL_8']

# conn = pymysql.connect(
#             host=DB_HOST,
#             user=DB_USER,
#             password=DB_PASSWORD,
#             db=DB_NAME,
#             port = DB_PORT,
#             charset='utf8mb4',
#             cursorclass=pymysql.cursors.DictCursor
#         )
# cursor = conn.cursor()

# sql = "SELECT * FROM users LIMIT 10;"
# cursor.execute(sql)

# results = cursor.fetchall()

# for row in results:
#     print(row)

# # 7. 연결 종료
# cursor.close()
# conn.close()
import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()

# 환경 변수에서 DB 접속 정보 불러오기
DB_USER = os.getenv("RDB_USER")
DB_PASSWORD = os.getenv("RDB_PASSWORD")
DB_HOST = os.getenv("RDB_HOST")
DB_PORT = os.getenv("RDB_PORT")
DB_NAME = os.getenv("RDB_NAME")

# SQLAlchemy 엔진 생성
engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# SQL 실행 → 데이터프레임으로 읽기
df = pd.read_sql("SELECT * FROM users", con=engine)

# 결과 확인
print(df.head())
