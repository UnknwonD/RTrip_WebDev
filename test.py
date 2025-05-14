import boto3
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

# 환경변수 로딩
load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
REGION_NAME = os.getenv("AWS_S3_REGION")

# S3 설정
prefix = 'resized_image/E/dpipata_resized/'

# boto3 S3 클라이언트 생성
s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=REGION_NAME
)

# S3 객체 목록 가져오기
response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)

# 0005.jpg로 끝나는 파일 필터링
target_files = [
    obj['Key'] for obj in response.get('Contents', [])
    if obj['Key'].endswith('0005.jpg')
]

# 출력
print("✅ 0005.jpg로 끝나는 파일 목록:")
for key in target_files:
    print(f"- {key}")

# ✅ 첫 번째 이미지 시각화
if target_files:
    key = target_files[0]
    print(f"\n🖼️ 시각화할 이미지: {key}")

    # S3에서 이미지 로딩
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
    img_data = obj['Body'].read()

    # 이미지 시각화
    image = Image.open(BytesIO(img_data))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"S3 Image: {os.path.basename(key)}")
    plt.show()
else:
    print("❌ '0005.jpg'로 끝나는 파일이 없습니다.")
