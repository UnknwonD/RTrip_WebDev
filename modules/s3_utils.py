import json
import random
from config import s3, BUCKET_NAME

# S3에서 key로 파일을 읽고 JSON으로 반환

def get_json_from_s3(key):
    try:
        file_obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        body = file_obj['Body'].read().decode('utf-8').strip()

        if not body:
            print(f"[SKIP] 빈 JSON 파일: {key}")
            return None

        return json.loads(body)

    except Exception as e:
        print(f"[ERROR] JSON 파싱 실패: {key} → {str(e)}")
        return None


# prefix로 시작하는 S3 객체 리스트 반환

def list_s3_objects(prefix):
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
    return [obj for obj in response.get('Contents', []) if obj['Key'].endswith(".json")]


# key에 대해 1시간짜리 presigned URL 생성

def get_s3_signed_urls():
    prefix = 'data/resized_image/E/'
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)

    all_keys = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.jpg')]
    selected_keys = random.sample(all_keys, k=min(10, len(all_keys)))

    signed_urls = [
        s3.generate_presigned_url('get_object', Params={'Bucket': BUCKET_NAME, 'Key': key}, ExpiresIn=3600)
        for key in selected_keys
    ]

    return [{"url": url, "area": ""} for url in signed_urls]


# S3에 JSON 저장
def put_json_to_s3(key, data):
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=json.dumps(data, ensure_ascii=False).encode('utf-8'),
        ContentType='application/json'
    )