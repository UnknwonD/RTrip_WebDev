import json
import random
from config import s3, BUCKET_NAME
from datetime import datetime 
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

# 모든 여행 일정 불러오기 (최신순)
def load_all_travel_plans(user_id):
    prefix = f"user_travel_plans/{user_id}/"
    try:
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
        contents = response.get("Contents", [])
        if not contents:
            return []

        def extract_datetime_from_key(obj):
            key = obj["Key"]
            try:
                filename = key.split("/")[-1].replace(".json", "")
                return datetime.strptime(filename.split("__")[-1], "%Y%m%d_%H%M%S")
            except Exception:
                return datetime.min

        sorted_keys = sorted(contents, key=extract_datetime_from_key, reverse=True)

        travel_plans = []
        for obj in sorted_keys:
            key = obj["Key"]
            content = s3.get_object(Bucket=BUCKET_NAME, Key=key)["Body"].read().decode("utf-8")
            plan_data = json.loads(content)
            travel_plans.append({
                "key": key.split("/")[-1].replace(".json", ""),
                "data": plan_data,
                "saved_at": extract_datetime_from_key(obj).strftime("%Y-%m-%d %H:%M:%S"),
                "title": plan_data.get("meta", {}).get("custom_title", plan_data.get("0", {}).get("title", "제목 없음"))
            })

        return travel_plans

    except Exception as e:
        print(f"[ERROR] 전체 여행 일정 불러오기 실패: {e}")
        return []