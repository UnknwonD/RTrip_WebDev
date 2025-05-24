from config import s3, BUCKET_NAME

def extract_travel_styles(form_or_session, default=4):
    return [
        min(max(1, int(form_or_session.get(f"TRAVEL_STYL_{i}", default))), 7)
        for i in range(1, 9)
    ]

def get_presigned_image_urls(photo_df):
    result = []
    for _, row in photo_df.iterrows():
        try:
            url = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": BUCKET_NAME, "Key": f"rtrip/images/{row['PHOTO_FILE_NM']}"},
                ExpiresIn=3600
            )
            result.append({
                "url": url,
                "area": row["VISIT_AREA_NM"]
            })
        except Exception as e:
            print(f"[ERROR] URL 생성 실패: {str(e)}")
            continue
    return result