import json
import joblib
import faiss
import pandas as pd
import numpy as np

categorical_cols = [
    'GENDER', 'EDU_NM', 'EDU_FNSH_SE', 'MARR_STTS', 'JOB_NM',
    'HOUSE_INCOME', 'TRAVEL_TERM',
    'TRAVEL_LIKE_SIDO_1', 'TRAVEL_LIKE_SIDO_2', 'TRAVEL_LIKE_SIDO_3',
]

numerical_cols = [
    'AGE_GRP', 'FAMILY_MEMB', 'TRAVEL_NUM', 'TRAVEL_COMPANIONS_NUM',
    'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4',
    'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
    'TRAVEL_MOTIVE_1', 'TRAVEL_MOTIVE_2',
    'INCOME'
]

def find_nearest_neighbors(user, k=5):
    # 여기서 필요한 데이터 정보들
    df = pd.read_csv("../data/VL_csv/tn_traveller_master_여행객 Master_E.csv")
    travel = pd.read_csv('../data/VL_CSV/tn_travel_여행_E.csv')
    area = pd.read_csv('../data/VL_CSV/tn_visit_area_info_방문지정보_Cleaned_E.csv')
    photo = pd.read_csv('../data/VL_csv/tn_tour_photo_관광사진_E.csv')
    sgg = pd.read_csv('../data/VL_csv/tc_sgg_시군구코드.csv')
    
    use_cols = categorical_cols + numerical_cols
    
    filtered_user = pd.DataFrame({key: user[key] if key not in numerical_cols else int(user[key]) for key in use_cols if key in user})
    pipeline = joblib.load('tn_traveller_pipeline.pkl')
    
    filtered_user.loc[0, 'TRAVEL_LIKE_SIDO_1'] = sgg[sgg['SIDO_NM'] == filtered_user.loc[0, 'TRAVEL_LIKE_SIDO_1']]['SGG_CD1'].values[0]
    filtered_user.loc[0, 'TRAVEL_LIKE_SIDO_2'] = sgg[sgg['SIDO_NM'] == filtered_user.loc[0, 'TRAVEL_LIKE_SIDO_2']]['SGG_CD1'].values[0]
    filtered_user.loc[0, 'TRAVEL_LIKE_SIDO_3'] = sgg[sgg['SIDO_NM'] == filtered_user.loc[0, 'TRAVEL_LIKE_SIDO_3']]['SGG_CD1'].values[0]
    
    input_vec = pipeline.named_steps['preprocess'].transform(filtered_user)
    
    X_all = joblib.load('../data/VL_CSV/X_all.pkl').astype('float32') # 모든 사용자의 벡터 (로컬에 저장해두고 사용할 것)
    input_vec = input_vec.astype('float32')

    d = X_all.shape[1]  # feature 수
    index = faiss.IndexFlatL2(d)  # L2 거리 기반

    # 인덱스에 전체 유저 벡터 추가
    index.add(X_all)

    # 유사 유저 top 10 검색
    k = 10
    D, I = index.search(input_vec, k)  # D: 거리, I: 인덱스

    # 결과 출력
    print("Top 10 유사 유저 인덱스:", I[0])
    print("Top 10 거리:", D[0])
    
    similar_users = df.iloc[I[0]]
    
    travel = travel[travel['TRAVELER_ID'].isin(similar_users['TRAVELER_ID'])]

    travel_ids = travel['TRAVEL_ID'].to_list()
    filter_area = ['집', '사무실', '학교', '기타']

    area = area[(area['TRAVEL_ID'].isin(travel_ids)) & (~area['VISIT_AREA_NM'].isin(filter_area))]
    
    cond = photo['VISIT_AREA_ID'].isin(area['VISIT_AREA_ID'].to_list()[:10])

    photo = photo[cond]
    
    return photo

# SELECT VISIT_AREA_ID

# SELECT DISTINCT(VISIT_AREA_ID)
# FROM meta_photo


# from

# 유저 travel id ( travel 과 엮기)
# travel id -> place_info ID 엮고 -> place_info 에서 visit_area_id (dist)
# metaphoto -> travel_id , distinct(visit_area_id)

