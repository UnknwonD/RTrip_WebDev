# import faiss
# import joblib
# import numpy as np
# import pandas as pd
# from modules.s3_utils import get_user_recommended_images_and_areas
# import boto3
# import pymysql



# style_cols = ['TRAVEL_STYL_1',
#             'TRAVEL_STYL_2',
#             'TRAVEL_STYL_3',
#             'TRAVEL_STYL_4',
#             'TRAVEL_STYL_5',
#             'TRAVEL_STYL_6',
#             'TRAVEL_STYL_7',
#             'TRAVEL_STYL_8']

# def find_nearest_users(input_vec, k=5):
#     # 불러와야되는 정보 (pymysql로 df로 불러와야됨)
#     user = pd.read_csv("./data/tn_traveller_master_여행객 Master_E.csv") # 다불러오셈

#     style_df = user[style_cols]
#     style_array = style_df.to_numpy().astype('float32')
    
#     input_vec = np.array(input_vec, dtype='float32')
    
#     d = style_array.shape[1]
#     index = faiss.IndexFlatL2(d)
#     index.add(style_array)
#     D, I = index.search(input_vec, k)
    
#     # 유사 유저의 여행 정보 추출
#     similar_users = user.iloc[I[0]]

#     ############################################################################################################
#     # 여기서부터 S3 불러오기 필요함
#     travel = pd.read_csv('./data/tn_travel_여행_E.csv') # 1번째 필터링하면 됨 - user정보에서 traveler_id가져올거임 where절에 넣어서
#     travel = travel[travel['TRAVELER_ID'].isin(similar_users['TRAVELER_ID'])] ####### 여기를 travel 데이터 필터링
#     travel_ids = travel['TRAVEL_ID'].to_list()
    
#     return get_user_recommended_images_and_areas(travel_ids)
