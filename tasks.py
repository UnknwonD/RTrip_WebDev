import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from celery import Celery

celery_app = Celery(
    'rtrip',
    broker='redis://redis:6379/0',     # 작업 요청을 보내는 곳
    backend='redis://redis:6379/0'     # 작업 결과를 받는 곳 (보내줄 곳)
)

# 로컬 테스트용
# celery_app = Celery(
#     'rtrip',
#     broker='redis://localhost:6379/0',
#     backend='redis://localhost:6379/0'
# )

celery = celery_app

import numpy as np

def convert_to_serializable(obj):
    """numpy 타입을 JSON 직렬화 가능한 타입으로 변환"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        return obj

@celery_app.task
def run_gnn(travel_input):
    import uuid, torch
    from gnn_module import main_optimized_test
    from extract_latlng import fill_missing_coords_with_kakao
    from modules.rds_utils import travel_plans_with_debug
    
    region_map = {'eastern' : '동부권',
     'western' : '서부권',
     'capital' : '수도권',
     'jeju' : '제주도'}
    
    target_region = region_map[travel_input['selected_region']]

    route, recommender, unique_recommendations, travel_context_tensor = main_optimized_test(travel_input, target_region)
    
    plan_id = str(uuid.uuid4())[:8]
    
    dummy_ids = [[d['id'] for d in v] for k, v in route.items()]
    travel_plan_list = travel_plans_with_debug(dummy_ids, travel_input['date_range'])
    filled_plan_list = fill_missing_coords_with_kakao(travel_plan_list)
    
    # 모든 데이터를 JSON 직렬화 가능한 형태로 변환
    travel_context_tensor = travel_context_tensor.cpu().detach().numpy().tolist()
    route = convert_to_serializable(route)
    unique_recommendations = convert_to_serializable(unique_recommendations)
    filled_plan_list = convert_to_serializable(filled_plan_list)

    return {
        'plan_id': plan_id,
        'travel_plan_list': filled_plan_list,
        'route': route,
        'unique_recommendations': unique_recommendations,
        'travel_context_tensor': travel_context_tensor,
        'target_region': target_region
    }