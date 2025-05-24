import pickle
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear

user_feature_keys = [
    'GENDER', 'EDU_NM', 'EDU_FNSH_SE', 'MARR_STTS', 'JOB_NM', 'HOUSE_INCOME',
    'TRAVEL_TERM', 'TRAVEL_LIKE_SIDO_1', 'TRAVEL_LIKE_SIDO_2', 'TRAVEL_LIKE_SIDO_3',
    'AGE_GRP', 'FAMILY_MEMB', 'TRAVEL_NUM', 'TRAVEL_COMPANIONS_NUM',
    'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4',
    'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
    'TRAVEL_MOTIVE_1', 'TRAVEL_MOTIVE_2', 'INCOME'
]

travel_feature_keys = [
    'LODGOUT_COST', 'ACTIVITY_COST', 'TOTAL_COST', 'DURATION', 'PURPOSE_1',
    'PURPOSE_10', 'PURPOSE_11', 'PURPOSE_12', 'PURPOSE_13', 'PURPOSE_2',
    'PURPOSE_21', 'PURPOSE_22', 'PURPOSE_23', 'PURPOSE_24', 'PURPOSE_25',
    'PURPOSE_26', 'PURPOSE_27', 'PURPOSE_28', 'PURPOSE_3', 'PURPOSE_4',
    'PURPOSE_5', 'PURPOSE_6', 'PURPOSE_7', 'PURPOSE_8', 'PURPOSE_9',
    'MVMN_NM_ENC', 'age_ENC', 'whowith_ENC', 'mission_ENC'
]

# 다시 변수 정의 (환경 리셋됨)
# purpose_options = [
#     (1, "쇼핑"), (2, "테마파크, 놀이시설, 동/식물원 방문"), (3, "역사 유적지 방문"),
#     (4, "시티투어"), (5, "야외 스포츠, 레포츠 활동"), (6, "지역 문화예술/공연/전시시설 관람"),
#     (7, "유흥/오락(나이트라이프)"), (8, "캠핑"), (9, "지역 축제/이벤트 참가"),
#     (10, "온천/스파"), (11, "교육/체험 프로그램 참가"), (12, "드라마 촬영지 방문"),
#     (13, "종교/성지 순례"), (21, "Well-ness 여행"), (22, "SNS 인생샷 여행"),
#     (23, "호캉스 여행"), (24, "신규 여행지 발굴"), (25, "반려동물 동반 여행"),
#     (26, "인플루언서 따라하기 여행"), (27, "친환경 여행(플로깅 여행)"), (28, "등반 여행")
# ]

purpose_options =[
    (1, "🛍️ 쇼핑 & 트렌드 탐방"), (2, "🏛️ 문화·예술·역사 체험"), (3, "🎢 테마파크 & 놀이시설"), (4, "🏙️ 도심 여행 & 휴식" ),
    (5, "🏕️ 아웃도어 & 액티비티"), (6, "♨️ 온천 & 힐링 여행"), (7, "📸 SNS 핫플 & 인생샷"), (8, "✨ 신규 & 미개척 지역 탐방"),
    (9, "🌿 친환경 & 지속가능한 여행")
]

# 이름 변경
movement_options = [
    (1, "자가용"),
    (2, "대중교통"),
    (3, "기타 이동수단")
]

# 다시 변수 정의 (환경 리셋됨)
# whowith_options = [
#     ("커플 (2인 여행)", "커플"), ("나홀로 여행", "나홀로 여행"),
#     ("자녀동반", "자녀동반"), ("부부", "부부"), ("3인 이상 친구", "3인 이상 친구"),
#     ("부모 동반", "부모 동반"), ("3대 동반 여행", "3대 동반 여행")
# ]

# whowith_options = [
#     ("단독여행", "나홀로 여행"),
#     ("2인여행", "커플"),
#     ("2인여행", "부부"),
#     ("가족여행" , "자녀동반"),
#     ("가족여행", "부모 동반"),
#     ("가족여행", "3대 동반 여행"),
#     ("친구/지인 여행", "3인 이상 친구"),
#     ("기타", "기타")
# ]

whowith_options = [
    ("단독여행", ["나홀로 여행"]),
    ("2인여행", ["커플", "부부"]),
    ("가족여행", ["자녀동반", "부모 동반", "3대 동반 여행"]),
    ("친구/지인 여행", ["3인 이상 친구"]),
    ("기타", ["기타"])
]


class RouteGNN(nn.Module):
    def __init__(self, metadata, hidden_channels=128):
        super().__init__()
        self.metadata = metadata

        self.embeddings = nn.ModuleDict({
            'user': Linear(17, hidden_channels),
            'travel': Linear(21, hidden_channels),
            'visit_area': Linear(34, hidden_channels),
        })

        self.gnn1 = HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden_channels)
            for edge_type in metadata[1]
        }, aggr='sum')

        self.gnn2 = HeteroConv({
            edge_type: SAGEConv((hidden_channels, hidden_channels), hidden_channels)
            for edge_type in metadata[1]
        }, aggr='sum')

        self.link_predictor = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            node_type: self.embeddings[node_type](x) if x is not None else None
            for node_type, x in x_dict.items()
        }

        x_dict = self.gnn1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items() if v is not None}
        x_dict = self.gnn2(x_dict, edge_index_dict)

        return x_dict

    def predict_link(self, node_embed, edge_index):
        src, dst = edge_index
        z_src = node_embed[src]
        z_dst = node_embed[dst]
        z = torch.cat([z_src, z_dst], dim=-1)
        return self.link_predictor(z).squeeze(-1)
    
# 유저 정보

def get_age_group(birthdate_str):
    """
    'YYYY-MM-DD' 형식의 생년월일 문자열을 받아
    20, 30, 40 등의 나이대로 변환하는 함수
    """
    from datetime import datetime
    
    birth_year = int(birthdate_str[:4])
    current_year = datetime.now().year
    age = current_year - birth_year + 1  # 한국식 나이
    age_group = (age // 10) * 10
    return age_group

def map_sido(sido:str):
    sido_code_map = {
        '서울특별시': '11',
        '부산광역시': '26',
        '대구광역시': '27',
        '인천광역시': '28',
        '광주광역시': '29',
        '대전광역시': '30',
        '울산광역시': '31',
        '세종특별자치시': '36',
        '경기도': '41',
        '강원도': '42',
        '충청북도': '43',
        '충청남도': '44',
        '전라북도': '45',
        '전라남도': '46',
        '경상북도': '47',
        '경상남도': '48',
        '제주특별자치도': '50'
    }

    return int(sido_code_map[sido])

def process_user_input(user_info:dict):
    user_feature_cols = [
    'GENDER', 'TRAVEL_TERM', 'TRAVEL_NUM',
    'TRAVEL_LIKE_SIDO_1', 'TRAVEL_LIKE_SIDO_2', 'TRAVEL_LIKE_SIDO_3',
    'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4',
    'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
    'TRAVEL_MOTIVE_1', 'TRAVEL_MOTIVE_2',
    'AGE_GRP'
    ]
    
    # 1. 나잇대 계산
    user_info['AGE_GRP'] = get_age_group(user_info['BIRTHDATE'])
    
    # 2. 시도 변환
    for i in range(1, 4):
        user_info[f"TRAVEL_LIKE_SIDO_{i}"] = map_sido(user_info[f"TRAVEL_LIKE_SIDO_{i}"])
    
    # 3. 컬럼 필터링 (순서에 맞게)
    user_info = {k: int(user_info[k]) for k in user_feature_cols}
    
    return pd.DataFrame([user_info]).fillna(0).astype(np.float32).to_numpy()    

# 여행 정보
def process_travel_input(travel_info:dict):
    from datetime import datetime
    travel_feature_cols = [
        'TOTAL_COST_BINNED_ENCODED',
        'WITH_PET',
        'MONTH',
        'DURATION',
        'MVMN_기타',
        'MVMN_대중교통',
        'MVMN_자가용',
        'TRAVEL_PURPOSE_1',
        'TRAVEL_PURPOSE_2',
        'TRAVEL_PURPOSE_3',
        'TRAVEL_PURPOSE_4',
        'TRAVEL_PURPOSE_5',
        'TRAVEL_PURPOSE_6',
        'TRAVEL_PURPOSE_7',
        'TRAVEL_PURPOSE_8',
        'TRAVEL_PURPOSE_9',
        'WHOWITH_2인여행',
        'WHOWITH_가족여행',
        'WHOWITH_기타',
        'WHOWITH_단독여행',
        'WHOWITH_친구/지인 여행']
    
    
    # mission_ENC에 0 = 반려동물 동반 (WITH_PET)
    travel_info['mission_ENC'] = travel_info['mission_ENC'].strip().split(',')
    if '0' in travel_info['mission_ENC']:
        travel_info['WITH_PET'] = 1
    else:
        travel_info['WITH_PET'] = 0
        
    # TRAVEL_PURPOSE_1 ~~ TRAVEL_PURPOSE_9 (0으로 들어온 입력은 제거해줘야됨) 
    for i in range(1,10):
        if str(i) in travel_info['mission_ENC']:
            travel_info[f'TRAVEL_PURPOSE_{i}'] = 1
        else:
            travel_info[f'TRAVEL_PURPOSE_{i}'] = 0
        
    # MONTH
    dates = travel_info['date_range'].split(' - ')
    travel_info['start_date'] = datetime.strptime(dates[0].strip(), "%Y-%m-%d")
    travel_info['end_date'] = datetime.strptime(dates[1].strip(), "%Y-%m-%d")
    
    travel_info['MONTH'] = travel_info['end_date'].month
    
    # DURATION
    travel_info['DURATION'] = (travel_info['end_date'] - travel_info['start_date']).days
    
    # MNVM_기타, MVMN_대중교통, MVMN_자가용
    for m in ['자가용', '대중교통', '기타']:
        travel_info[f"MVMN_{m}"] = False
    
    if travel_info['MVMN_NM_ENC'] == '1':
        travel_info['MVMN_자가용'] = True
    elif travel_info['MVMN_NM_ENC'] == '2':
        travel_info['MVMN_대중교통'] = True
    else:
        travel_info['MVMN_기타'] = True
    
    # WHOWITH는 1부터 5까지 숫자로 들어옴 -> 원핫 인코딩으로 수정할 것
    # dict에 들어오는 숫자 의미: WHOWITH_단독여행, WHOWITH_2인여행, WHOWITH_가족여행, WHOWITH_친구/지인여행, WHOWITH_기타
    whowith_onehot = [0] * 5
    idx = int(travel_info['whowith_ENC']) - 1
    if 0 <= idx < 5:
        whowith_onehot[idx] = 1
    
    travel_info.update({
    'WHOWITH_단독여행': whowith_onehot[0],
    'WHOWITH_2인여행': whowith_onehot[1],
    'WHOWITH_가족여행': whowith_onehot[2],
    'WHOWITH_친구/지인 여행': whowith_onehot[3],
    'WHOWITH_기타': whowith_onehot[4],
    })
    
    # TOTAL_COST_BINNED_ENCODED
    travel_info['TOTAL_COST_BINNED_ENCODED'] = travel_info['TOTAL_COST'][-1]
    
    # 컬럼 필터링 (순서에 맞게)
    travel_info = {k: int(travel_info[k]) for k in travel_feature_cols}
    
    return pd.DataFrame([travel_info]).fillna(0).astype(np.float32).to_numpy(), travel_info['DURATION']

def recommend_route(node_embed, edge_index, edge_scores, start_node=None, max_steps=5):
    """
    visit_area 노드 임베딩, 엣지 index, score가 주어졌을 때
    가장 높은 score 기준으로 동선을 구성하는 greedy 경로 추천 함수
    """
    from collections import defaultdict

    # 엣지를 점수 기준으로 정렬
    scored_edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist(), edge_scores.tolist()))
    scored_edges.sort(key=lambda x: -x[2])  # 높은 점수 순

    # 경로 생성
    visited = set()
    route = []

    current = start_node if start_node is not None else scored_edges[0][0]
    visited.add(current)
    route.append(current)

    for _ in range(max_steps - 1):
        # current에서 시작하는 후보 중 아직 방문하지 않은 곳
        candidates = [dst for src, dst, score in scored_edges if src == current and dst not in visited]
        if not candidates:
            break
        next_node = candidates[0]  # greedy하게 최고 점수 선택
        visited.add(next_node)
        route.append(next_node)
        current = next_node

    return route  # index 형태


def infer_route(model, data, user_input, travel_input, k=5, device='cpu', batch_size=1000000):
    model.eval()
    data = data.to(device)
    user_input = user_input.to(device)
    travel_input = travel_input.to(device)

    with torch.no_grad():
        # 유저/여행 feature + 기존 raw feature 합치기
        x_dict_raw = {
            'user': torch.cat([data['user'].x, user_input], dim=0),       # [N+1, 17]
            'travel': torch.cat([data['travel'].x, travel_input], dim=0), # [M+1, 21]
            'visit_area': data['visit_area'].x                             # [V, feature_dim]
        }

        # 모델 forward
        x_dict = model(x_dict_raw, data.edge_index_dict)
        visit_area_embed = x_dict['visit_area']

        # 모든 visit_area 노드 쌍 조합 (너무 많으면 메모리 폭발!)
        n = visit_area_embed.size(0)
        all_edges = torch.combinations(torch.arange(n, device=device), r=2).t()

        # batch-wise로 score 계산 (메모리 폭발 방지)
        def predict_link_batch(node_embed, all_edges, batch_size=1000000):
            from tqdm import tqdm
            scores = []
            for i in tqdm(range(0, all_edges.size(1), batch_size)):
                batch_edges = all_edges[:, i:i+batch_size]
                batch_scores = model.predict_link(node_embed, batch_edges)
                scores.append(batch_scores)
            return torch.cat(scores, dim=0)

        edge_scores = predict_link_batch(visit_area_embed, all_edges, batch_size)

        # 경로 구성 (Greedy 방식)
        route = recommend_route(visit_area_embed, all_edges, edge_scores, max_steps=k)

    return route

def select_best_location_by_distance(route_ids, visit_area_df):
    selected_names = []

    for idx, vid in enumerate(route_ids):
        candidates = visit_area_df[visit_area_df['VISIT_AREA_ID'] == vid]

        # 후보가 하나일 경우 바로 선택
        if len(candidates) == 1:
            selected_names.append(candidates.iloc[0]['VISIT_AREA_NM'])
            continue

        # 이전/다음 위치 좌표 확보
        prev_coord = None
        next_coord = None

        if idx > 0:
            prev_id = route_ids[idx - 1]
            prev_row = visit_area_df[visit_area_df['VISIT_AREA_ID'] == prev_id]
            if not prev_row.empty:
                prev_coord = (prev_row.iloc[0]['X_COORD'], prev_row.iloc[0]['Y_COORD'])

        if idx < len(route_ids) - 1:
            next_id = route_ids[idx + 1]
            next_row = visit_area_df[visit_area_df['VISIT_AREA_ID'] == next_id]
            if not next_row.empty:
                next_coord = (next_row.iloc[0]['X_COORD'], next_row.iloc[0]['Y_COORD'])

        # 거리 계산 함수
        def total_distance(row):
            x, y = row['X_COORD'], row['Y_COORD']
            dist = 0
            if prev_coord:
                dist += np.linalg.norm(np.array([x, y]) - np.array(prev_coord))
            if next_coord:
                dist += np.linalg.norm(np.array([x, y]) - np.array(next_coord))
            return dist

        # 최단 거리 후보 선택
        best_row = candidates.loc[candidates.apply(total_distance, axis=1).idxmin()]
        selected_names.append(best_row['VISIT_AREA_NM'])

    return selected_names


#############################################
#
#           모델 추론 함수
#
#############################################
def run_inference(user_info, travel_info, model, data, visit_area_id_to_index, visit_area_df):
    
    user_tensor = process_user_input(user_info)
    travel_tensor, duration = process_travel_input(travel_info)
    
    user_input = torch.tensor(user_tensor, dtype=torch.float)  # 17차원
    travel_input = torch.tensor(travel_tensor, dtype=torch.float)  # 21차원
    
    route_indices = infer_route(model, data, user_input, travel_input, k=(8 * duration))
    
    index_to_id = {v: k for k, v in visit_area_id_to_index.items()}
    route_ids = [index_to_id[idx] for idx in route_indices]
    
    names = select_best_location_by_distance(route_ids, visit_area_df)
    
    return route_ids, names