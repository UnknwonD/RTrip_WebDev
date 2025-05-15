from typing import List, Dict
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import HeteroData
from typing import Dict, List

import pickle
from datetime import datetime
import pandas as pd
import numpy as np

# app.py 안에서 템플릿에 넘겨줄 리스트
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
purpose_options = [
    (1, "쇼핑"), (2, "테마파크, 놀이시설, 동/식물원 방문"), (3, "역사 유적지 방문"),
    (4, "시티투어"), (5, "야외 스포츠, 레포츠 활동"), (6, "지역 문화예술/공연/전시시설 관람"),
    (7, "유흥/오락(나이트라이프)"), (8, "캠핑"), (9, "지역 축제/이벤트 참가"),
    (10, "온천/스파"), (11, "교육/체험 프로그램 참가"), (12, "드라마 촬영지 방문"),
    (13, "종교/성지 순례"), (21, "Well-ness 여행"), (22, "SNS 인생샷 여행"),
    (23, "호캉스 여행"), (24, "신규 여행지 발굴"), (25, "반려동물 동반 여행"),
    (26, "인플루언서 따라하기 여행"), (27, "친환경 여행(플로깅 여행)"), (28, "등반 여행")
]

movement_options = [
    (1, "자가용/렌트카/캠핑카 (직접 운전)"),
    (2, "대중교통(택시, 지하철 등)"),
    (3, "기타 이동수단")
]

whowith_options = [
    ("커플 (2인 여행)", "커플"), ("나홀로 여행", "나홀로 여행"),
    ("자녀동반", "자녀동반"), ("부부", "부부"), ("3인 이상 친구", "3인 이상 친구"),
    ("부모 동반", "부모 동반"), ("3대 동반 여행", "3대 동반 여행")
]

class PpiKkoTwistGNN(nn.Module):  # 삐삐꼬는 GNN
    def __init__(self, metadata, user_input_dim, travel_input_dim, hidden_dim=128, num_layers=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input Projections
        self.input_proj = nn.ModuleDict({
            'user': nn.Linear(user_input_dim, hidden_dim),
            'travel': nn.Linear(travel_input_dim, hidden_dim),
            'visit_area': nn.Identity()
        })

        # Deep HeteroConv Layers
        self.convs = nn.ModuleList([
            HeteroConv(
                {etype: SAGEConv((-1, -1), hidden_dim) for etype in metadata[1]},
                aggr='sum'
            ) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.ModuleDict({ntype: nn.LayerNorm(hidden_dim) for ntype in metadata[0]})
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(0.35)

        # Multi-Expert System (location / preference / category)
        self.expert_location = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.expert_preference = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.expert_category = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Multihead Attention Gating: attend across experts
        self.attn_gate = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.attn_query = nn.Parameter(torch.randn(1, hidden_dim))

        self.final_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_dict, edge_index_dict, feedback_mask=None):
        # 1. Input projection
        x_dict = {k: self.input_proj[k](v) if k in self.input_proj else v for k, v in x_dict.items()}

        # 2. Deep GNN layers with residuals
        for i in range(self.num_layers):
            h_dict = self.convs[i](x_dict, edge_index_dict)
            h_dict = {
                k: self.dropout(F.relu(self.norms[i][k](v))) + x_dict[k]
                for k, v in h_dict.items() if k in x_dict
            }
            x_dict = h_dict

        h_visit = x_dict['visit_area']  # [num_nodes, hidden_dim]

        # 3. Expert Predictions
        loc = self.expert_location(h_visit)         # [N, 1]
        pref = self.expert_preference(h_visit)      # [N, 1]
        cat = self.expert_category(h_visit)         # [N, 1]
        experts = torch.cat([loc, pref, cat], dim=1).unsqueeze(1)  # [N, 1, 3]

        # 4. Multi-head Attention Gating
        q = self.attn_query.expand(h_visit.size(0), -1).unsqueeze(1)  # [N, 1, H]
        attn_out, _ = self.attn_gate(q, h_visit.unsqueeze(1), h_visit.unsqueeze(1))  # [N, 1, H]
        final_score = self.final_proj(attn_out.squeeze(1)).squeeze(-1)  # [N]

        if feedback_mask is not None:
            final_score = final_score + feedback_mask

        return final_score

class LiteTwistGNN(nn.Module):
    def __init__(self, metadata, user_input_dim, travel_input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.ModuleDict({
            'user': nn.Linear(user_input_dim, hidden_dim),
            'travel': nn.Linear(travel_input_dim, hidden_dim),
            'visit_area': nn.Identity()  # zero-init at inference
        })

        # Lightweight HeteroConv stack (2 layers)
        self.convs = nn.ModuleList([
            HeteroConv(
                {etype: SAGEConv((-1, -1), hidden_dim) for etype in metadata[1]},
                aggr='sum'
            ) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.ModuleDict({ntype: nn.LayerNorm(hidden_dim) for ntype in metadata[0]})
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(0.25)

        # Unified MLP scorer
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_dict, edge_index_dict, feedback_mask=None):
        # Projection
        x_dict = {
            k: self.input_proj[k](v) if k in self.input_proj else v
            for k, v in x_dict.items()
        }

        # GNN Layers
        for i in range(self.num_layers):
            h_dict = self.convs[i](x_dict, edge_index_dict)
            h_dict = {
                k: self.dropout(F.relu(self.norms[i][k](v))) + x_dict[k]
                for k, v in h_dict.items() if k in x_dict
            }
            x_dict = h_dict

        h_visit = x_dict['visit_area']
        out = self.mlp(h_visit).squeeze(-1)

        if feedback_mask is not None:
            out = out + feedback_mask

        return out

# 2. 추천 추론 함수
def recommend_from_input(model: nn.Module,
                         user_input_df,  # shape: (1, 25)
                         travel_input_df,  # shape: (1, 29)
                         base_data: HeteroData,
                         visit_area_id_map: Dict[str, int],
                         topk: int = 5) -> List[Dict[str, float]]:
    
    # 1. tensor 변환
    user_tensor = torch.tensor(user_input_df.values, dtype=torch.float)
    travel_tensor = torch.tensor(travel_input_df.values, dtype=torch.float)

    # 2. base_data 복사 및 입력 추가
    data = base_data[3].clone()
    data['user'].x = torch.cat([data['user'].x, user_tensor], dim=0)
    data['travel'].x = torch.cat([data['travel'].x, travel_tensor], dim=0)

    uid = data['user'].x.size(0) - 1
    tid = data['travel'].x.size(0) - 1

    # 3. 엣지 연결 (단방향 + 역방향)
    data[('user', 'traveled', 'travel')].edge_index = torch.cat([
        data[('user', 'traveled', 'travel')].edge_index,
        torch.tensor([[uid], [tid]], dtype=torch.long)
    ], dim=1)

    data[('travel', 'traveled_by', 'user')].edge_index = torch.cat([
        data[('travel', 'traveled_by', 'user')].edge_index,
        torch.tensor([[tid], [uid]], dtype=torch.long)
    ], dim=1)

    # 4. 추론
    with torch.no_grad():
        scores = model(data.x_dict, data.edge_index_dict)
        k = min(topk, scores.size(0))
        topk_result = torch.topk(scores, k)
        indices = topk_result.indices.tolist()
        values = topk_result.values.tolist()

    # 5. index → visit_area_id 변환
    index_to_id = {v: k for k, v in visit_area_id_map.items()}
    results = []

    for i, v in zip(indices, values):
        va_id = index_to_id.get(i, f"UNKNOWN_{i}")
        results.append({
            "visit_area_id": va_id,
            "score": round(float(v), 4)
        })

    return results

def preprocess_gnn(user_json: dict, travel_input_raw: dict):
    # 1. USER 처리
    user_input = {}
    for key in user_feature_keys:
        val = user_json.get(key, 0)
        user_input[key] = float(val) if str(val).isdigit() else 0

    # 2. TRAVEL 처리
    travel_input = {k: 0.0 for k in travel_feature_keys}
    
    # for k in ['LODGOUT_COST', 'ACTIVITY_COST', 'TOTAL_COST']:
    #     raw_val = travel_input_raw.get(k, "0")
    #     travel_input[k] = float(raw_val) * 10000

    
    # 여행 기간 → DURATION 계산
    date_range = travel_input_raw.get('date_range', '')
    try:
        start_str, end_str = [d.strip() for d in date_range.split('-')]
        start = datetime.strptime(start_str, "%Y-%m-%d")
        end = datetime.strptime(end_str, "%Y-%m-%d")
        travel_input['DURATION'] = max((end - start).days, 1)
    except:
        travel_input['DURATION'] = 1.0

    scaler = pickle.load(open('./pickle/cost_scaler.pkl', 'rb'))
    # 1. 수치형 변수 목록
    num_cols = ["LODGOUT_COST", "ACTIVITY_COST", "TOTAL_COST", "DURATION"]

    # 2. travel_input_raw = 사용자가 입력한 문자열 딕셔너리 (ex. from Flask form)
    #    예시: {'LODGOUT_COST': '2.5', 'DURATION': '3', ...}
    #    → 값은 float으로, 예산은 ×10000 처리
    input_vals = []
    for col in num_cols:
        val = float(travel_input_raw.get(col, 0))
        if col in ["LODGOUT_COST", "ACTIVITY_COST", "TOTAL_COST"]:
            val *= 10000  # 만원 단위 → 원 단위
        input_vals.append(val)

    # 3. transform 적용
    # scaler는 이미 fit되어 있어야 함
    scaled_vals = scaler.transform([input_vals])[0]  # 결과: [0.23, 0.41, 0.37, 0.67] 등

    # 4. 다시 travel_input에 저장
    for i, col in enumerate(num_cols):
        travel_input[col] = scaled_vals[i]
    
    # 이동수단, 나이대, 동반자
    travel_input['MVMN_NM_ENC'] = float(travel_input_raw.get('MVMN_NM_ENC', 0))
    travel_input['age_ENC'] = float(0) if int(user_json['AGE_GRP']) <= float(39) else 1.0

    mission_type = travel_input_raw.get('mission_type', 'normal')
    if mission_type == 'special':
        travel_input['whowith_ENC'] = float(7)
        # 모든 PURPOSE를 0으로 유지
    else:
        whowith_label_to_index = {
            '3대 동반 여행': 0,
            '3인 이상 친구': 1,
            '나홀로 여행': 2,
            '부모 동반': 3,
            '부부': 4,
            '자녀동반': 5,
            '커플': 6,
            '특별미션': 7,
        }
        whowith_raw = travel_input_raw.get('whowith_ENC', '커플')
        travel_input['whowith_ENC'] = float(whowith_label_to_index.get(whowith_raw, 0))

        # mission_ENC → PURPOSE one-hot
        missions = travel_input_raw.get('mission_ENC', '')
        for code in missions.split(','):
            code = code.strip()
            if code.isdigit():
                key = f"PURPOSE_{code}"
                if key in travel_input:
                    travel_input[key] = 1.0

    # mission_ENC 원래 컬럼도 float로 추가
    travel_input['mission_ENC'] = float(0) if mission_type == 'normal' and missions else 1.0

    
    user_df = pd.DataFrame([user_input]).reindex(columns=user_feature_keys, fill_value=0)
    travel_df = pd.DataFrame([travel_input]).reindex(columns=travel_feature_keys, fill_value=0)

    ################# CSV 저장 (디버깅용) #################
    # user_df.to_csv('user_input.csv', index=False)
    # travel_df.to_csv('travel_input.csv', index=False)
    ####################################################

    return user_df, travel_df
