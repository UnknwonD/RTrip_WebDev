from typing import List, Dict
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv

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

# 1. 여행 목적(PURPOSE) 코드
purpose_options = [
    (1, "쇼핑"), (2, "테마파크, 놀이시설, 동/식물원 방문"), (3, "역사 유적지 방문"),
    (4, "시티투어"), (5, "야외 스포츠, 레포츠 활동"), (6, "지역 문화예술/공연/전시시설 관람"),
    (7, "유흥/오락(나이트라이프)"), (8, "캠핑"), (9, "지역 축제/이벤트 참가"),
    (10, "온천/스파"), (11, "교육/체험 프로그램 참가"), (12, "드라마 촬영지 방문"),
    (13, "종교/성지 순례"), (21, "Well-ness 여행"), (22, "SNS 인생샷 여행"),
    (23, "호캉스 여행"), (24, "신규 여행지 발굴"), (25, "반려동물 동반 여행"),
    (26, "인플루언서 따라하기 여행"), (27, "친환경 여행(플로깅 여행)"), (28, "등반 여행")
]

# 2. 이동수단 그룹
movement_options = [
    (1, "자가용/렌트카/캠핑카 (운전자형)"),
    (2, "택시 (비개인 운전자형)"),
    (3, "지하철/버스 등 대중교통"),
    (4, "KTX/비행기/배 등 장거리 교통"),
    (5, "도보/자전거 등 비자동 교통"),
    (6, "기타")
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

# 1. 모델 로더 함수
def load_model(model_path: str, metadata, input_dims=(25, 29)) -> nn.Module:
    model = PpiKkoTwistGNN(
        metadata=metadata,
        user_input_dim=input_dims[0],
        travel_input_dim=input_dims[1],
        hidden_dim=128,
        num_layers=8
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

# 2. 추천 추론 함수
def recommend_from_input(model: nn.Module,
                         user_input: Dict[str, float],
                         travel_input: Dict[str, float],
                         base_data,
                         visit_area_id_map: Dict[str, int],
                         topk: int = 5) -> List[Dict[str, float]]:
    # 입력 변환
    user_tensor = torch.tensor([list(user_input.values())], dtype=torch.float)
    travel_tensor = torch.tensor([list(travel_input.values())], dtype=torch.float)

    # HeteroData 복사 및 입력 추가
    data = base_data.clone()
    data['user'].x = torch.cat([data['user'].x, user_tensor], dim=0)
    data['travel'].x = torch.cat([data['travel'].x, travel_tensor], dim=0)
    uid, tid = data['user'].x.size(0) - 1, data['travel'].x.size(0) - 1

    # 엣지 연결 (user ↔ travel)
    data[('user', 'traveled', 'travel')].edge_index = torch.cat([
        data[('user', 'traveled', 'travel')].edge_index,
        torch.tensor([[uid], [tid]], dtype=torch.long)
    ], dim=1)
    data[('travel', 'traveled_by', 'user')].edge_index = torch.cat([
        data[('travel', 'traveled_by', 'user')].edge_index,
        torch.tensor([[tid], [uid]], dtype=torch.long)
    ], dim=1)

    # 추론
    with torch.no_grad():
        scores = model(data.x_dict, data.edge_index_dict)
        k = min(topk, scores.size(0))
        topk_result = torch.topk(scores, k)
        indices = topk_result.indices.tolist()
        values = topk_result.values.tolist()

    # index → visit_area_id 변환
    index_to_id = {v: k for k, v in visit_area_id_map.items()}
    return [{"visit_area_id": index_to_id[i], "score": round(v, 4)} for i, v in zip(indices, values)]

from datetime import datetime
import pandas as pd

def preprocess_gnn(user_json: dict, travel_input_raw: dict):
    # 1. USER 처리
    user_input = {}
    for key in user_feature_keys:
        val = user_json.get(key, 0)
        user_input[key] = float(val) if str(val).isdigit() else 0

    # 2. TRAVEL 처리
    travel_input = {k: 0.0 for k in travel_feature_keys}

    # 비용: 만원 → 원
    for k in ['LODGOUT_COST', 'ACTIVITY_COST', 'TOTAL_COST']:
        raw_val = travel_input_raw.get(k, "0")
        travel_input[k] = float(raw_val) * 10000

    # 여행 기간 → DURATION 계산
    date_range = travel_input_raw.get('date_range', '')
    try:
        start_str, end_str = [d.strip() for d in date_range.split('-')]
        start = datetime.strptime(start_str, "%Y-%m-%d")
        end = datetime.strptime(end_str, "%Y-%m-%d")
        travel_input['DURATION'] = max((end - start).days, 1)
    except:
        travel_input['DURATION'] = 1.0

    # 이동수단, 나이대, 동반자
    travel_input['MVMN_NM_ENC'] = float(travel_input_raw.get('MVMN_NM_ENC', 0))
    travel_input['age_ENC'] = float(0) if int(user_json['AGE_GRP']) <= float(39) else 1.0

    mission_type = travel_input_raw.get('mission_type', 'normal')
    if mission_type == 'special':
        travel_input['whowith_ENC'] = "특별미션"
        # 모든 PURPOSE를 0으로 유지
    else:
        travel_input['whowith_ENC'] = float(travel_input_raw.get('whowith_ENC', 0))

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

    ################# DF 변환 (디버깅용) #################
    user_df = pd.DataFrame([user_input]).reindex(columns=user_feature_keys, fill_value=0)
    travel_df = pd.DataFrame([travel_input]).reindex(columns=travel_feature_keys, fill_value=0)

    user_df.to_csv('user_input.csv', index=False)
    travel_df.to_csv('travel_input.csv', index=False)
    ####################################################

    return user_df, travel_df
