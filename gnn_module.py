# 최적화된 GNN 추천 시스템
import os
# 기존 환경변수들 + 추가
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'  # 추가
os.environ['NUMBA_DISABLE_INTEL_SVML'] = '1'  # 추가

# Numba 캐시 비활성화 (디버깅용)
os.environ['NUMBA_CACHE_DIR'] = '/tmp'
os.environ['NUMBA_DISABLE_JIT'] = '0'  # JIT 유지하되 안전 모드

from dotenv import load_dotenv
load_dotenv(override = True)
kakao_api_key = os.getenv("KAKAO_REST_API")

import pandas as pd
import numpy as np
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import HeteroData
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from math import radians, cos, sin, sqrt, atan2
import random
import warnings
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import numba
from numba import jit, prange

warnings.filterwarnings('ignore')

# M1 Mac에서 multiprocessing 문제 해결
import sys
if sys.platform == "darwin":
    import multiprocessing
    multiprocessing.set_start_method('fork', force=True)

# 상수 정의 (기존과 동일)
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

purpose_options = [
    (1, "🛍️ 쇼핑 & 트렌드 탐방"), (2, "🏛️ 문화·예술·역사 체험"), 
    (3, "🎢 테마파크 & 놀이시설"), (4, "🏙️ 도심 여행 & 휴식"),
    (5, "🏕️ 아웃도어 & 액티비티"), (6, "♨️ 온천 & 힐링 여행"), 
    (7, "📸 SNS 핫플 & 인생샷"), (8, "✨ 신규 & 미개척 지역 탐방"),
    (9, "🌿 친환경 & 지속가능한 여행")
]

movement_options = [
    (1, "자가용"), (2, "대중교통"), (3, "기타 이동수단")
]

whowith_options = [
    ("단독여행", ["나홀로 여행"]),
    ("2인여행", ["커플", "부부"]),
    ("가족여행", ["자녀동반", "부모 동반", "3대 동반 여행"]),
    ("친구/지인 여행", ["3인 이상 친구"]),
    ("기타", ["기타"])
]

# Numba JIT으로 거리 계산 최적화
@jit(nopython=True)
def haversine_distance(lat1, lon1, lat2, lon2):
    """벡터화된 하버사인 거리 계산"""
    R = 6371.0
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c

# @jit(nopython=True, parallel=True)
@jit(nopython=True)
def calculate_distance_matrix(coords):
    """병렬화된 거리 행렬 계산"""
    n = len(coords)
    distances = np.zeros((n, n))
    
    for i in prange(n):
        for j in range(i+1, n):
            dist = haversine_distance(coords[i, 1], coords[i, 0], coords[j, 1], coords[j, 0])
            distances[i, j] = dist
            distances[j, i] = dist
    
    return distances


class OptimizedGNN(nn.Module):
    """최적화된 GNN 모델 - 더 가벼운 구조"""
    def __init__(self, in_channels, hidden_channels, out_channels, travel_context_dim, 
                 num_heads=2, dropout=0.1):  # heads와 dropout 줄임
        super().__init__()
        
        # 더 간단한 구조로 변경
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, 
                           dropout=dropout, concat=True, edge_dim=5)
        self.gat2 = GATConv(hidden_channels * num_heads, out_channels, 
                           heads=1, dropout=dropout, concat=False, edge_dim=5)
        
        self.bn1 = nn.BatchNorm1d(hidden_channels * num_heads)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 간소화된 travel encoder
        self.travel_encoder = nn.Linear(travel_context_dim, out_channels)
        
        # 간소화된 fusion
        self.fusion = nn.Linear(out_channels * 2, out_channels)
        
        # 간소화된 preference head
        self.preference_head = nn.Linear(out_channels, 1)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, data, travel_context):
        x = data['visit_area'].x
        edge_index = data['visit_area', 'moved_to', 'visit_area'].edge_index
        edge_attr = data['visit_area', 'moved_to', 'visit_area'].edge_attr
        
        # GNN layers
        x = self.gat1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        graph_embedding = self.gat2(x, edge_index, edge_attr)
        graph_embedding = self.bn2(graph_embedding)
        
        # Travel context
        travel_embedding = self.travel_encoder(travel_context)
        travel_embedding_expanded = travel_embedding.expand(graph_embedding.size(0), -1)
        
        # Fusion
        fused = torch.cat([graph_embedding, travel_embedding_expanded], dim=1)
        final_embedding = self.fusion(fused)
        
        # Scores
        preference_scores = torch.sigmoid(self.preference_head(final_embedding))
        
        return final_embedding, preference_scores


class FastDataProcessor:
    """최적화된 데이터 프로세서"""
    def __init__(self):
        self.visit_scaler = RobustScaler()
        self.travel_scaler = StandardScaler()
        self.exclude_keywords = {
            '역', '터미널', '공항', '휴게소', '정류장', '톨게이트', '교차로', '출구', '입구',
            'IC', 'JC', '나들목', '분기점', '요금소', '주차장', '주유소', '충전소',
            '아파트', '원룸', '오피스텔', '빌라', '주택', '빌딩', '상가', '모텔', '집', '교직원', '하나로마트', '마트', '아파트'
        }
        
        # 숙소 키워드
        self.accommodation_keywords = {
            '호텔', '모텔', '리조트', '펜션', '게스트하우스', '민박', 'hotel', 'motel', 
            'resort', '콘도', '캠핑장', '글램핑', 'camping', '스테이', 'stay',
            '숙박', '여관', '료칸', '한옥스테이', '농가민박', '어촌민박'
        }
        self._cache = {}
        
    @lru_cache(maxsize=10000)
    def is_accommodation(self, name):
        """장소명으로 숙소 여부 판단"""
        if pd.isna(name):
            return False
        name_str = str(name).lower()
        
        for keyword in self.accommodation_keywords:
            if keyword.lower() in name_str:
                return True
        return False
        
    @lru_cache(maxsize=10000)
    def should_exclude_location(self, name):
        """캐싱된 위치 제외 확인"""
        if pd.isna(name):
            return False
        name_str = str(name).lower()
        
        for keyword in self.exclude_keywords:
            if keyword.lower() in name_str:
                tourist_keywords = {'관광', '테마', '파크', '랜드', '월드', '호텔',
                                  '맛집', '식당', '카페', '박물관', '전시', '갤러리', '문화'}
                if any(tk in name_str for tk in tourist_keywords) and keyword != '아파트':
                    continue
                return True
        return False
    
    def process_visit_area_features(self, visit_area_df):
        """벡터화된 특성 처리"""
        visit_area_df = visit_area_df.copy()
        
        # 벡터화된 결측치 처리
        visit_area_df['X_COORD'].fillna(visit_area_df['X_COORD'].mean(), inplace=True)
        visit_area_df['Y_COORD'].fillna(visit_area_df['Y_COORD'].mean(), inplace=True)
        visit_area_df['VISIT_CHC_REASON_CD'].fillna(0, inplace=True)
        
        features = visit_area_df[['X_COORD', 'Y_COORD']].values
        
        # One-hot encoding 최적화
        type_dummies = pd.get_dummies(visit_area_df['VISIT_AREA_TYPE_CD'], prefix='type', sparse=True)
        reason_dummies = pd.get_dummies(visit_area_df['VISIT_CHC_REASON_CD'], prefix='reason', sparse=True)
        
        # 벡터화된 정규화
        satisfaction_cols = ['DGSTFN', 'REVISIT_INTENTION', 'RCMDTN_INTENTION']
        for col in satisfaction_cols:
            visit_area_df[col].fillna(3, inplace=True)
        
        satisfaction_values = visit_area_df[satisfaction_cols].values
        normalized_satisfaction = (satisfaction_values - 1) / 4.0
        
        # 벡터화된 인기도 계산
        popularity_score = (normalized_satisfaction[:, 0] * 0.4 + 
                          normalized_satisfaction[:, 1] * 0.3 + 
                          normalized_satisfaction[:, 2] * 0.3)
        
        # 벡터화된 제외 페널티
        exclude_penalty = visit_area_df['VISIT_AREA_NM'].apply(self.should_exclude_location).values * -0.5
        
        # 효율적인 결합
        all_features = np.hstack([
            features,
            type_dummies.values,
            reason_dummies.values,
            normalized_satisfaction,
            popularity_score.reshape(-1, 1),
            exclude_penalty.reshape(-1, 1)
        ])
        
        return self.visit_scaler.fit_transform(all_features.astype(np.float32))


class FastRecommendationEngine:
    """최적화된 추천 엔진 - 지역별 모델 호환성 추가"""
    def __init__(self, 
                 device, 
                 model_path='./pickle/improved_travel_recommendation_model.pt', 
                 data_path='./pickle/improved_travel_data.pkl',
                 max_places_per_day=8, 
                 min_places_per_day=5,
                 max_distance_km=50):
        
        torch.set_num_threads(1)
        
        # 1. 데이터 로드
        try:
            with open(data_path, 'rb') as f:
                self.data_dict = pickle.load(f)
            
            # 🔧 기존 포맷 호환성 유지
            self.visit_area_df = self.data_dict['visit_area_df']
            self.graph_data = self.data_dict['graph_data']
            self.visit_scaler = self.data_dict['visit_scaler']
            self.travel_scaler = self.data_dict['travel_scaler']
            
            # 🆕 새로운 정보 사용 (있는 경우)
            self.id_to_index = self.data_dict.get('id_to_index', {})
            self.region_info = self.data_dict.get('region_info', {})
            
            print(f"✅ 데이터 로드 완료:")
            print(f"  - 방문지 데이터: {len(self.visit_area_df)}개 레코드")
            if self.region_info:
                print(f"  - 그래프 노드: {self.region_info.get('num_nodes', 'N/A')}개")
            if self.id_to_index:
                print(f"  - ID 매핑: {len(self.id_to_index)}개")
                
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            raise RuntimeError(f"데이터 파일을 로드할 수 없습니다: {data_path}")
        
        # 🆕 2. 실제 사용되는 방문지 데이터만 추출 (지역별 모델 대응)
        if self.id_to_index:
            # ID 매핑이 있는 경우: 실제 사용되는 데이터만 필터링
            used_area_ids = set(self.id_to_index.keys())
            mask = self.visit_area_df['NEW_VISIT_AREA_ID'].isin(used_area_ids)
            self.filtered_visit_df = self.visit_area_df[mask].copy()
            
            # 중복 제거 및 정렬 (학습 시와 동일한 순서)
            agg_dict = {
                'VISIT_AREA_NM': 'first',
                'X_COORD': 'mean',
                'Y_COORD': 'mean',
                'ROAD_NM_ADDR': 'first',
                'LOTNO_ADDR': 'first',
                'VISIT_AREA_TYPE_CD': 'first'
            }
            
            # 만족도 컬럼들이 있는 경우 추가
            satisfaction_cols = ['DGSTFN', 'REVISIT_INTENTION', 'RCMDTN_INTENTION']
            for col in satisfaction_cols:
                if col in self.filtered_visit_df.columns:
                    agg_dict[col] = 'mean'
            
            self.filtered_visit_df = self.filtered_visit_df.groupby('NEW_VISIT_AREA_ID').agg(agg_dict).reset_index()
            
            # ID 순서대로 정렬
            ordered_ids = sorted(self.id_to_index.keys())
            id_to_row_idx = {area_id: i for i, area_id in enumerate(ordered_ids)}
            self.filtered_visit_df['sort_order'] = self.filtered_visit_df['NEW_VISIT_AREA_ID'].map(id_to_row_idx)
            self.filtered_visit_df = self.filtered_visit_df.sort_values('sort_order').reset_index(drop=True)
            self.filtered_visit_df = self.filtered_visit_df.drop('sort_order', axis=1)
            
            print(f"  - 필터링된 방문지: {len(self.filtered_visit_df)}개")
            
            # 🆕 필터링된 데이터 사용 플래그
            self.use_filtered_data = True
        else:
            # ID 매핑이 없는 경우: 원본 데이터 사용
            self.filtered_visit_df = self.visit_area_df.copy()
            self.use_filtered_data = False
            print("  ⚠️ ID 매핑 정보가 없어 원본 데이터 사용")
        
        # 3. 좌표 사전 처리 및 거리 행렬 계산 (필터링된 데이터 기준)
        self.coords = self.filtered_visit_df[['X_COORD', 'Y_COORD']].values
        # NaN 값 처리
        self.coords = np.nan_to_num(self.coords, nan=self.coords[~np.isnan(self.coords).any(axis=1)].mean(axis=0)[0] if len(self.coords[~np.isnan(self.coords).any(axis=1)]) > 0 else 0)
        self.distance_matrix = None  # 필요시 계산
        
        # 4. 디바이스 설정
        self.device = device
        self.graph_data = self.graph_data.to(self.device)
        
        # 5. 최적화된 모델 로드 (기존 코드 유지)
        checkpoint = torch.load(model_path, map_location=self.device)
        model_config = checkpoint['model_config']
        
        # 기존 모델과 호환성을 위해 원래 구조 사용
        try:
            # 먼저 원래 모델 구조로 시도
            from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool
            
            # 원래 모델 정의 (ImprovedTravelGNN)
            class OriginalGNN(nn.Module):
                def __init__(self, in_channels, hidden_channels, out_channels, travel_context_dim, 
                             num_heads=4, dropout=0.2):
                    super().__init__()
                    self.gat1 = GATConv(in_channels, hidden_channels // num_heads, 
                                       heads=num_heads, dropout=dropout, concat=True, edge_dim=5)
                    self.gat2 = GATConv(hidden_channels, hidden_channels // num_heads, 
                                       heads=num_heads, dropout=dropout, concat=True, edge_dim=5)
                    self.gat3 = GATConv(hidden_channels, out_channels, 
                                       heads=1, dropout=dropout, concat=False, edge_dim=5)
                    
                    self.bn1 = nn.BatchNorm1d(hidden_channels)
                    self.bn2 = nn.BatchNorm1d(hidden_channels)
                    self.bn3 = nn.BatchNorm1d(out_channels)
                    
                    self.travel_encoder = nn.Sequential(
                        nn.Linear(travel_context_dim, hidden_channels),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_channels, out_channels)
                    )
                    
                    self.distance_attention = nn.Sequential(
                        nn.Linear(2, hidden_channels // 2),
                        nn.ReLU(),
                        nn.Linear(hidden_channels // 2, 1),
                        nn.Sigmoid()
                    )
                    
                    self.fusion_net = nn.Sequential(
                        nn.Linear(out_channels * 2, hidden_channels),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_channels, out_channels),
                        nn.ReLU(),
                        nn.Linear(out_channels, out_channels)
                    )
                    
                    self.preference_head = nn.Sequential(
                        nn.Linear(out_channels, hidden_channels // 2),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_channels // 2, 1),
                        nn.Sigmoid()
                    )
                    
                    self.dropout = nn.Dropout(dropout)
                    
                def forward(self, data, travel_context, return_attention=False):
                    x = data['visit_area'].x
                    edge_index = data['visit_area', 'moved_to', 'visit_area'].edge_index
                    edge_attr = data['visit_area', 'moved_to', 'visit_area'].edge_attr
                    
                    coords = x[:, :2]
                    
                    x1 = self.gat1(x, edge_index, edge_attr)
                    x1 = self.bn1(x1)
                    x1 = F.relu(x1)
                    x1 = self.dropout(x1)
                    
                    x2 = self.gat2(x1, edge_index, edge_attr)
                    x2 = self.bn2(x2)
                    x2 = F.relu(x2 + x1)
                    x2 = self.dropout(x2)
                    
                    graph_embedding = self.gat3(x2, edge_index, edge_attr)
                    graph_embedding = self.bn3(graph_embedding)
                    
                    distance_weights = self.distance_attention(coords)
                    graph_embedding = graph_embedding * distance_weights
                    
                    travel_embedding = self.travel_encoder(travel_context)
                    travel_embedding_expanded = travel_embedding.expand(graph_embedding.size(0), -1)
                    
                    fused_features = torch.cat([graph_embedding, travel_embedding_expanded], dim=1)
                    final_embedding = self.fusion_net(fused_features)
                    
                    preference_scores = self.preference_head(final_embedding)
                    
                    return final_embedding, preference_scores
            
            # 원래 모델로 로드
            self.model = OriginalGNN(**model_config).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            print("✅ 기존 모델 구조로 로드 성공")
            
        except Exception as e:
            print(f"⚠️  기존 모델 로드 실패: {e}")
            print("💡 최적화된 모델 구조 사용")
            
            # 최적화된 모델 사용
            self.model = OptimizedGNN(
                in_channels=model_config['in_channels'],
                hidden_channels=model_config['hidden_channels'] // 2,
                out_channels=model_config['out_channels'],
                travel_context_dim=model_config['travel_context_dim'],
                num_heads=2,
                dropout=0.1
            ).to(self.device)
            
            # 가중치는 로드하지 않고 새로 학습된 것처럼 사용
            
        self.model.eval()
        
        # 6. 초기화 (기존 코드 유지)
        self.max_distance_km = max_distance_km
        self.excluded_ids = set()
        self.processor = FastDataProcessor()
        self.route_generator = FastRouteGenerator(max_places_per_day, min_places_per_day)
        
        self.min_recommendations_per_day = 3
        self.min_total_recommendations = 10
        
        # 7. 캐시 초기화
        self._embedding_cache = None
        self._score_cache = None
    
    
    @torch.no_grad()
    def get_recommendations(self, travel_context, top_k=50, diversity_weight=0.3,
                          excluded_ids=None, filter_useless=True, consider_distance=True):
        """🔧 개선된 추천 로직 - 지역별 모델 호환"""
        self.model.eval()
        
        # 임베딩 캐싱
        if self._embedding_cache is None:
            embeddings, preference_scores = self.model(self.graph_data, travel_context)
            self._embedding_cache = embeddings
            self._score_cache = preference_scores.squeeze()
        else:
            embeddings = self._embedding_cache
            preference_scores = self._score_cache
        
        scores = preference_scores.clone()
        
        # 🔧 개선된 벡터화 필터링 - 데이터 크기 호환성 고려
        if filter_useless:
            # 사용할 데이터 결정
            target_df = self.filtered_visit_df if self.use_filtered_data else self.visit_area_df
            
            try:
                exclude_mask = torch.tensor([
                    self.processor.should_exclude_location(name) 
                    for name in target_df['VISIT_AREA_NM']
                ], device=self.device)
                
                # 크기 검증
                if len(exclude_mask) == len(scores):
                    scores[exclude_mask] *= 0.3
                else:
                    print(f"⚠️ 마스크 크기 불일치: exclude_mask={len(exclude_mask)}, scores={len(scores)}")
                    # 안전한 대체 처리
                    for i, name in enumerate(target_df['VISIT_AREA_NM']):
                        if i < len(scores) and self.processor.should_exclude_location(name):
                            scores[i] *= 0.3
            except Exception as e:
                print(f"⚠️ 필터링 중 오류: {e}, 스킵")
        
        # 🔧 개선된 제외 ID 처리
        if excluded_ids:
            try:
                target_df = self.filtered_visit_df if self.use_filtered_data else self.visit_area_df
                
                exclude_mask = torch.tensor([
                    target_df.iloc[i]['NEW_VISIT_AREA_ID'] in excluded_ids 
                    for i in range(min(len(target_df), len(scores)))
                ], device=self.device)
                
                if len(exclude_mask) == len(scores):
                    scores[exclude_mask] = -1.0
                else:
                    # 안전한 대체 처리
                    for i in range(min(len(target_df), len(scores))):
                        area_id = target_df.iloc[i]['NEW_VISIT_AREA_ID']
                        if area_id in excluded_ids:
                            scores[i] = -1.0
            except Exception as e:
                print(f"⚠️ 제외 ID 처리 중 오류: {e}, 스킵")
        
        # 유효한 점수가 있는지 확인
        valid_scores = scores > 0
        num_valid = torch.sum(valid_scores).item()
        
        if num_valid == 0:
            # 유효한 점수가 없으면 필터링 조건을 완화
            if filter_useless:
                # 필터링을 완화하여 다시 시도
                scores = preference_scores.clone()
                if excluded_ids:
                    try:
                        target_df = self.filtered_visit_df if self.use_filtered_data else self.visit_area_df
                        for i in range(min(len(target_df), len(scores))):
                            area_id = target_df.iloc[i]['NEW_VISIT_AREA_ID']
                            if area_id in excluded_ids:
                                scores[i] = -1.0
                    except:
                        pass
                valid_scores = scores > 0
                num_valid = torch.sum(valid_scores).item()
            
            if num_valid == 0:
                # 그래도 없으면 상위 점수 강제 선택
                scores = preference_scores.clone()
                valid_scores = scores > -float('inf')
                num_valid = torch.sum(valid_scores).item()
        
        # 빠른 상위 k 선택
        if consider_distance and top_k > 10 and num_valid > 10:
            recommendations = self._fast_distance_recommendation(scores, top_k)
        else:
            # 단순 점수 기반 선택
            k_value = min(top_k, num_valid)
            if k_value > 0:
                top_indices = torch.topk(scores, k_value, sorted=True).indices
                recommendations = top_indices.cpu().numpy().tolist()
            else:
                recommendations = []
        
        # 최소 개수 보장
        if len(recommendations) < self.min_total_recommendations:
            # 점수 순으로 추가 선택
            all_indices = torch.argsort(scores, descending=True)
            for idx in all_indices:
                if idx.item() not in recommendations:
                    recommendations.append(idx.item())
                    if len(recommendations) >= self.min_total_recommendations:
                        break
        
        return recommendations, embeddings, preference_scores
    
    def _fast_distance_recommendation(self, scores, top_k):
        """최적화된 거리 기반 추천"""
        # 유효한 점수를 가진 인덱스만 선택
        valid_mask = scores > 0
        valid_indices = torch.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            # 유효한 점수가 없으면 빈 리스트 반환
            return []
        
        # 유효한 점수 중에서만 선택
        valid_scores = scores[valid_indices]
        k_value = min(top_k * 2, len(valid_indices))
        
        if k_value == 0:
            return []
        
        print(len(valid_scores), k_value)
        valid_scores = valid_scores.squeeze()

        # 상위 후보 선택
        top_k_result = torch.topk(valid_scores, k_value, sorted=True)
        top_candidates = valid_indices[top_k_result.indices]
        candidates = top_candidates.cpu().numpy()
        
        if len(candidates) <= top_k:
            return candidates.tolist()
        
        # 거리 행렬이 없으면 계산
        if self.distance_matrix is None:
            self.distance_matrix = calculate_distance_matrix(self.coords)
        
        # 그리디 선택
        selected = [candidates[0]]
        remaining = list(candidates[1:])
        
        while len(selected) < top_k and remaining:
            last_idx = selected[-1]
            
            # 벡터화된 거리 계산
            distances = self.distance_matrix[last_idx, remaining]
            distance_scores = 1 / (1 + distances * 0.1)
            
            # 스코어 결합
            combined_scores = scores[remaining].cpu().numpy() * 0.6 + distance_scores * 0.4
            
            best_idx = np.argmax(combined_scores)
            selected.append(remaining[best_idx])
            remaining.pop(best_idx)
        
        return selected
    
    def _filter_distant_locations(self, routes, all_recommendations):
        """거리가 너무 먼 장소를 필터링하고 대체"""
        if not routes or len(routes) == 0:
            return routes
        
        # 거리 행렬 확인
        if self.distance_matrix is None:
            self.distance_matrix = calculate_distance_matrix(self.coords)
        
        filtered_routes = {}
        used_indices = set()  # 이미 사용된 인덱스 추적
        
        # 각 날짜의 장소들을 저장
        for day, locations in routes.items():
            for loc in locations:
                used_indices.add(loc['idx'])
        
        # 각 날짜별로 처리
        for day in sorted(routes.keys()):
            locations = routes[day]
            if not locations:
                filtered_routes[day] = []
                continue
            
            # 1. 일정 내 이상치 감지 및 제거
            filtered_day = self._filter_day_outliers(locations, day, routes, used_indices)
            
            # 2. 이전/다음 날과의 연속성 체크
            if day > 0 and (day-1) in filtered_routes:
                filtered_day = self._check_day_continuity(
                    filtered_routes[day-1], filtered_day, day, used_indices
                )
            
            filtered_routes[day] = filtered_day
            
            # 사용된 인덱스 업데이트
            for loc in filtered_day:
                used_indices.add(loc['idx'])
        
        # 각 날짜가 최소 장소 수를 만족하는지 확인
        filtered_routes = self._ensure_minimum_places(filtered_routes, all_recommendations, used_indices)
        
        return filtered_routes

    def _filter_day_outliers(self, locations, day, all_routes, used_indices):
        """하루 일정 내에서 거리가 먼 이상치 제거"""
        if len(locations) <= 2:
            return locations
        
        filtered = []
        location_indices = [loc['idx'] for loc in locations]
        
        # 각 장소의 평균 거리 계산
        avg_distances = []
        for i, loc in enumerate(locations):
            distances = []
            for j, other_loc in enumerate(locations):
                if i != j:
                    dist = self.distance_matrix[loc['idx']][other_loc['idx']]
                    distances.append(dist)
            avg_dist = np.mean(distances) if distances else 0
            avg_distances.append(avg_dist)
        
        # 평균 거리와 표준편차 계산
        mean_avg_dist = np.mean(avg_distances)
        std_avg_dist = np.std(avg_distances)
        
        # 이상치 판단 (평균 + 2*표준편차 또는 최대 거리 임계값)
        threshold = min(mean_avg_dist + 2 * std_avg_dist, self.max_distance_km)
        
        # 이상치가 아닌 장소들만 유지
        outliers = []
        for i, (loc, avg_dist) in enumerate(zip(locations, avg_distances)):
            if avg_dist <= threshold:
                filtered.append(loc)
            else:
                outliers.append((i, loc, avg_dist))
                print(f"  ⚠️ Day {day+1}: '{loc['name']}' 제거 (평균 거리: {avg_dist:.1f}km)")
        
        # 제거된 장소 대체
        for _, outlier, _ in outliers:
            replacement = self._find_replacement_location(
                filtered, outlier['idx'], used_indices, day
            )
            if replacement:
                filtered.append(replacement)
                used_indices.add(replacement['idx'])  # 이 줄 추가!
                print(f"  ✅ '{outlier['name']}' → '{replacement['name']}' 대체")
        
        return filtered

    def _check_day_continuity(self, prev_day_locations, current_day_locations, day, used_indices):
        """전날 마지막 장소와 현재 날 첫 장소 간의 거리 체크"""
        if not prev_day_locations or not current_day_locations:
            return current_day_locations
        
        last_prev = prev_day_locations[-1]
        first_current = current_day_locations[0]
        
        # 거리 계산
        distance = self.distance_matrix[last_prev['idx']][first_current['idx']]
        
        if distance > self.max_distance_km:
            print(f"\n  ⚠️ Day {day}→{day+1} 간 거리 문제: {distance:.1f}km")
            print(f"     '{last_prev['name']}' → '{first_current['name']}'")
            
            # 첫 번째 장소를 더 가까운 곳으로 대체
            replacement = self._find_nearby_replacement(
                last_prev['idx'], first_current['idx'], used_indices, prefer_closer=True
            )
            
            if replacement:
                current_day_locations[0] = replacement
                print(f"  ✅ 시작 장소 변경: '{replacement['name']}'")
        
        return current_day_locations

    def _find_replacement_location(self, current_locations, outlier_idx, used_indices, day):
        """🔧 개선된 대체 장소 찾기 - 필터링된 데이터 사용"""
        if not current_locations:
            return None
        
        # 현재 일정의 중심점 계산
        current_indices = [loc['idx'] for loc in current_locations]
        center_coords = np.mean([self.coords[idx] for idx in current_indices], axis=0)
        
        # 사용할 데이터 결정
        target_df = self.filtered_visit_df if self.use_filtered_data else self.visit_area_df
        
        # 모든 후보 장소와의 거리 계산
        candidates = []
        for idx in range(len(target_df)):
            if idx in used_indices or idx == outlier_idx:
                continue
            
            # 중심점과의 거리
            dist_to_center = np.sqrt(
                (self.coords[idx][0] - center_coords[0])**2 + 
                (self.coords[idx][1] - center_coords[1])**2
            )
            
            # 임계값 이내의 장소만 후보로
            if dist_to_center <= self.max_distance_km * 0.7:  # 70% 이내
                area = target_df.iloc[idx]
                area_id = area['NEW_VISIT_AREA_ID']  # ID 체크 추가
                if not self.processor.should_exclude_location(area['VISIT_AREA_NM']) and area_id != 0:
                    candidates.append((idx, dist_to_center, area_id))  # area_id도 포함
        
        if not candidates:
            return None
        
        # 가장 가까운 후보 선택
        candidates.sort(key=lambda x: x[1])
        best_idx = candidates[0][0]
        
        row = target_df.iloc[best_idx]
        addr = row['ROAD_NM_ADDR'] if pd.notna(row['ROAD_NM_ADDR']) else row['LOTNO_ADDR']
        
        return {
            'id': row['NEW_VISIT_AREA_ID'],
            'name': row['VISIT_AREA_NM'],
            'coords': self.coords[best_idx],
            'addr': addr,
            'idx': best_idx,
            'type': row.get('VISIT_AREA_TYPE_CD', 0)
        }

    def _find_nearby_replacement(self, reference_idx, replace_idx, used_indices, prefer_closer=True):
        """🔧 개선된 근처 대체 찾기 - 필터링된 데이터 사용"""
        candidates = []
        
        # 사용할 데이터 결정
        target_df = self.filtered_visit_df if self.use_filtered_data else self.visit_area_df
        
        for idx in range(len(target_df)):
            if idx in used_indices or idx == replace_idx:
                continue
            
            dist = self.distance_matrix[reference_idx][idx]
            
            if dist <= self.max_distance_km * 0.5:  # 최대 거리의 50% 이내
                area = target_df.iloc[idx]
                area_id = area['NEW_VISIT_AREA_ID']  # ID 체크 추가
                if not self.processor.should_exclude_location(area['VISIT_AREA_NM']) and area_id != 0:
                    candidates.append((idx, dist, area_id))  # area_id도 포함
        
        if not candidates:
            return None
        
        # 거리순 정렬
        candidates.sort(key=lambda x: x[1])
        best_idx = candidates[0][0]
        
        row = target_df.iloc[best_idx]
        addr = row['ROAD_NM_ADDR'] if pd.notna(row['ROAD_NM_ADDR']) else row['LOTNO_ADDR']
        return {
            'id': row['NEW_VISIT_AREA_ID'],
            'name': row['VISIT_AREA_NM'],
            'coords': self.coords[best_idx],
            'idx': best_idx,
            'addr': addr,
            'type': row.get('VISIT_AREA_TYPE_CD', 0)
        }

    def _ensure_minimum_places(self, routes, all_recommendations, used_indices):
        """🔧 개선된 최소 장소 수 보장 - 필터링된 데이터 사용"""
        # 사용할 데이터 결정
        target_df = self.filtered_visit_df if self.use_filtered_data else self.visit_area_df
        
        for day, locations in routes.items():
            if len(locations) < self.min_recommendations_per_day:
                needed = self.min_recommendations_per_day - len(locations)
                print(f"\n  ℹ️ Day {day+1}: {needed}개 장소 추가 필요")
                
                # 사용하지 않은 추천 중에서 선택
                for idx in all_recommendations:
                    if idx not in used_indices and idx < len(target_df):
                        row = target_df.iloc[idx]
                        addr = row['ROAD_NM_ADDR'] if pd.notna(row['ROAD_NM_ADDR']) else row['LOTNO_ADDR']
                        new_loc = {
                            'id': row['NEW_VISIT_AREA_ID'],
                            'name': row['VISIT_AREA_NM'],
                            'coords': self.coords[idx],
                            'addr' : addr,
                            'idx': idx,
                            'type': row.get('VISIT_AREA_TYPE_CD', 0)
                        }
                        locations.append(new_loc)
                        used_indices.add(idx)
                        needed -= 1
                        
                        if needed == 0:
                            break
        
        return routes
    
    def optimize_routes(self, recommendations, travel_tensor):
        """🔧 개선된 경로 생성 - 필터링된 데이터 사용"""
        travel_duration = max(1, int(travel_tensor[0, 3]))
        
        # 사용할 데이터 결정
        target_df = self.filtered_visit_df if self.use_filtered_data else self.visit_area_df
        
        # 중복 제거 (기존 코드)
        unique_recommendations = []
        seen_ids = set()
        
        for idx in recommendations:
            if idx < len(target_df):
                area_id = target_df.iloc[idx]['NEW_VISIT_AREA_ID']
                if area_id not in seen_ids and area_id != 0:
                    unique_recommendations.append(idx)
                    seen_ids.add(area_id)
        
        # 최소 개수 확보 (기존 코드)
        min_required = travel_duration * self.min_recommendations_per_day
        if len(unique_recommendations) < min_required:
            for idx in recommendations:
                if idx not in unique_recommendations and idx < len(target_df):
                    area_id = target_df.iloc[idx]['NEW_VISIT_AREA_ID']
                    if area_id != 0:
                        unique_recommendations.append(idx)
                        if len(unique_recommendations) >= min_required:
                            break
        
        # 빠른 경로 생성 (필터링된 데이터 사용)
        optimized_routes = self.route_generator.generate_routes(
            unique_recommendations, target_df, travel_duration, self.coords
        )
        
        # 거리 기반 필터링 추가
        optimized_routes = self._filter_distant_locations(optimized_routes, unique_recommendations)
        
        return optimized_routes, unique_recommendations
    
    def feedback_model(self, feedback, travel_context_tensor, travel_duration, 
                      unique_recommendations, embeddings):
        """최적화된 피드백 처리 (기존 인터페이스 유지)"""
        # 캐시 무효화
        self._embedding_cache = None
        self._score_cache = None
        
        # 사용할 데이터 결정
        target_df = self.filtered_visit_df if self.use_filtered_data else self.visit_area_df
        
        liked_indices = [unique_recommendations[i] for i in feedback.get("liked", []) 
                        if i < len(unique_recommendations)]
        disliked_indices = [unique_recommendations[i] for i in feedback.get("disliked", []) 
                           if i < len(unique_recommendations)]
        
        self.update_preferences(disliked_place_ids=disliked_indices)
        
        # 제외 ID 설정 (안전한 처리)
        excluded_ids = set()
        for idx in disliked_indices:
            if idx < len(target_df):
                excluded_ids.add(target_df.iloc[idx]['NEW_VISIT_AREA_ID'])
        
        # 새로운 추천
        recommendations, _, _ = self.get_recommendations(
            travel_context_tensor, top_k=50, excluded_ids=excluded_ids
        )
        
        # 경로 최적화
        optimized_routes, _ = self.optimize_routes(recommendations, 
                                                   travel_context_tensor.cpu().numpy())
        
        return optimized_routes
    
    def update_preferences(self, liked_place_ids=None, disliked_place_ids=None):
        """사용자 선호도 업데이트"""
        if liked_place_ids:
            print(f"👍 {len(liked_place_ids)}개 장소 선호도 반영")
        if disliked_place_ids:
            print(f"👎 {len(disliked_place_ids)}개 장소 비선호도 반영")
            self.excluded_ids.update(disliked_place_ids)
        
        # 캐시 무효화
        self._embedding_cache = None
        self._score_cache = None


class FastRouteGenerator:
    """최적화된 경로 생성기"""
    def __init__(self, max_places_per_day=12, min_places_per_day=3):
        self.distance_cache = {}
        self.max_places_per_day = max_places_per_day 
        self.min_places_per_day = min_places_per_day
        
    def _balance_daily_routes(self, routes, travel_duration):
        """일자별 장소 수를 균형있게 재분배"""
        if not routes:
            return routes
        
        # 🆕 숙소와 관광지 분리
        processor = FastDataProcessor()
        all_accommodations = []
        all_attractions = []
        
        for day, locations in sorted(routes.items()):
            for loc in locations:
                if processor.is_accommodation(loc['name']):
                    all_accommodations.append((day, loc))
                else:
                    all_attractions.append(loc)
        
        if not all_attractions:
            return routes
        
        # 관광지만으로 재분배
        total_places = len(all_attractions)
        ideal_per_day = total_places / travel_duration
        
        if ideal_per_day > self.max_places_per_day:
            ideal_per_day = self.max_places_per_day
        elif ideal_per_day < self.min_places_per_day and total_places >= self.min_places_per_day:
            ideal_per_day = self.min_places_per_day
        
        # 관광지 재분배
        new_routes = {}
        location_idx = 0
        
        for day in range(travel_duration):
            day_locations = []
            
            if day < travel_duration - 1:
                places_for_day = int(ideal_per_day)
                remaining_days = travel_duration - day
                remaining_places = total_places - location_idx
                if remaining_places / remaining_days > places_for_day:
                    places_for_day += 1
                
                places_for_day = min(places_for_day, self.max_places_per_day - 1)  # 숙소 자리 확보
                end_idx = min(location_idx + places_for_day, len(all_attractions))
                day_locations = all_attractions[location_idx:end_idx]
            else:
                remaining = all_attractions[location_idx:]
                if len(remaining) > self.max_places_per_day - 1:  # 숙소 자리 확보
                    day_locations = remaining[:self.max_places_per_day - 1]
                else:
                    day_locations = remaining
            
            new_routes[day] = day_locations
            location_idx += len(day_locations)
        
        # 🆕 각 날짜에 숙소 하나씩 추가 (중복 방지)
        used_accommodations = set()
        accommodation_list = [loc for day, loc in all_accommodations]
        
        for day in range(travel_duration):
            if day in new_routes and accommodation_list:
                # 아직 사용하지 않은 숙소 찾기
                available_accommodations = [acc for acc in accommodation_list 
                                        if acc['id'] not in used_accommodations]
                
                if available_accommodations:
                    selected_accommodation = available_accommodations[0]
                    new_routes[day].append(selected_accommodation)
                    used_accommodations.add(selected_accommodation['id'])
                elif accommodation_list:  # 모든 숙소가 사용됐으면 첫 번째 재사용
                    new_routes[day].append(accommodation_list[0])
        
        # 로그 출력
        print(f"\n📊 일정 재분배 결과:")
        for day, locations in sorted(new_routes.items()):
            attraction_count = sum(1 for loc in locations if not processor.is_accommodation(loc['name']))
            accommodation_count = sum(1 for loc in locations if processor.is_accommodation(loc['name']))
            print(f"  Day {day+1}: 관광지 {attraction_count}개 + 숙소 {accommodation_count}개")
        
        return new_routes
    
    def generate_routes(self, recommendations, visit_area_df, travel_duration, coords):
        """빠른 경로 생성"""
        if not recommendations:
            return {}
        
        # travel_duration 범위 제한
        travel_duration = max(1, min(travel_duration, 30))
        
        # 유효한 좌표 확인 (기존 로직 유지)
        if coords is None or len(coords) == 0:
            # 기존 균등 분배 로직...
            pass
        
        # 🆕 숙소와 관광지 분리
        accommodations = []
        attractions = []
        processor = FastDataProcessor()  # 프로세서 인스턴스 생성
        
        for idx in recommendations:
            if idx < len(visit_area_df) and idx < len(coords):
                row = visit_area_df.iloc[idx]
                addr = row['ROAD_NM_ADDR'] if pd.notna(row['ROAD_NM_ADDR']) else row['LOTNO_ADDR']
                location = {
                    'id': row['NEW_VISIT_AREA_ID'],
                    'name': row['VISIT_AREA_NM'],
                    'coords': coords[idx],
                    'idx': idx,
                    'addr': addr,
                    'type': row.get('VISIT_AREA_TYPE_CD', 0)
                }
                
                # 숙소 여부 판단
                if processor.is_accommodation(row['VISIT_AREA_NM']):
                    accommodations.append(location)
                else:
                    attractions.append(location)
        
        if not attractions and not accommodations:
            return {}
        
        # 관광지가 없으면 숙소도 관광지로 취급
        if not attractions:
            attractions = accommodations
            accommodations = []
        
        if travel_duration == 1:
            # 1일: 단순 정렬
            result = {0: attractions[:min(8, len(attractions))]}
            # 🆕 숙소가 있으면 하나만 추가
            if accommodations:
                result[0].append(accommodations[0])
            return result
        
        # 다일: 관광지로 클러스터링/분배 (기존 로직 사용)
        if len(attractions) < travel_duration * 2:
            # 균등 분배
            places_per_day = len(attractions) // travel_duration
            routes = {}
            for day in range(travel_duration):
                start_idx = day * places_per_day
                end_idx = start_idx + places_per_day
                if day == travel_duration - 1:
                    end_idx = len(attractions)
                routes[day] = attractions[start_idx:end_idx]
        else:
            try:
                # 클러스터링 (기존 로직)
                location_coords = np.array([loc['coords'] for loc in attractions])
                n_clusters = min(travel_duration, len(attractions) // 2)
                
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
                labels = kmeans.fit_predict(location_coords)
                
                # 클러스터별 그룹화
                clusters = {}
                for i, label in enumerate(labels):
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(attractions[i])
                
                # 클러스터 순서 정렬
                cluster_centers = kmeans.cluster_centers_
                cluster_order = sorted(range(n_clusters), 
                                    key=lambda i: cluster_centers[i][1])
                
                # 일자별 배정
                routes = {}
                day = 0
                for cluster_idx in cluster_order:
                    if cluster_idx in clusters and day < travel_duration:
                        routes[day] = clusters[cluster_idx]
                        day += 1
                
                # 빈 날짜 채우기
                if len(routes) < travel_duration:
                    remaining_locs = [loc for locs in routes.values() for loc in locs]
                    locs_per_day = len(remaining_locs) // travel_duration
                    
                    routes = {}
                    for day in range(travel_duration):
                        start = day * locs_per_day
                        end = start + locs_per_day if day < travel_duration - 1 else len(remaining_locs)
                        routes[day] = remaining_locs[start:end]
                
            except Exception as e:
                print(f"⚠️  클러스터링 실패: {e}, 균등 분배 사용")
                # 클러스터링 실패 시 균등 분배
                places_per_day = len(attractions) // travel_duration
                routes = {}
                for day in range(travel_duration):
                    start_idx = day * places_per_day
                    end_idx = start_idx + places_per_day
                    if day == travel_duration - 1:
                        end_idx = len(attractions)
                    routes[day] = attractions[start_idx:end_idx]
        
        # 🆕 각 날짜에 숙소 하나씩 추가
        if accommodations:
            for day in range(travel_duration):
                if day in routes:
                    # 숙소를 날짜 순서대로 배정 (순환)
                    accommodation_idx = day % len(accommodations)
                    routes[day].append(accommodations[accommodation_idx])
        
        # 기존 균형 조정 로직 적용
        routes = self._balance_daily_routes(routes, travel_duration)
        
        return routes


# 빠른 입력 처리 함수
def process_travel_input_fast(travel_info: dict):
    """최적화된 여행 정보 전처리"""
    # 사전 초기화
    result = {col: 0 for col in [
        'TOTAL_COST_BINNED_ENCODED', 'WITH_PET', 'MONTH', 'DURATION',
        'MVMN_기타', 'MVMN_대중교통', 'MVMN_자가용'
    ] + [f'TRAVEL_PURPOSE_{i}' for i in range(1, 10)] + [
        'WHOWITH_2인여행', 'WHOWITH_가족여행', 'WHOWITH_기타',
        'WHOWITH_단독여행', 'WHOWITH_친구/지인 여행'
    ]}
    
    try:
        # 미션 처리
        missions = set(travel_info.get('mission_ENC', '').strip().split(','))
        result['WITH_PET'] = 1 if '0' in missions else 0
        
        for i in range(1, 10):
            if str(i) in missions:
                result[f'TRAVEL_PURPOSE_{i}'] = 1
        
        # 날짜 처리
        date_range = travel_info.get('date_range', '')
        if ' ~ ' in date_range:
            dates = date_range.split(' ~ ')
            start_date = datetime.strptime(dates[0].strip(), "%Y-%m-%d")
            end_date = datetime.strptime(dates[1].strip(), "%Y-%m-%d")
            
            result['MONTH'] = end_date.month
            result['DURATION'] = max(1, (end_date - start_date).days + 1)  # 최소 1일
        else:
            # 날짜가 없으면 기본값
            result['MONTH'] = datetime.now().month
            result['DURATION'] = 1
        
        # 교통수단
        mvmn_map = {'1': 'MVMN_자가용', '2': 'MVMN_대중교통'}
        mvmn_key = travel_info.get('MVMN_NM_ENC', '3')
        result[mvmn_map.get(mvmn_key, 'MVMN_기타')] = 1
        
        # 동행자
        whowith_map = {
            1: 'WHOWITH_단독여행', 2: 'WHOWITH_2인여행',
            3: 'WHOWITH_가족여행', 4: 'WHOWITH_친구/지인 여행', 5: 'WHOWITH_기타'
        }
        whowith_idx = int(travel_info.get('whowith_ENC', '5'))
        if whowith_idx in whowith_map:
            result[whowith_map[whowith_idx]] = 1
        else:
            result['WHOWITH_기타'] = 1
        
        # 비용
        result['TOTAL_COST_BINNED_ENCODED'] = int(travel_info.get('TOTAL_COST', '2'))
        
    except Exception as e:
        print(f"⚠️  입력 처리 중 오류: {e}")
        # 기본값 사용
        result['MONTH'] = datetime.now().month
        result['DURATION'] = 1
        result['MVMN_기타'] = 1
        result['WHOWITH_기타'] = 1
        result['TOTAL_COST_BINNED_ENCODED'] = 2
    
    # 벡터 생성
    travel_feature_cols = [
        'TOTAL_COST_BINNED_ENCODED', 'WITH_PET', 'MONTH', 'DURATION',
        'MVMN_기타', 'MVMN_대중교통', 'MVMN_자가용',
        'TRAVEL_PURPOSE_1', 'TRAVEL_PURPOSE_2', 'TRAVEL_PURPOSE_3',
        'TRAVEL_PURPOSE_4', 'TRAVEL_PURPOSE_5', 'TRAVEL_PURPOSE_6',
        'TRAVEL_PURPOSE_7', 'TRAVEL_PURPOSE_8', 'TRAVEL_PURPOSE_9',
        'WHOWITH_2인여행', 'WHOWITH_가족여행', 'WHOWITH_기타',
        'WHOWITH_단독여행', 'WHOWITH_친구/지인 여행'
    ]
    
    return np.array([[result[k] for k in travel_feature_cols]], dtype=np.float32)

def _regenerate_full_routes(recommender, travel_context_tensor, excluded_ids, travel_duration, top_k=50):
    """전체 경로를 재생성하는 함수"""
    print("\n🔄 전체 경로 재생성 모드")
    
    # 캐시 무효화
    recommender._embedding_cache = None
    recommender._score_cache = None
    
    # 완전히 새로운 추천 생성
    recommendations, _, _ = recommender.get_recommendations(
        travel_context_tensor,
        top_k=top_k,
        diversity_weight=0.3,
        excluded_ids=excluded_ids,
        filter_useless=True,
        consider_distance=True
    )
    
    # 새로운 경로 최적화
    travel_tensor = travel_context_tensor.cpu().numpy()
    new_routes, _ = recommender.optimize_routes(recommendations, travel_tensor)
    
    # 제외된 장소 정보 출력
    if excluded_ids:
        print(f"  ❌ {len(excluded_ids)}개 장소 제외됨")
    
    # 새로운 경로 요약
    total_places = sum(len(route) for route in new_routes.values())
    print(f"  ✅ 새로운 경로 생성: 총 {total_places}개 장소")
    
    return new_routes


def process_user_feedback(recommender, optimized_routes, travel_context_tensor, 
                         removed_place_ids, replace_only=True, unique_recommendations=None):
    """
    사용자 피드백을 처리하여 경로를 수정하는 함수
    
    Parameters:
    -----------
    recommender : FastRecommendationEngine
        추천 엔진 인스턴스
    optimized_routes : dict
        현재 최적화된 경로 {day: [locations]}
    travel_context_tensor : torch.Tensor
        여행 컨텍스트 텐서
    removed_place_ids : list
        제거할 장소들의 ID 리스트
    replace_only : bool
        True: 제거된 장소만 대체
        False: 전체 경로 재생성
    unique_recommendations : list, optional
        기존 추천 인덱스 리스트
    
    Returns:
    --------
    dict : 수정된 경로
    """
    # 제거할 ID를 set으로 변환
    excluded_ids = set(removed_place_ids)
    
    if replace_only:
        # 옵션 1: 제거된 장소만 대체
        return _replace_specific_places(recommender, optimized_routes, 
                                      travel_context_tensor, excluded_ids)
    else:
        # 옵션 2: 전체 경로 재생성
        return _regenerate_full_routes(recommender, travel_context_tensor, 
                                     excluded_ids, len(optimized_routes), top_k=200)


def _replace_specific_places(recommender, optimized_routes, travel_context_tensor, excluded_ids):
    """특정 장소만 대체하는 함수"""
    print("\n🔄 특정 장소 대체 모드")
    
    # 1. 제거할 장소 찾기 및 대체 필요 수 계산
    replacement_needed = {}
    current_place_ids = set()
    
    for day, locations in optimized_routes.items():
        replacement_needed[day] = []
        for i, loc in enumerate(locations):
            current_place_ids.add(loc['id'])
            if loc['id'] in excluded_ids:
                replacement_needed[day].append(i)
                print(f"  - Day {day+1}: '{loc['name']}' 제거 예정")
    
    # 2. 대체할 장소 찾기
    total_replacements = sum(len(indices) for indices in replacement_needed.values())
    
    if total_replacements == 0:
        print("  ℹ️ 제거할 장소가 없습니다.")
        return optimized_routes
    
    # 현재 경로에 있는 모든 ID + 제외할 ID
    all_excluded_ids = current_place_ids.union(excluded_ids)
    
    # 새로운 추천 받기
    new_recommendations, _, _ = recommender.get_recommendations(
        travel_context_tensor, 
        top_k=total_replacements + 100,  # 여유분 포함
        excluded_ids=all_excluded_ids,
        filter_useless=True,
        consider_distance=True
    )
    
    # 3. 새로운 장소 정보 생성
    new_locations = []
    used_area_ids = set()  # 이미 사용된 장소 ID 추적

    for idx in new_recommendations:
        if idx < len(recommender.visit_area_df):
            row = recommender.visit_area_df.iloc[idx]
            area_id = row['NEW_VISIT_AREA_ID']
            
            # 기존 제외 조건 + 이미 사용된 ID도 제외
            if area_id not in all_excluded_ids and area_id != 0 and area_id not in used_area_ids:
                addr = row['ROAD_NM_ADDR'] if pd.notna(row['ROAD_NM_ADDR']) else row['LOTNO_ADDR']
                new_locations.append({
                    'id': area_id,
                    'name': row['VISIT_AREA_NM'],
                    'coords': [row['X_COORD'], row['Y_COORD']],
                    'idx': idx,
                    'addr': addr,  # 주소 정보도 추가
                    'type': row.get('VISIT_AREA_TYPE_CD', 0)
                })
                used_area_ids.add(area_id)  # 사용된 ID 추가
                
                if len(new_locations) >= total_replacements:
                    break
        
    # 4. 각 날짜별로 장소 대체
    new_routes = {}
    replacement_idx = 0
    
    for day, locations in optimized_routes.items():
        new_day_locations = locations.copy()
        
        # 제거할 장소들을 새로운 장소로 대체
        for idx in sorted(replacement_needed[day], reverse=True):
            if replacement_idx < len(new_locations):
                old_name = new_day_locations[idx]['name']
                new_name = new_locations[replacement_idx]['name']
                new_day_locations[idx] = new_locations[replacement_idx]
                print(f"  ✅ Day {day+1}: '{old_name}' → '{new_name}'")
                replacement_idx += 1
            else:
                # 대체할 장소가 부족하면 제거만
                new_day_locations.pop(idx)
                print(f"  ❌ Day {day+1}: 위치 {idx+1} 제거 (대체 장소 부족)")
        
        new_routes[day] = new_day_locations
    
    print(f"\n  📊 총 {replacement_idx}개 장소 대체 완료")
    return new_routes


def main_optimized_test(travel_example, target_region) -> dict:
    """최적화된 테스트 함수"""
    print("🚀 최적화된 GNN 추천 시스템 시작!")
    print("=" * 60)
    
    import time
    start_time = time.time()
    
    try:
        # 전처리
        travel_tensor = process_travel_input_fast(travel_example)
        travel_duration = int(travel_tensor[0, 3])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"📱 사용 디바이스: {device}")
        print(f"📅 여행 기간: {travel_example.get('date_range', 'N/A')} ({travel_duration}일)")
        
        travel_context_tensor = torch.tensor(travel_tensor, dtype=torch.float32).to(device)
        
        # 추천 엔진 초기화
        model_path=f'./pickle/{target_region}/improved_travel_recommendation_model.pt'
        data_path=f'./pickle/{target_region}/improved_travel_data.pkl'
        
        recommender = FastRecommendationEngine(device, model_path, data_path)
        
        # 추천 생성
        recommendations, embeddings, _ = recommender.get_recommendations(
            travel_context_tensor, top_k=50, diversity_weight=0.3
        )
        
        # 경로 최적화
        optimized_routes, unique_recommendations = recommender.optimize_routes(
            recommendations, travel_tensor
        )
        
        end_time = time.time()
        
        print(f"\n⏱️ 처리 시간: {end_time - start_time:.2f}초")
        print("\n🗓️ 최적화된 여행 일정:")
        
        total_places = 0
        for day, route in sorted(optimized_routes.items()):
            print(f"\n📅 Day {day + 1}:")
            for i, loc in enumerate(route):
                print(f" {i+1}. [{loc['id']:3d}] {loc['name']} {loc['addr']}")
            total_places += len(route)
        
        print(f"\n총 {total_places}개 장소 추천")
        
        return optimized_routes, recommender, unique_recommendations, travel_context_tensor
        
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()
        
        # 에러 발생 시 기본 경로 반환
        return {0: [{'id': 0, 'name': '추천 생성 실패', 'coords': [0, 0], 'idx': 0, 'type': 0}]}


def feedback_usage_user(recommender:FastRecommendationEngine,
                        origin_routes:dict,
                        unique_recommendations,
                        travel_context_tensor,
                        disliked_ids:list,
                        replace_only = False):
    """피드백 활용 재추천"""

    # 사용자가 특정 장소를 싫어한다고 가정
    # 예: 첫째 날 첫 번째 장소와 둘째 날 두 번째 장소
    disliked_ids = []
    if 0 in origin_routes and len(origin_routes[0]) > 0:
        disliked_ids.append(origin_routes[0][0]['id'])
    if 1 in origin_routes and len(origin_routes[1]) > 1:
        disliked_ids.append(origin_routes[1][1]['id'])
    
    # 4. 피드백 처리 - 방법 1: 특정 장소만 대체 - replace_only = True
    # 5. 피드백 처리 - 방법 2: 전체 재생성 - replace_only = False
    new_routes = process_user_feedback(
        recommender, origin_routes, travel_context_tensor,
        disliked_ids, replace_only=replace_only, unique_recommendations=unique_recommendations)
    
    return new_routes

def get_lat_lon_kakao(address, api_key):
    """
    주소(도로명/지번)를 입력받아 위도(lat), 경도(lon) 반환
    """
    
    import requests
    # 카카오맵 주소검색 API URL
    url = "https://dapi.kakao.com/v2/local/search/address.json"

    # 요청 헤더 (API 키 입력)
    headers = {
        "Authorization": f"KakaoAK {api_key}"
    }

    # 요청 파라미터 (주소)
    params = {
        "query": address
    }

    # API 요청
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        result = response.json()
        documents = result.get('documents')
        if documents:
            # 첫번째 결과에서 좌표 꺼내오기
            lon = float(documents[0]['address']['x'])
            lat = float(documents[0]['address']['y'])
            return lat, lon
        else:
            print("주소 결과를 찾을 수 없어요.")
            return None, None
    else:
        print(f"API 요청 오류 {response.status_code}: {response.text}")
        return None, None

if __name__ == "__main__":
    # 테스트 실행
    travel_example = {
        'mission_ENC': '0,1,2',
        'date_range': '2025-09-28 - 2025-09-29',
        'TOTAL_COST': '2',
        'MVMN_NM_ENC': '2',
        'whowith_ENC': '2',
        'mission_type': 'normal'
    }
    
    route = main_optimized_test(travel_example)
    print("=" * 100)