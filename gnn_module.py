# 최적화된 GNN 추천 시스템
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

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

@jit(nopython=True, parallel=True)
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
            '아파트', '원룸', '오피스텔', '빌라', '주택', '빌딩', '상가', '모텔', '집'
        }
        self._cache = {}  # 캐싱 추가
        
    @lru_cache(maxsize=10000)
    def should_exclude_location(self, name):
        """캐싱된 위치 제외 확인"""
        if pd.isna(name):
            return False
        name_str = str(name).lower()
        
        for keyword in self.exclude_keywords:
            if keyword.lower() in name_str:
                tourist_keywords = {'관광', '테마', '파크', '랜드', '월드', '리조트', '호텔',
                                  '맛집', '식당', '카페', '박물관', '전시', '갤러리', '문화'}
                if any(tk in name_str for tk in tourist_keywords):
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
    """최적화된 추천 엔진"""
    def __init__(self, device, model_path='./pickle/improved_travel_recommendation_model.pt', 
                 data_path='./pickle/improved_travel_data.pkl'):
        torch.set_num_threads(1)
        
        # 1. 데이터 로드
        try:
            with open(data_path, 'rb') as f:
                self.data_dict = pickle.load(f)
            
            self.visit_area_df = self.data_dict['visit_area_df']
            self.graph_data = self.data_dict['graph_data']
            self.visit_scaler = self.data_dict['visit_scaler']
            self.travel_scaler = self.data_dict['travel_scaler']
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            raise RuntimeError(f"데이터 파일을 로드할 수 없습니다: {data_path}")
        
        # 2. 좌표 사전 처리 및 거리 행렬 계산
        self.coords = self.visit_area_df[['X_COORD', 'Y_COORD']].values
        # NaN 값 처리
        self.coords = np.nan_to_num(self.coords, nan=self.coords[~np.isnan(self.coords).any(axis=1)].mean(axis=0)[0] if len(self.coords[~np.isnan(self.coords).any(axis=1)]) > 0 else 0)
        self.distance_matrix = None  # 필요시 계산
        
        # 3. 디바이스 설정
        self.device = device
        self.graph_data = self.graph_data.to(self.device)
        
        # 4. 최적화된 모델 로드
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
        
        # 5. 초기화
        self.excluded_ids = set()
        self.processor = FastDataProcessor()
        self.route_generator = FastRouteGenerator()
        
        self.min_recommendations_per_day = 3
        self.min_total_recommendations = 10
        
        # 6. 캐시 초기화
        self._embedding_cache = None
        self._score_cache = None
        
    @torch.no_grad()
    def get_recommendations(self, travel_context, top_k=50, diversity_weight=0.3,
                          excluded_ids=None, filter_useless=True, consider_distance=True):
        """최적화된 추천 로직"""
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
        
        # 벡터화된 필터링
        if filter_useless:
            exclude_mask = torch.tensor([
                self.processor.should_exclude_location(name) 
                for name in self.visit_area_df['VISIT_AREA_NM']
            ], device=self.device)
            scores[exclude_mask] *= 0.3
        
        # 제외 ID 처리 (벡터화)
        if excluded_ids:
            exclude_mask = torch.tensor([
                self.visit_area_df.iloc[i]['NEW_VISIT_AREA_ID'] in excluded_ids 
                for i in range(len(scores))
            ], device=self.device)
            scores[exclude_mask] = -1.0
        
        # 유효한 점수가 있는지 확인
        valid_scores = scores > 0
        num_valid = torch.sum(valid_scores).item()
        
        if num_valid == 0:
            # 유효한 점수가 없으면 필터링 조건을 완화
            if filter_useless:
                # 필터링을 완화하여 다시 시도
                scores = preference_scores.clone()
                if excluded_ids:
                    exclude_mask = torch.tensor([
                        self.visit_area_df.iloc[i]['NEW_VISIT_AREA_ID'] in excluded_ids 
                        for i in range(len(scores))
                    ], device=self.device)
                    scores[exclude_mask] = -1.0
                valid_scores = scores > 0
                num_valid = torch.sum(valid_scores).item()
            
            if num_valid == 0:
                # 그래도 없으면 상위 점수 강제 선택
                scores = preference_scores.clone()
                valid_scores = scores > -float('inf')
                num_valid = torch.sum(valid_scores).item()
        
        # 빠른 상위 k 선택
        print(top_k, len(scores))
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
    
    def optimize_routes(self, recommendations, travel_tensor):
        """최적화된 경로 생성"""
        travel_duration = max(1, int(travel_tensor[0, 3]))  # 최소 1일 보장
        
        # 중복 제거 (벡터화)
        unique_recommendations = []
        seen_ids = set()
        
        for idx in recommendations:
            if idx < len(self.visit_area_df):
                area_id = self.visit_area_df.iloc[idx]['NEW_VISIT_AREA_ID']
                if area_id not in seen_ids and area_id != 0:
                    unique_recommendations.append(idx)
                    seen_ids.add(area_id)
        
        # 최소 개수 확보
        min_required = travel_duration * self.min_recommendations_per_day
        if len(unique_recommendations) < min_required:
            for idx in recommendations:
                if idx not in unique_recommendations and idx < len(self.visit_area_df):
                    area_id = self.visit_area_df.iloc[idx]['NEW_VISIT_AREA_ID']
                    if area_id != 0:
                        unique_recommendations.append(idx)
                        if len(unique_recommendations) >= min_required:
                            break
        
        # 빠른 경로 생성
        optimized_routes = self.route_generator.generate_routes(
            unique_recommendations, self.visit_area_df, travel_duration, self.coords
        )
        
        return optimized_routes, unique_recommendations
    
    def feedback_model(self, feedback, travel_context_tensor, travel_duration, 
                      unique_recommendations, embeddings):
        """최적화된 피드백 처리 (기존 인터페이스 유지)"""
        # 캐시 무효화
        self._embedding_cache = None
        self._score_cache = None
        
        liked_indices = [unique_recommendations[i] for i in feedback.get("liked", []) 
                        if i < len(unique_recommendations)]
        disliked_indices = [unique_recommendations[i] for i in feedback.get("disliked", []) 
                           if i < len(unique_recommendations)]
        
        # 제외 ID 설정
        excluded_ids = {self.visit_area_df.iloc[idx]['NEW_VISIT_AREA_ID'] 
                       for idx in disliked_indices}
        
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
    def __init__(self):
        self.distance_cache = {}
    
    def generate_routes(self, recommendations, visit_area_df, travel_duration, coords):
        """빠른 경로 생성"""
        if not recommendations:
            return {}
        
        # travel_duration 범위 제한
        travel_duration = max(1, min(travel_duration, 30))  # 최대 30일로 제한
        
        # 유효한 좌표 확인
        if coords is None or len(coords) == 0:
            # 좌표가 없으면 균등 분배
            places_per_day = len(recommendations) // max(1, travel_duration)
            routes = {}
            for day in range(travel_duration):
                start_idx = day * places_per_day
                end_idx = start_idx + places_per_day
                if day == travel_duration - 1:
                    end_idx = len(recommendations)
                routes[day] = []
                for idx in recommendations[start_idx:end_idx]:
                    if idx < len(visit_area_df):
                        row = visit_area_df.iloc[idx]
                        routes[day].append({
                            'id': row['NEW_VISIT_AREA_ID'],
                            'name': row['VISIT_AREA_NM'],
                            'coords': [0, 0],  # 기본값
                            'idx': idx,
                            'type': row.get('VISIT_AREA_TYPE_CD', 0)
                        })
            return routes
        
        locations = []
        for idx in recommendations:
            if idx < len(visit_area_df) and idx < len(coords):
                row = visit_area_df.iloc[idx]
                locations.append({
                    'id': row['NEW_VISIT_AREA_ID'],
                    'name': row['VISIT_AREA_NM'],
                    'coords': coords[idx],
                    'idx': idx,
                    'type': row.get('VISIT_AREA_TYPE_CD', 0)
                })
        
        if not locations:
            return {}
        
        if travel_duration == 1:
            # 1일: 단순 정렬
            return {0: locations[:min(8, len(locations))]}
        
        # 다일: K-means 클러스터링
        if len(locations) < travel_duration * 2:
            # 균등 분배
            places_per_day = len(locations) // travel_duration
            routes = {}
            for day in range(travel_duration):
                start_idx = day * places_per_day
                end_idx = start_idx + places_per_day
                if day == travel_duration - 1:
                    end_idx = len(locations)
                routes[day] = locations[start_idx:end_idx]
            return routes
        
        try:
            # 클러스터링
            location_coords = np.array([loc['coords'] for loc in locations])
            n_clusters = min(travel_duration, len(locations) // 2)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
            labels = kmeans.fit_predict(location_coords)
            
            # 클러스터별 그룹화
            clusters = {}
            for i, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(locations[i])
            
            # 클러스터 순서 정렬 (단순화)
            cluster_centers = kmeans.cluster_centers_
            cluster_order = sorted(range(n_clusters), 
                                 key=lambda i: cluster_centers[i][1])  # Y 좌표 기준
            
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
            
            return routes
            
        except Exception as e:
            print(f"⚠️  클러스터링 실패: {e}, 균등 분배 사용")
            # 클러스터링 실패 시 균등 분배
            places_per_day = len(locations) // travel_duration
            routes = {}
            for day in range(travel_duration):
                start_idx = day * places_per_day
                end_idx = start_idx + places_per_day
                if day == travel_duration - 1:
                    end_idx = len(locations)
                routes[day] = locations[start_idx:end_idx]
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
        if ' - ' in date_range:
            dates = date_range.split(' - ')
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
                                     excluded_ids, len(optimized_routes))


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
        top_k=total_replacements + 10,  # 여유분 포함
        excluded_ids=all_excluded_ids,
        filter_useless=True,
        consider_distance=True
    )
    
    # 3. 새로운 장소 정보 생성
    new_locations = []
    for idx in new_recommendations:
        if idx < len(recommender.visit_area_df):
            row = recommender.visit_area_df.iloc[idx]
            area_id = row['NEW_VISIT_AREA_ID']
            if area_id not in all_excluded_ids and area_id != 0:
                new_locations.append({
                    'id': area_id,
                    'name': row['VISIT_AREA_NM'],
                    'coords': [row['X_COORD'], row['Y_COORD']],
                    'idx': idx,
                    'type': row.get('VISIT_AREA_TYPE_CD', 0)
                })
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


def _regenerate_full_routes(recommender, travel_context_tensor, excluded_ids, travel_duration):
    """전체 경로를 재생성하는 함수"""
    print("\n🔄 전체 경로 재생성 모드")
    
    # 캐시 무효화
    recommender._embedding_cache = None
    recommender._score_cache = None
    
    # 완전히 새로운 추천 생성
    recommendations, _, _ = recommender.get_recommendations(
        travel_context_tensor,
        top_k=50,
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


def simulate_feedback_interaction(recommender, optimized_routes, travel_context_tensor, 
                                 unique_recommendations=None):
    """피드백 상호작용 시뮬레이션 함수"""
    print("\n" + "="*60)
    print("🎯 피드백 시뮬레이션")
    
    # 현재 경로에서 임의로 제거할 장소 선택
    all_places = []
    for day, locations in optimized_routes.items():
        for loc in locations:
            all_places.append(loc)
    
    if len(all_places) < 3:
        print("장소가 너무 적어 피드백 시뮬레이션을 건너뜁니다.")
        return optimized_routes
    
    # 랜덤하게 2-3개 장소 선택하여 제거
    num_to_remove = min(3, len(all_places) // 3)
    places_to_remove = random.sample(all_places, num_to_remove)
    removed_ids = [place['id'] for place in places_to_remove]
    
    print(f"\n👎 싫어요 표시된 장소:")
    for place in places_to_remove:
        print(f"  - [{place['id']}] {place['name']}")
    
    # 테스트 1: 특정 장소만 대체
    print("\n--- 테스트 1: 특정 장소만 대체 ---")
    new_routes_replace = process_user_feedback(
        recommender, optimized_routes, travel_context_tensor,
        removed_ids, replace_only=True, unique_recommendations=unique_recommendations
    )
    
    # 테스트 2: 전체 경로 재생성  
    print("\n--- 테스트 2: 전체 경로 재생성 ---")
    new_routes_full = process_user_feedback(
        recommender, optimized_routes, travel_context_tensor,
        removed_ids, replace_only=False, unique_recommendations=unique_recommendations
    )
    
    return new_routes_replace, new_routes_full


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
                                     excluded_ids, len(optimized_routes))


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
        top_k=total_replacements + 10,  # 여유분 포함
        excluded_ids=all_excluded_ids,
        filter_useless=True,
        consider_distance=True
    )
    
    # 3. 새로운 장소 정보 생성
    new_locations = []
    for idx in new_recommendations:
        if idx < len(recommender.visit_area_df):
            row = recommender.visit_area_df.iloc[idx]
            area_id = row['NEW_VISIT_AREA_ID']
            if area_id not in all_excluded_ids and area_id != 0:
                new_locations.append({
                    'id': area_id,
                    'name': row['VISIT_AREA_NM'],
                    'coords': [row['X_COORD'], row['Y_COORD']],
                    'idx': idx,
                    'type': row.get('VISIT_AREA_TYPE_CD', 0)
                })
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


def _regenerate_full_routes(recommender, travel_context_tensor, excluded_ids, travel_duration):
    """전체 경로를 재생성하는 함수"""
    print("\n🔄 전체 경로 재생성 모드")
    
    # 캐시 무효화
    recommender._embedding_cache = None
    recommender._score_cache = None
    
    # 완전히 새로운 추천 생성
    recommendations, _, _ = recommender.get_recommendations(
        travel_context_tensor,
        top_k=50,
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


def simulate_feedback_interaction(recommender, optimized_routes, travel_context_tensor, 
                                 unique_recommendations=None):
    """피드백 상호작용 시뮬레이션 함수"""
    print("\n" + "="*60)
    print("🎯 피드백 시뮬레이션")
    
    # 현재 경로에서 임의로 제거할 장소 선택
    all_places = []
    for day, locations in optimized_routes.items():
        for loc in locations:
            all_places.append(loc)
    
    if len(all_places) < 3:
        print("장소가 너무 적어 피드백 시뮬레이션을 건너뜁니다.")
        return optimized_routes
    
    # 랜덤하게 2-3개 장소 선택하여 제거
    num_to_remove = min(3, len(all_places) // 3)
    places_to_remove = random.sample(all_places, num_to_remove)
    removed_ids = [place['id'] for place in places_to_remove]
    
    print(f"\n👎 싫어요 표시된 장소:")
    for place in places_to_remove:
        print(f"  - [{place['id']}] {place['name']}")
    
    # 테스트 1: 특정 장소만 대체
    print("\n--- 테스트 1: 특정 장소만 대체 ---")
    new_routes_replace = process_user_feedback(
        recommender, optimized_routes, travel_context_tensor,
        removed_ids, replace_only=True, unique_recommendations=unique_recommendations
    )
    
    # 테스트 2: 전체 경로 재생성  
    print("\n--- 테스트 2: 전체 경로 재생성 ---")
    new_routes_full = process_user_feedback(
        recommender, optimized_routes, travel_context_tensor,
        removed_ids, replace_only=False, unique_recommendations=unique_recommendations
    )
    
    return new_routes_replace, new_routes_full


def main_optimized_test(travel_example) -> dict:
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
        recommender = FastRecommendationEngine(device)
        
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
                print(f" {i+1}. [{loc['id']:3d}] {loc['name']}")
            total_places += len(route)
        
        print(f"\n총 {total_places}개 장소 추천")
        
        # 피드백 시뮬레이션 (옵션)
        if False:  # 피드백 테스트 비활성화 (필요시 True로 변경)
            replaced_routes, regenerated_routes = simulate_feedback_interaction(
                recommender, optimized_routes, travel_context_tensor, unique_recommendations
            )
            
            # 결과 비교
            print("\n" + "="*60)
            print("📊 피드백 처리 결과 비교")
            
            print("\n[대체 모드 결과]")
            for day, route in sorted(replaced_routes.items()):
                print(f"Day {day + 1}: {len(route)}개 장소")
            
            print("\n[재생성 모드 결과]")  
            for day, route in sorted(regenerated_routes.items()):
                print(f"Day {day + 1}: {len(route)}개 장소")
        
        return optimized_routes
        
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()
        
        # 에러 발생 시 기본 경로 반환
        return {0: [{'id': 0, 'name': '추천 생성 실패', 'coords': [0, 0], 'idx': 0, 'type': 0}]}


# 별도 사용 예제 함수
def example_feedback_usage():
    """피드백 함수 사용 예제"""
    # 1. 초기 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    recommender = FastRecommendationEngine(device)
    
    travel_info = {
        'mission_ENC': '1,2,3',
        'date_range': '2025-10-01 - 2025-10-03',
        'TOTAL_COST': '3',
        'MVMN_NM_ENC': '1',
        'whowith_ENC': '3',
        'mission_type': 'normal'
    }
    
    # 2. 초기 추천 생성
    travel_tensor = process_travel_input_fast(travel_info)
    travel_context_tensor = torch.tensor(travel_tensor, dtype=torch.float32).to(device)
    
    recommendations, _, _ = recommender.get_recommendations(travel_context_tensor)
    optimized_routes, unique_recommendations = recommender.optimize_routes(recommendations, travel_tensor)
    
    # 3. 사용자가 특정 장소를 싫어한다고 가정
    # 예: 첫째 날 첫 번째 장소와 둘째 날 두 번째 장소
    disliked_ids = []
    if 0 in optimized_routes and len(optimized_routes[0]) > 0:
        disliked_ids.append(optimized_routes[0][0]['id'])
    if 1 in optimized_routes and len(optimized_routes[1]) > 1:
        disliked_ids.append(optimized_routes[1][1]['id'])
    
    # 4. 피드백 처리 - 방법 1: 특정 장소만 대체
    new_routes_v1 = process_user_feedback(
        recommender, optimized_routes, travel_context_tensor,
        disliked_ids, replace_only=True, unique_recommendations=unique_recommendations
    )
    
    # 5. 피드백 처리 - 방법 2: 전체 재생성
    new_routes_v2 = process_user_feedback(
        recommender, optimized_routes, travel_context_tensor,
        disliked_ids, replace_only=False
    )
    
    return new_routes_v1, new_routes_v2


# 별도 사용 예제 함수
def example_feedback_usage():
    """피드백 함수 사용 예제"""
    # 1. 초기 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    recommender = FastRecommendationEngine(device)
    
    travel_info = {
        'mission_ENC': '1,2,3',
        'date_range': '2025-10-01 - 2025-10-03',
        'TOTAL_COST': '3',
        'MVMN_NM_ENC': '1',
        'whowith_ENC': '3',
        'mission_type': 'normal'
    }
    
    # 2. 초기 추천 생성
    travel_tensor = process_travel_input_fast(travel_info)
    travel_context_tensor = torch.tensor(travel_tensor, dtype=torch.float32).to(device)
    
    recommendations, _, _ = recommender.get_recommendations(travel_context_tensor)
    optimized_routes, unique_recommendations = recommender.optimize_routes(recommendations, travel_tensor)
    
    # 3. 사용자가 특정 장소를 싫어한다고 가정
    # 예: 첫째 날 첫 번째 장소와 둘째 날 두 번째 장소
    disliked_ids = []
    if 0 in optimized_routes and len(optimized_routes[0]) > 0:
        disliked_ids.append(optimized_routes[0][0]['id'])
    if 1 in optimized_routes and len(optimized_routes[1]) > 1:
        disliked_ids.append(optimized_routes[1][1]['id'])
    
    # 4. 피드백 처리 - 방법 1: 특정 장소만 대체
    new_routes_v1 = process_user_feedback(
        recommender, optimized_routes, travel_context_tensor,
        disliked_ids, replace_only=True, unique_recommendations=unique_recommendations
    )
    
    # 5. 피드백 처리 - 방법 2: 전체 재생성
    new_routes_v2 = process_user_feedback(
        recommender, optimized_routes, travel_context_tensor,
        disliked_ids, replace_only=False
    )
    
    return new_routes_v1, new_routes_v2


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