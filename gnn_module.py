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

whowith_options = [
    ("단독여행", ["나홀로 여행"]),
    ("2인여행", ["커플", "부부"]),
    ("가족여행", ["자녀동반", "부모 동반", "3대 동반 여행"]),
    ("친구/지인 여행", ["3인 이상 친구"]),
    ("기타", ["기타"])
]


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import HeteroData
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
import random
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ImprovedTravelGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, travel_context_dim, 
                 num_heads=4, dropout=0.2):
        super().__init__()
        
        # GNN layers with edge features
        self.gat1 = GATConv(in_channels, hidden_channels // num_heads, 
                           heads=num_heads, dropout=dropout, concat=True, 
                           edge_dim=5)  # edge features 고려
        self.gat2 = GATConv(hidden_channels, hidden_channels // num_heads, 
                           heads=num_heads, dropout=dropout, concat=True,
                           edge_dim=5)
        self.gat3 = GATConv(hidden_channels, out_channels, 
                           heads=1, dropout=dropout, concat=False,
                           edge_dim=5)
        
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)
        
        # Travel context encoder
        self.travel_encoder = nn.Sequential(
            nn.Linear(travel_context_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )
        
        # Distance-aware attention module
        self.distance_attention = nn.Sequential(
            nn.Linear(2, hidden_channels // 2),  # x, y coordinates
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid()
        )
        
        # Fusion network
        self.fusion_net = nn.Sequential(
            nn.Linear(out_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        # Preference head
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
        
        # Extract coordinates for distance attention
        coords = x[:, :2]  # Assuming first two features are X_COORD, Y_COORD
        
        # GNN layers with edge features
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
        
        # Apply distance-based attention
        distance_weights = self.distance_attention(coords)
        graph_embedding = graph_embedding * distance_weights
        
        # Travel context encoding
        travel_embedding = self.travel_encoder(travel_context)
        travel_embedding_expanded = travel_embedding.expand(graph_embedding.size(0), -1)
        
        # Fusion
        fused_features = torch.cat([graph_embedding, travel_embedding_expanded], dim=1)
        final_embedding = self.fusion_net(fused_features)
        
        # Preference scores
        preference_scores = self.preference_head(final_embedding)
        
        return final_embedding, preference_scores

class EnhancedDataProcessor:
    def __init__(self):
        self.visit_scaler = RobustScaler()
        self.travel_scaler = StandardScaler()
        # 제외할 키워드 목록
        self.exclude_keywords = [
            '역', '터미널', '공항', '휴게소', '정류장', '톨게이트', '교차로', '출구', '입구',
            'IC', 'JC', '나들목', '분기점', '요금소', '주차장', '주유소', '충전소',
            '아파트', '원룸', '오피스텔', '빌라', '주택', '빌딩', '상가', '모텔'
        ]
        
    def should_exclude_location(self, name):
        """위치를 제외해야 하는지 확인"""
        if pd.isna(name):
            return False
        name_str = str(name).lower()
        
        # 특정 키워드가 포함되고, 다른 유용한 단어가 없는 경우 제외
        for keyword in self.exclude_keywords:
            if keyword.lower() in name_str:
                # 예외 처리: 관광지로서의 역할이 있는 경우
                tourist_keywords = ['관광', '테마', '파크', '랜드', '월드', '리조트', '호텔', 
                                   '맛집', '식당', '카페', '박물관', '전시', '갤러리', '문화']
                if any(tk in name_str for tk in tourist_keywords):
                    continue
                return True
        return False
        
    def process_visit_area_features(self, visit_area_df):
        visit_area_df = visit_area_df.copy()
        
        # 좌표 결측치 처리
        visit_area_df['X_COORD'] = visit_area_df['X_COORD'].fillna(visit_area_df['X_COORD'].mean())
        visit_area_df['Y_COORD'] = visit_area_df['Y_COORD'].fillna(visit_area_df['Y_COORD'].mean())
        visit_area_df['VISIT_CHC_REASON_CD'] = visit_area_df['VISIT_CHC_REASON_CD'].fillna(0)
        
        features = visit_area_df[['X_COORD', 'Y_COORD']].copy()
        
        # One-hot encoding
        type_onehot = pd.get_dummies(visit_area_df['VISIT_AREA_TYPE_CD'], prefix='type')
        reason_onehot = pd.get_dummies(visit_area_df['VISIT_CHC_REASON_CD'].fillna(0), prefix='reason')
        
        # 정규화된 만족도 점수
        for col in ['DGSTFN', 'REVISIT_INTENTION', 'RCMDTN_INTENTION']:
            visit_area_df[col] = visit_area_df[col].fillna(3)
            visit_area_df[f'{col}_norm'] = (visit_area_df[col] - 1) / 4.0
        
        # 인기도 점수
        visit_area_df['popularity_score'] = (
            visit_area_df['DGSTFN_norm'] * 0.4 + 
            visit_area_df['REVISIT_INTENTION_norm'] * 0.3 + 
            visit_area_df['RCMDTN_INTENTION_norm'] * 0.3
        )
        
        # 제외할 장소에 대한 페널티 추가
        exclude_penalty = visit_area_df['VISIT_AREA_NM'].apply(self.should_exclude_location).astype(float) * -0.5
        
        # 모든 특성 결합
        features = pd.concat([
            features, type_onehot, reason_onehot,
            visit_area_df[['DGSTFN_norm', 'REVISIT_INTENTION_norm', 'RCMDTN_INTENTION_norm', 'popularity_score']],
            pd.DataFrame({'exclude_penalty': exclude_penalty})
        ], axis=1)
        
        return self.visit_scaler.fit_transform(features.values.astype(np.float32))
    
    def create_enhanced_edges(self, move_df, visit_area_df):
        edges = []
        edge_weights = []
        
        for travel_id, group in move_df.groupby("TRAVEL_ID"):
            group = group.sort_values("TRIP_ID").reset_index(drop=True)
            
            for i in range(1, len(group)):
                from_id = group.loc[i-1, "END_NEW_ID"]
                to_id = group.loc[i, "END_NEW_ID"]
                
                if pd.notna(from_id) and pd.notna(to_id):
                    duration = group.loc[i, "DURATION_MINUTES"] if "DURATION_MINUTES" in group.columns else 0
                    transport = group.loc[i, "MVMN_CD_1"] if "MVMN_CD_1" in group.columns else 0
                    
                    edges.append([int(from_id), int(to_id), duration, transport])
                    edge_weights.append(1.0)
        
        edges_df = pd.DataFrame(edges, columns=["FROM_ID", "TO_ID", "DURATION_MINUTES", "MVMN_CD_1"])
        
        # 교통수단 원핫 인코딩
        edges_df["MVMN_TYPE"] = edges_df["MVMN_CD_1"].apply(
            lambda code: "drive" if code in [1,2,3] else "public" if code in [4,5,6,7,8,9,10,11,12,13,50] else "other"
        )
        edges_df["is_drive"] = (edges_df["MVMN_TYPE"] == "drive").astype(int)
        edges_df["is_public"] = (edges_df["MVMN_TYPE"] == "public").astype(int)
        edges_df["is_other"] = (edges_df["MVMN_TYPE"] == "other").astype(int)
        
        edge_index = torch.tensor(edges_df[["FROM_ID", "TO_ID"]].values.T, dtype=torch.long)
        edge_attr = torch.tensor(np.column_stack([
            edges_df[["DURATION_MINUTES"]].fillna(0).values,
            edges_df[["is_drive", "is_public", "is_other"]].values,
            np.array(edge_weights).reshape(-1, 1)
        ]), dtype=torch.float32)
        
        return edge_index, edge_attr

class SmartRecommendationEngine:
    def __init__(self, device, model_path='./pickle/improved_travel_recommendation_model.pt', data_path='./pickle/improved_travel_data.pkl'):
        # 1. 데이터 로드
        with open(data_path, 'rb') as f:
            self.data_dict = pickle.load(f)
        
        self.visit_area_df = self.data_dict['visit_area_df']
        self.graph_data = self.data_dict['graph_data']
        self.visit_scaler = self.data_dict['visit_scaler']
        self.travel_scaler = self.data_dict['travel_scaler']
        
        # 2. 디바이스 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.graph_data = self.graph_data.to(self.device)
        
        # 3. 모델 로드
        checkpoint = torch.load(model_path, map_location=self.device)
        model_config = checkpoint['model_config']
        
        self.model = ImprovedTravelGNN(**model_config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 4. 피드백 시스템 초기화
        self.excluded_ids = set()
        self.device = device
        
        self.user_feedback_history = []
        self.preference_weights = None
        self.processor = EnhancedDataProcessor()
        self.route_generator = OptimizedRouteGenerator(distance_threshold_km=50)  # 거리 임계값 줄임
    
    def feedback_model(self, feedback, travel_context_tensor, travel_duration, unique_recommendations, embeddings):
        liked_indices = [unique_recommendations[i] for i in feedback["liked"] if i < len(unique_recommendations)]
        disliked_indices = [unique_recommendations[i] for i in feedback["disliked"] if i < len(unique_recommendations)]
        
        if liked_indices:
            print(f"👍 좋아요: {[self.visit_area_df.iloc[idx]['VISIT_AREA_NM'] for idx in liked_indices]}")
        if disliked_indices:
            print(f"👎 싫어요: {[self.visit_area_df.iloc[idx]['VISIT_AREA_NM'] for idx in disliked_indices]}")
        
        self.update_with_feedback(liked_indices, disliked_indices, embeddings)
        
        # 제외된 항목 반영
        excluded_ids = {self.visit_area_df.iloc[idx]['NEW_VISIT_AREA_ID'] for idx in disliked_indices}
        
        recommendations, embeddings, _ = self.get_recommendations(
            travel_context_tensor, top_k=30, diversity_weight=0.3, 
            excluded_ids=excluded_ids, filter_useless=True, consider_distance=True
        )
        
        unique_recommendations, seen_ids = [], set()
        for idx in recommendations:
            if idx < len(self.visit_area_df):
                row = self.visit_area_df.iloc[idx]
                area_id = row['NEW_VISIT_AREA_ID']
                name = row['VISIT_AREA_NM']
                
                if (area_id not in seen_ids and 
                    area_id not in excluded_ids and 
                    area_id != 0 and
                    not self.processor.should_exclude_location(name)):
                    unique_recommendations.append(idx)
                    seen_ids.add(area_id)
                
                if len(unique_recommendations) >= 15:
                    break
        
        optimized_routes = self.route_generator.generate_daily_routes(
            unique_recommendations, self.visit_area_df, travel_duration
        )
        
        return optimized_routes
    
    # 중복 방문지 제거 및 유효성 검사
    def optimize_routes(self, recommendations, travel_tensor):    
        unique_recommendations, seen_ids = [], set()
        for idx in recommendations:
            if idx < len(self.visit_area_df):
                row = self.visit_area_df.iloc[idx]
                area_id = row['NEW_VISIT_AREA_ID']
                name = row['VISIT_AREA_NM']
                
                # 중복 체크 및 쓸모없는 장소 재확인
                if (area_id not in seen_ids and 
                    area_id != 0 and 
                    not self.processor.should_exclude_location(name)):
                    unique_recommendations.append(idx)
                    seen_ids.add(area_id)
                
                if len(unique_recommendations) >= 15:  # 여유있게 선택
                    break
        
        # 최적화 경로 생성
        travel_duration = int(travel_tensor[0, 3])
        optimized_routes = self.route_generator.generate_daily_routes(
            unique_recommendations, self.visit_area_df, travel_duration
        )
        return optimized_routes, unique_recommendations
    
    # SmartRecommendationEngine 클래스 내 get_recommendations 메소드
    def get_recommendations(self, travel_context, top_k=10, diversity_weight=0.3, 
                            excluded_ids=None, filter_useless=True, consider_distance=True):

        self.model.eval()
        data = self.graph_data
        
        with torch.no_grad():
            embeddings, preference_scores = self.model(data, travel_context)
        
        scores = preference_scores.squeeze()

        if filter_useless:
            for idx in range(len(scores)):
                name = self.visit_area_df.iloc[idx]['VISIT_AREA_NM']
                if self.processor.should_exclude_location(name):
                    scores[idx] *= 0.1  # 너무 낮지 않도록 수정 가능 (예: 0.3)

        if excluded_ids:
            for exclude_id in excluded_ids:
                matching_indices = self.visit_area_df[
                    self.visit_area_df['NEW_VISIT_AREA_ID'] == exclude_id
                ].index.tolist()
                for idx in matching_indices:
                    scores[idx] *= 0.1
        
        # 추천 점수 최소 임계값 설정
        min_allowed_score = 0.01
        scores[scores < min_allowed_score] = min_allowed_score

        recommendations = self._distance_aware_recommendation(
            scores, embeddings, top_k, diversity_weight
        )
        
        return recommendations, embeddings, preference_scores

    
    def _distance_aware_recommendation(self, scores, embeddings, top_k, diversity_weight):
        """거리를 고려한 순차적 추천"""
        recommendations = []
        remaining_indices = [i for i in range(len(scores)) if scores[i] > 0]
        
        if not remaining_indices:
            return recommendations
        
        # 점수가 높은 상위 후보 중에서 시작점 선택
        valid_scores = [(i, scores[i].item()) for i in remaining_indices]
        valid_scores = sorted(valid_scores, key=lambda x: x[1], reverse=True)
        
        # 상위 10개 중에서 랜덤하게 시작
        top_candidates = valid_scores[:10]
        if top_candidates:
            start_idx = random.choice(top_candidates)[0]
            recommendations.append(start_idx)
            remaining_indices.remove(start_idx)
        
        # 나머지 추천: 거리와 점수를 동시에 고려
        while len(recommendations) < top_k and remaining_indices:
            last_idx = recommendations[-1]
            last_coords = self.visit_area_df.iloc[last_idx][['X_COORD', 'Y_COORD']].values
            
            best_score = -float('inf')
            best_idx = None
            
            for idx in remaining_indices:
                if scores[idx] <= 0:
                    continue
                
                # 거리 계산
                curr_coords = self.visit_area_df.iloc[idx][['X_COORD', 'Y_COORD']].values
                distance = np.sqrt(np.sum((last_coords - curr_coords) ** 2))
                
                # 거리 페널티 (가까울수록 높은 점수)
                distance_score = 1 / (1 + distance * 0.1)  # 거리가 멀수록 점수 감소
                
                # 다양성 계산
                similarities = []
                for rec_idx in recommendations:
                    sim = F.cosine_similarity(
                        embeddings[idx:idx+1], 
                        embeddings[rec_idx:rec_idx+1]
                    ).item()
                    similarities.append(sim)
                
                diversity = 1 - max(similarities) if similarities else 1
                
                # 최종 점수: 선호도 + 거리 + 다양성
                relevance = scores[idx].item()
                final_score = (
                    0.4 * relevance +  # 선호도
                    0.4 * distance_score +  # 거리
                    0.2 * diversity  # 다양성
                )
                
                if final_score > best_score:
                    best_score = final_score
                    best_idx = idx
            
            if best_idx is not None:
                recommendations.append(best_idx)
                remaining_indices.remove(best_idx)
            else:
                break
        
        return recommendations
    
    def _mmr_recommendation(self, scores, embeddings, top_k, diversity_weight):
        """기존 MMR 기반 추천 (fallback)"""
        recommendations = []
        remaining_indices = list(range(len(scores)))
        
        # 첫 번째 추천
        if remaining_indices:
            valid_scores = [(i, scores[i].item()) for i in remaining_indices if scores[i] > 0]
            if valid_scores:
                valid_scores = sorted(valid_scores, key=lambda x: x[1], reverse=True)
                top_candidates = valid_scores[:5]
                if top_candidates:
                    best_idx = random.choice(top_candidates)[0]
                    recommendations.append(best_idx)
                    remaining_indices.remove(best_idx)
        
        # 나머지 추천
        for _ in range(min(top_k - 1, len(remaining_indices))):
            if not remaining_indices:
                break
                
            best_score = -float('inf')
            best_idx = None
            
            for idx in remaining_indices:
                relevance = scores[idx].item()
                
                if relevance < 0:
                    continue
                
                # 다양성 계산
                similarities = []
                for rec_idx in recommendations:
                    sim = F.cosine_similarity(
                        embeddings[idx:idx+1], 
                        embeddings[rec_idx:rec_idx+1]
                    ).item()
                    similarities.append(sim)
                
                diversity = 1 - max(similarities) if similarities else 1
                final_score = (1 - diversity_weight) * relevance + diversity_weight * diversity
                
                if final_score > best_score:
                    best_score = final_score
                    best_idx = idx
            
            if best_idx is not None:
                recommendations.append(best_idx)
                remaining_indices.remove(best_idx)
        
        return recommendations
    
    def _apply_preference_weights(self, scores, embeddings):
        """피드백 기반 점수 조정"""
        if not self.preference_weights:
            return scores
            
        adjusted_scores = scores.clone()
        
        for i, row in self.visit_area_df.iterrows():
            if i >= len(adjusted_scores):
                break
                
            area_type = row.get('VISIT_AREA_TYPE_CD', 0)
            
            # 선호 타입이면 점수 증가
            if area_type in self.preference_weights.get('preferred_types', []):
                adjusted_scores[i] *= 1.2
            
            # 비선호 타입이면 점수 감소
            if area_type in self.preference_weights.get('avoided_types', []):
                adjusted_scores[i] *= 0.8
        
        return adjusted_scores
    
    def update_with_feedback(self, liked_items, disliked_items, embeddings):
        """사용자 피드백 업데이트"""
        feedback = {
            'liked': liked_items,
            'disliked': disliked_items,
            'embeddings': embeddings.cpu().numpy()
        }
        self.user_feedback_history.append(feedback)
        
        self.preference_weights = self._calculate_preference_weights()
        
        print(f"✅ 피드백 업데이트 완료: 좋아요 {len(liked_items)}개, 싫어요 {len(disliked_items)}개")
        
        return self.preference_weights
    
    def _calculate_preference_weights(self):
        """피드백 히스토리를 바탕으로 선호도 가중치 계산"""
        if not self.user_feedback_history:
            return None
            
        liked_features = []
        disliked_features = []
        
        for feedback in self.user_feedback_history:
            for item_idx in feedback['liked']:
                if item_idx < len(self.visit_area_df):
                    liked_features.append(self.visit_area_df.iloc[item_idx])
            
            for item_idx in feedback['disliked']:
                if item_idx < len(self.visit_area_df):
                    disliked_features.append(self.visit_area_df.iloc[item_idx])
        
        preferred_types = [item.get('VISIT_AREA_TYPE_CD', 0) for item in liked_features]
        avoided_types = [item.get('VISIT_AREA_TYPE_CD', 0) for item in disliked_features]
        
        return {
            'preferred_types': list(set(preferred_types)),
            'avoided_types': list(set(avoided_types)),
            'preferred_regions': [(item.get('X_COORD', 0), item.get('Y_COORD', 0)) for item in liked_features]
        }

class OptimizedRouteGenerator:
    def __init__(self, distance_threshold_km=50):  # 임계값을 50km로 줄임
        self.distance_threshold_km = distance_threshold_km
        
    
    def calculate_distance(self, coord1, coord2):
        from math import radians, cos, sin, sqrt, atan2
        try:
            R = 6371  # 지구 반경(km)

            lat1, lon1 = radians(coord1[1]), radians(coord1[0])
            lat2, lon2 = radians(coord2[1]), radians(coord2[0])

            dlat = lat2 - lat1
            dlon = lon2 - lon1

            a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))

            distance = R * c
            return distance
        except:
            # 예외 발생 시 매우 큰 값을 반환해 이상치 처리
            return float('inf')
        
        
    def _two_opt_improvement(self, route, coords):
        """2-opt 알고리즘으로 경로 개선"""
        n = len(route)
        if n <= 3:
            return route
            
        improved = True
        best_route = route[:]
        
        while improved:
            improved = False
            for i in range(1, n - 2):
                for j in range(i + 1, n):
                    if j - i == 1:
                        continue
                    
                    new_route = best_route[:]
                    new_route[i:j] = reversed(new_route[i:j])
                    
                    if self._route_distance(new_route, coords) < self._route_distance(best_route, coords):
                        best_route = new_route
                        improved = True
        
        return best_route
    
    def _route_distance(self, route, coords):
        """경로의 총 거리 계산"""
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += self.calculate_distance(
                coords[route[i]], 
                coords[route[i + 1]]
            )
        return total_distance
    
    def _solve_tsp_simple(self, locations):
        """간단한 TSP 해법"""
        if len(locations) <= 2:
            return locations
            
        coords = np.array([loc['coords'] for loc in locations])
        n = len(coords)
        
        # Nearest Neighbor
        unvisited = set(range(1, n))
        current = 0
        route = [0]
        
        while unvisited:
            nearest = min(unvisited, 
                         key=lambda x: self.calculate_distance(coords[current], coords[x]))
            route.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        # 2-opt improvement
        route = self._two_opt_improvement(route, coords)
        
        return [locations[i] for i in route]
        
    def remove_outliers(self, coords, locations):
        """거리 이상치 제거"""
        if len(coords) <= 2:
            return coords, locations
            
        # 중심점 계산
        center = np.mean(coords, axis=0)
        
        # 각 점과 중심점의 거리 계산
        distances = [self.calculate_distance(coord, center) for coord in coords]
        
        # 거리 기준으로 필터링 (너무 먼 곳 제외)
        filtered_indices = []
        for i, dist in enumerate(distances):
            if dist <= self.distance_threshold_km:
                filtered_indices.append(i)
        
        # 최소 장소 수 유지
        if len(filtered_indices) < 6:
            # 거리순으로 정렬하여 가까운 곳부터 선택
            sorted_indices = np.argsort(distances)
            filtered_indices = sorted_indices[:min(10, len(distances))].tolist()
        
        filtered_coords = [coords[i] for i in filtered_indices]
        filtered_locations = [locations[i] for i in filtered_indices]
        
        return np.array(filtered_coords), filtered_locations
        
    def generate_daily_routes(self, recommendations, visit_area_df, travel_duration, 
                            optimization_method='region_based'):
        """지역 기반 일별 최적 경로 생성"""
        if travel_duration <= 0:
            travel_duration = 1
            
        coords = []
        locations = []
        
        for idx in recommendations:
            if idx < len(visit_area_df):
                row = visit_area_df.iloc[idx]
                coords.append([row['X_COORD'], row['Y_COORD']])
                locations.append({
                    'id': row['NEW_VISIT_AREA_ID'],
                    'name': row['VISIT_AREA_NM'],
                    'coords': [row['X_COORD'], row['Y_COORD']],
                    'idx': idx,
                    'type': row.get('VISIT_AREA_TYPE_CD', 0)
                })
        
        if len(coords) == 0:
            return {}
            
        coords = np.array(coords)
        coords = np.nan_to_num(coords, nan=0.0)
        
        # 이상치 제거
        coords, locations = self.remove_outliers(coords, locations)
        
        # 일수에 맞게 장소 수 조정
        places_per_day = max(3, min(5, len(locations) // travel_duration))
        total_places = min(places_per_day * travel_duration, len(locations))
        
        # 지역별 클러스터링으로 일정 생성
        if travel_duration == 1:
            # 1일 여행: TSP로 최적 경로만 생성
            optimized_order = self._solve_tsp_with_start(locations)
            return {0: optimized_order[:places_per_day]}
        else:
            # 다일 여행: 지역별로 묶기
            return self._create_regional_routes(coords, locations, travel_duration, places_per_day)
    
    def _create_regional_routes(self, coords, locations, travel_duration, places_per_day):
        """지역별로 묶어서 일정 생성"""
        from sklearn.cluster import KMeans
        
        # 적절한 클러스터 수 결정
        n_clusters = min(travel_duration, len(locations) // 2)
        
        if n_clusters < 2:
            # 장소가 너무 적으면 균등 분배
            return self._equal_distribution(locations, travel_duration, places_per_day)
        
        # K-means 클러스터링
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coords)
        
        # 클러스터별 그룹화
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append((i, locations[i]))
        
        # 클러스터 간 거리 계산 (순서 결정용)
        cluster_centers = kmeans.cluster_centers_
        cluster_order = self._order_clusters(cluster_centers)
        
        # 일자별 배정
        daily_routes = {}
        locations_per_day = len(locations) // travel_duration
        remainder = len(locations) % travel_duration
        
        current_day = 0
        current_day_locations = []
        
        for cluster_idx in cluster_order:
            if cluster_idx in clusters:
                cluster_locations = [loc for _, loc in clusters[cluster_idx]]
                
                # TSP로 클러스터 내 최적 경로
                if len(cluster_locations) > 1:
                    cluster_locations = self._solve_tsp_simple(cluster_locations)
                
                for loc in cluster_locations:
                    current_day_locations.append(loc)
                    
                    # 일자별 할당량 체크
                    day_quota = locations_per_day + (1 if current_day < remainder else 0)
                    if len(current_day_locations) >= day_quota:
                        daily_routes[current_day] = current_day_locations
                        current_day += 1
                        current_day_locations = []
                        
                        if current_day >= travel_duration:
                            break
                
                if current_day >= travel_duration:
                    break
        
        # 남은 장소 처리
        if current_day_locations and current_day < travel_duration:
            daily_routes[current_day] = current_day_locations
        
        # 각 일자별 장소 수 조정
        return self._adjust_daily_balance(daily_routes, places_per_day)
    
    def _order_clusters(self, cluster_centers):
        """클러스터를 가까운 순서로 정렬"""
        n_clusters = len(cluster_centers)
        if n_clusters <= 1:
            return list(range(n_clusters))
        
        # 첫 클러스터는 가장 남쪽 (또는 서쪽)
        start_idx = np.argmin(cluster_centers[:, 1])  # Y 좌표 기준
        
        visited = [start_idx]
        current = start_idx
        
        while len(visited) < n_clusters:
            min_dist = float('inf')
            next_idx = None
            
            for i in range(n_clusters):
                if i not in visited:
                    dist = self.calculate_distance(
                        cluster_centers[current], 
                        cluster_centers[i]
                    )
                    if dist < min_dist:
                        min_dist = dist
                        next_idx = i
            
            if next_idx is not None:
                visited.append(next_idx)
                current = next_idx
        
        return visited
    
    def _equal_distribution(self, locations, travel_duration, places_per_day):
        """균등 분배"""
        daily_routes = {}
        locations_per_day = len(locations) // travel_duration
        remainder = len(locations) % travel_duration
        
        start_idx = 0
        for day in range(travel_duration):
            count = locations_per_day + (1 if day < remainder else 0)
            count = min(count, places_per_day)  # 일별 최대 장소 수 제한
            daily_routes[day] = locations[start_idx:start_idx + count]
            start_idx += count
        
        return daily_routes
    
    def _adjust_daily_balance(self, daily_routes, target_size):
        """일별 장소 수 균형 조정"""
        adjusted = {}
        
        for day, locations in daily_routes.items():
            if len(locations) > target_size:
                # 너무 많으면 잘라내기
                adjusted[day] = locations[:target_size]
            elif len(locations) < 2:
                # 너무 적으면 다른 날에서 가져오기
                continue
            else:
                adjusted[day] = locations
        
        return adjusted
    
    def _solve_tsp_with_start(self, locations):
        """시작점을 고려한 TSP"""
        if len(locations) <= 2:
            return locations
        
        coords = np.array([loc['coords'] for loc in locations])
        n = len(coords)
        
        # 가장 접근하기 쉬운 곳을 시작점으로 (가장 서쪽 또는 남쪽)
        start = np.argmin(coords[:, 1])  # Y 좌표 기준
        
        # Nearest Neighbor from start
        unvisited = set(range(n))
        unvisited.remove(start)
        current = start
        route = [start]
        
        while unvisited:
            nearest = min(unvisited, 
                         key=lambda x: self.calculate_distance(coords[current], coords[x]))
            route.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        # 2-opt improvement
        route = self._two_opt_improvement(route, coords)
        
        return [locations[i] for i in route]
        

def process_travel_input(travel_info: dict):
    """여행 정보 전처리 함수"""
    travel_feature_cols = [
        'TOTAL_COST_BINNED_ENCODED', 'WITH_PET', 'MONTH', 'DURATION',
        'MVMN_기타', 'MVMN_대중교통', 'MVMN_자가용',
        'TRAVEL_PURPOSE_1', 'TRAVEL_PURPOSE_2', 'TRAVEL_PURPOSE_3',
        'TRAVEL_PURPOSE_4', 'TRAVEL_PURPOSE_5', 'TRAVEL_PURPOSE_6',
        'TRAVEL_PURPOSE_7', 'TRAVEL_PURPOSE_8', 'TRAVEL_PURPOSE_9',
        'WHOWITH_2인여행', 'WHOWITH_가족여행', 'WHOWITH_기타',
        'WHOWITH_단독여행', 'WHOWITH_친구/지인 여행'
    ]
    
    # 반려동물 동반
    travel_info['mission_ENC'] = travel_info['mission_ENC'].strip().split(',')
    travel_info['WITH_PET'] = 1 if '0' in travel_info['mission_ENC'] else 0
        
    # 여행 목적
    for i in range(1, 10):
        travel_info[f'TRAVEL_PURPOSE_{i}'] = 1 if str(i) in travel_info['mission_ENC'] else 0
        
    # 날짜 처리
    dates = travel_info['date_range'].split(' - ')
    start_date = datetime.strptime(dates[0].strip(), "%Y-%m-%d")
    end_date = datetime.strptime(dates[1].strip(), "%Y-%m-%d")
    
    travel_info['MONTH'] = end_date.month
    travel_info['DURATION'] = (end_date - start_date).days + 1  # +1 추가
    
    # 교통수단
    for m in ['자가용', '대중교통', '기타']:
        travel_info[f"MVMN_{m}"] = 0
    
    if travel_info['MVMN_NM_ENC'] == '1':
        travel_info['MVMN_자가용'] = 1
    elif travel_info['MVMN_NM_ENC'] == '2':
        travel_info['MVMN_대중교통'] = 1
    else:
        travel_info['MVMN_기타'] = 1
    
    # 동행자
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
    
    # 비용
    travel_info['TOTAL_COST_BINNED_ENCODED'] = int(travel_info['TOTAL_COST'])
    
    # 최종 벡터 생성
    travel_vector = [int(travel_info.get(k, 0)) for k in travel_feature_cols]
    
    return np.array([travel_vector]).astype(np.float32)

def simulate_user_feedback():
    """사용자 피드백 시뮬레이션"""
    feedback_options = [
        {"liked": [], "disliked": [0, 2]},  # 첫 번째와 세 번째 장소 싫어요
        {"liked": [1], "disliked": [4, 7]},  # 두 번째 장소 좋아요, 다른 곳들 싫어요
        {"liked": [0, 3], "disliked": [5]},  # 복수 좋아요/싫어요
    ]
    
    return random.choice(feedback_options)

def main_feedback_test(travel_example) -> dict:
    print("🚀 개선된 피드백 기반 경로 대체 테스트 시작!")
    print("=" * 60)

    travel_tensor = process_travel_input(travel_example)
    travel_duration = int(travel_tensor[0, 3])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 사용 디바이스: {device}")
    print(f"📅 여행 기간: {travel_example['date_range']} ({travel_tensor[0, 3]:.0f}일)")
    
    travel_context_tensor = torch.tensor(travel_tensor, dtype=torch.float32).to(device)
    
    recommender = SmartRecommendationEngine(device)
    
    # 초기 추천 (필터링 적용, 거리 고려)
    recommendations, embeddings, _ = recommender.get_recommendations(
        travel_context_tensor, top_k=30, diversity_weight=0.3, 
        filter_useless=True, consider_distance=True
    )
    
    optimized_routes, unique_recommendations = recommender.optimize_routes(recommendations, travel_tensor)
    
    print("\n🗓️ 초기 여행 일정 (최적화 및 필터링 적용):")
    total_places = 0
    for day, route in sorted(optimized_routes.items()):
        print(f"\n📅 Day {day + 1}:")
        for i, loc in enumerate(route):
            print(f" {i+1}. [{loc['id']:3d}] {loc['name']}")
        total_places += len(route)
    print(f"\n총 {total_places}개 장소 추천")
    
    return optimized_routes
    
    # 피드백 라운드 반복
    for round_num in range(1):
        print(f"\n{'='*60}")
        print(f"🔄 피드백 라운드 {round_num + 1}")
        
        feedback = simulate_user_feedback()
        optimized_routes = recommender.feedback_model(feedback, travel_context_tensor, travel_duration, unique_recommendations, embeddings)
        
        print("\n🎯 피드백 반영 후 최적화된 여행 일정:")
        total_places = 0
        for day, route in sorted(optimized_routes.items()):
            print(f"\n📅 Day {day + 1}:")
            for i, loc in enumerate(route):
                print(f" {i+1}. [{loc['id']:3d}] {loc['name']}")
            total_places += len(route)
        print(f"\n총 {total_places}개 장소 추천")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    # 여행 정보 (2일 여행)
    travel_example = {
        'mission_ENC': '0,1,2',
        'date_range': '2025-09-28 - 2025-09-29',  # 2일 여행으로 변경
        'start_date': '',
        'end_date': '',
        'TOTAL_COST': '2',
        'MVMN_NM_ENC': '2',
        'whowith_ENC': '2',
        'mission_type': 'normal'
    }
    route = main_feedback_test(travel_example)
    
    print("=" * 100)
    print(route)
    print("=" * 100)