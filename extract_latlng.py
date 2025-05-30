import os
import requests
from dotenv import load_dotenv
load_dotenv(override = True)

api_key = os.getenv("KAKAO_REST_API")


def get_lat_lon_by_name(name):
    """
    장소 이름으로 위도/경도와 주소 찾기 (카카오맵 keyword API 사용)
    반환: (lat, lon, address, road_address)
    """
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    headers = {
        "Authorization": f"KakaoAK {api_key}"
    }
    params = {
        "query": name
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        result = response.json()
        documents = result.get('documents')
        if documents:
            doc = documents[0]
            lon = float(doc['x'])
            lat = float(doc['y'])
            # 주소 정보 추출
            address = doc.get('address_name', '')  # 지번 주소
            road_address = doc.get('road_address_name', '')  # 도로명 주소
            return lat, lon, address, road_address
        else:
            print("이름 검색 결과를 찾을 수 없어요.")
            return None, None, None, None
    else:
        print(f"API 오류 {response.status_code}: {response.text}")
        return None, None, None, None


def get_lat_lon_kakao(address):
    """
    주소(도로명/지번)를 입력받아 위도(lat), 경도(lon) 반환
    """
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
    
def get_address_by_coords(lat, lon):
    """
    위도/경도로 주소 찾기 (카카오맵 좌표계변환 API 사용)
    반환: (address, road_address)
    """
    url = "https://dapi.kakao.com/v2/local/geo/coord2address.json"
    headers = {
        "Authorization": f"KakaoAK {api_key}"
    }
    params = {
        "x": lon,  # 경도
        "y": lat   # 위도
    }
    
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        result = response.json()
        documents = result.get('documents')
        if documents:
            doc = documents[0]
            # 지번 주소
            address = doc.get('address', {}).get('address_name', '') if doc.get('address') else ''
            # 도로명 주소
            road_address = doc.get('road_address', {}).get('address_name', '') if doc.get('road_address') else ''
            return address, road_address
        else:
            print("좌표로 주소를 찾을 수 없어요.")
            return None, None
    else:
        print(f"좌표→주소 API 오류 {response.status_code}: {response.text}")
        return None, None

def fill_missing_coords_with_kakao(travel_plan_list):
    for day_plan in travel_plan_list:
        for loc in day_plan['route']:
            # 🔥 좌표는 있지만 주소가 없는 경우 처리 추가
            if loc['x'] is not None and loc['y'] is not None and not loc.get('addr'):
                lat = float(loc['y'])
                lon = float(loc['x'])
                address, road_address = get_address_by_coords(lat, lon)
                
                if road_address:
                    loc['addr'] = road_address
                    print(f"[DEBUG] 좌표로 도로명 주소 업데이트: {loc['name']} → {road_address}")
                elif address:
                    loc['addr'] = address
                    print(f"[DEBUG] 좌표로 지번 주소 업데이트: {loc['name']} → {address}")
                else:
                    print(f"[DEBUG] ⚠️ 좌표로 주소 찾기 실패: {loc['name']} ({lat}, {lon})")
            
            # 기존 로직: 좌표가 없는 경우 처리
            elif loc['x'] is None and loc['y'] is None:
                # 1️⃣ 주소 기반 검색
                if loc['addr']:
                    lat, lon = get_lat_lon_kakao(loc['addr'])
                else:
                    lat, lon = None, None

                # 2️⃣ 주소가 없거나 검색 실패 → 이름 기반 검색 (주소도 함께 업데이트)
                if (lat is None or lon is None) and loc['name']:
                    lat, lon, address, road_address = get_lat_lon_by_name(loc['name'])
                    if lat and lon:
                        print(f"[DEBUG] 이름으로 검색 성공: {loc['name']} → ({lat}, {lon})")
                        # 주소 정보 업데이트 (도로명 주소 우선, 없으면 지번 주소)
                        if road_address:
                            loc['addr'] = road_address
                            print(f"[DEBUG] 도로명 주소 업데이트: {road_address}")
                        elif address:
                            loc['addr'] = address
                            print(f"[DEBUG] 지번 주소 업데이트: {address}")
                    else: 
                        # 이름에서 띄어쓰기 제거하고 재시도
                        name = loc['name'].replace(' ', '')
                        lat, lon, address, road_address = get_lat_lon_by_name(name)

                        if lat and lon:
                            print(f"[DEBUG] 이름으로 검색 성공 (띄어쓰기 제거): {name} → ({lat}, {lon})")
                            # 주소 정보 업데이트
                            if road_address:
                                loc['addr'] = road_address
                                print(f"[DEBUG] 도로명 주소 업데이트: {road_address}")
                            elif address:
                                loc['addr'] = address
                                print(f"[DEBUG] 지번 주소 업데이트: {address}")

                # 최종 좌표 채움
                if lat is not None and lon is not None:
                    loc['x'] = lon
                    loc['y'] = lat
                    print(f"[DEBUG] 최종 좌표 채움: {loc['name']} → ({loc['x']}, {loc['y']})")
                else:
                    print(f"[DEBUG] ⚠️ 좌표 찾기 실패: {loc['name']}")
                    
    return travel_plan_list



# 예제 사용
if __name__ == "__main__":
    test_address = "서울특별시 중구 세종대로 110"
    lat, lon = get_lat_lon_kakao(test_address)
    if lat and lon:
        print(f"주소: {test_address}")
        print(f"위도: {lat}, 경도: {lon}")

    lat, lon, address, road_address = get_lat_lon_by_name("서울숲")
    if lat and lon:
        print(f"이름: 서울숲")
        print(f"위도: {lat}, 경도: {lon}")
        print(f"지번 주소: {address}")
        print(f"도로명 주소: {road_address}")