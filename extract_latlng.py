import os
import requests
from dotenv import load_dotenv
load_dotenv(override = True)

api_key = os.getenv("KAKAO_REST_API")


def get_lat_lon_by_name(name):
    """
    장소 이름으로 위도/경도 찾기 (카카오맵 keyword API 사용)
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
            lon = float(documents[0]['x'])
            lat = float(documents[0]['y'])
            return lat, lon
        else:
            print("이름 검색 결과를 찾을 수 없어요.")
            return None, None
    else:
        print(f"API 오류 {response.status_code}: {response.text}")
        return None, None


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

def fill_missing_coords_with_kakao(travel_plan_list):
    for day_plan in travel_plan_list:
        for loc in day_plan['route']:
            if loc['x'] is None and loc['y'] is None:
                # 1️⃣ 주소 기반 검색
                if loc['addr']:
                    lat, lon = get_lat_lon_kakao(loc['addr'])
                else:
                    lat, lon = None, None

                # 2️⃣ 주소가 없거나 검색 실패 → 이름 기반 검색
                if (lat is None or lon is None) and loc['name']:
                    lat, lon = get_lat_lon_by_name(loc['name'])
                    if lat and lon:
                        print(f"[DEBUG] 이름으로 검색 성공: {loc['name']} → ({lat}, {lon})")
                    else: # 이름에서 띄어쓰기 제거하고 확인
                        name = loc['name'].replace(' ', '')
                        lat, lon = get_lat_lon_by_name(name)

                        if lat and lon:
                            print(f"[DEBUG] 이름으로 검색 성공: {loc['name']} → ({lat}, {lon})")

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

    lat, lon =  get_lat_lon_by_name("서울숲")
    if lat and lon:
        print(f"위도: {lat}, 경도: {lon}")