import requests
import xml.etree.ElementTree as ET
import json
import os
from dotenv import load_dotenv

load_dotenv()
OC = os.getenv("LAW_API_KEY")
BASE_URL = "https://www.law.go.kr/DRF"

def get_law_list(query: str) -> list:
    """법령 목록 검색 (JSON)"""
    url = f"{BASE_URL}/lawSearch.do"
    params = {
        "OC": OC,
        "target": "law",
        "type": "JSON",
        "query": query,
        "display": 10
    }
    res = requests.get(url, params=params)
    data = res.json()
    
    laws = data.get("LawSearch", {}).get("law", [])
    if isinstance(laws, dict):
        laws = [laws]
    
    return [{"법령명": l["법령명한글"], "법령ID": l["법령ID"]} for l in laws]


def get_law_text(law_id: str) -> list:
    url = f"{BASE_URL}/lawService.do"
    params = {
        "OC": OC,
        "target": "law",
        "ID": law_id,
        "type": "XML"
    }
    res = requests.get(url, params=params)
    root = ET.fromstring(res.content)
    
    articles = []
    for article in root.iter("조문단위"):
        num = article.findtext("조문번호", "")
        title = article.findtext("조문제목", "")
        
        parts = []
        
        # 1. 조문내용 직접 있는 경우
        content = article.findtext("조문내용", "").strip()
        if content:
            parts.append(content)
        
        # 2. 항 구조 합치기 (있으면 추가)
        for hang in article.iter("항"):
            hang_num = hang.findtext("항번호", "").strip()
            hang_content = hang.findtext("항내용", "").strip()
            if hang_content:
                parts.append(f"{hang_num} {hang_content}".strip())
            
            # 3. 호 구조까지 합치기
            for ho in hang.iter("호"):
                ho_content = ho.findtext("호내용", "").strip()
                if ho_content:
                    parts.append(ho_content)
        
        full_content = " ".join(parts).strip()
        
        if full_content and len(full_content) >= 30:
            articles.append({
                "조문번호": num,
                "조문제목": title,
                "조문내용": full_content
            })
    
    seen = set()
    deduped = []
    for article in articles:
        num = article["조문번호"]
        if num not in seen:
            seen.add(num)
            deduped.append(article)
    
    return deduped


def save_law(law_name: str):
    """법령 검색 → 본문 수집 → JSON 저장"""
    print(f"[1] {law_name} 검색 중...")
    laws = get_law_list(law_name)
    
    if not laws:
        print("검색 결과 없음")
        return
    
    # 첫 번째 결과 사용
    target = laws[0]
    print(f"[2] {target['법령명']} (ID: {target['법령ID']}) 본문 수집 중...")
    
    articles = get_law_text(target["법령ID"])
    print(f"[3] 조문 {len(articles)}개 수집 완료")
    
    # 저장
    save_path = f"data/raw/{law_name}.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    
    print(f"[4] 저장 완료 → {save_path}")
    return articles


if __name__ == "__main__":
    # API 승인 후 테스트
    save_law("근로기준법")
    save_law("남녀고용평등법")
    save_law("고용보험법")