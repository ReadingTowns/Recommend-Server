from kiwipiepy import Kiwi
from collections import Counter

kiwi = Kiwi()

STOPWORDS = {
    "책", "글", "이야기", "작가", "저자", "작품", "내용", "넷플릭스",
    "독자", "문장", "부분", "느낌", "생각", "표현", "한국", "구매", "때"
    }

def extract_top_nouns(text: str) -> str:
    if not text:
        return ""
    
    top_k = 10
    
    # 명사만 추출
    result = kiwi.analyze(text)
    nouns = [m[0] for m in result[0][0] if m[1] == 'NNG' or m[1] == 'NNP']

    # 불용어 제거
    filtered_nouns = [word for word in nouns if word not in STOPWORDS]
    
    # 가장 많이 나온 단어 10개 선택
    top_nouns = [word for word, _ in Counter(filtered_nouns).most_common(top_k)]
    
    return " ".join(top_nouns)