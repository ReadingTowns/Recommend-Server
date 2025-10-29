# DB 관련 import와 설정 제거 - 더 이상 필요 없음

def build_recommendation_response(recommended_books_with_scores):
    # 결과 리스트 생성 (이미 추천 결과에 image 포함됨)
    result = [
        {
            "bookId": b['book_id'],
            "bookImage": b.get('image', ''),  # 이미 추천 결과에 포함된 image 사용
            "bookName": b['book_name'],
            "author": b.get('author', ''),
            "publisher": b.get('publisher', ''),
            "keyword": b['keyword'],
            "similarity": float(score),
            "review_keywords": review_kw,
            "relatedUserKeywords": b.get('related_user_keywords', [])
        }
        for b, score, review_kw in recommended_books_with_scores
    ]

    return {"recommendations": result}