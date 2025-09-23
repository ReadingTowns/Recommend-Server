from fastapi import APIRouter, Query, HTTPException
from app.recommend import model

router = APIRouter()

@router.get("/hc")
def healthyCheck():
    return {
        "message" : "test OK"
    }

@router.get("/recommend")
def recommend(book_ids: str = Query(..., description="추천 기준 책 ID, 콤마로 구분")):

    try:
        # 1. Query 파라미터 문자열 → 정수 리스트
        ids = [int(x) for x in book_ids.split(",")]

        # 2. 추천 결과 호출
        recommended_books_with_scores = model.recommend_books_by_user_books(ids)

        # 3. 결과 JSON으로 반환
        result = [
            {
                "book_id": b.book_id,
                "book_name": b.book_name,
                "author": b.author,
                "publisher": b.publisher,
                "keyword": b.keyword,
                "similarity": float(score),
                "review keywords": keywords,
            }
            for b, score, keywords in recommended_books_with_scores
        ]
        return {"recommendations": result}

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))