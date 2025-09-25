from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from app.recommend import model
from app.db import fetch_books, fetch_book_reviews
from app.recommend.bert_recommender import BertRecommender

router = APIRouter()
bert_rec = BertRecommender()

# Pydantic 모델 정의
class TextSearchRequest(BaseModel):
    query: str
    top_k: int = 10
    use_combined: bool = True

# BERT 추천기 초기화를 위한 startup 함수
def initialize_bert():
    try:
        records = fetch_books()
        bert_rec.build_keyword_embeddings(records)
        
        review_records = fetch_book_reviews()
        if review_records:
            bert_rec.build_combined_embeddings(records, review_records)
            print(f"BERT embeddings built: {len(records)} books with keywords, {len(review_records)} with reviews")
        else:
            print(f"BERT embeddings built: {len(records)} books (keywords only, no reviews found)")
    except Exception as e:
        print(f"Warning: BERT recommender initialization failed: {e}")
        print("Continuing without BERT recommender...")

# 앱 시작 시 초기화 실행
initialize_bert()

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

# 키워드만 사용한 BERT 추천 (단일 책)
@router.get("/recommend/bert-keywords")
def recommend_bert_keywords(book_id: int = Query(...), top_k: int = Query(10, ge=1, le=50)):
    """BERT 임베딩을 사용한 키워드 기반 추천"""
    if bert_rec.keyword_embeddings is None:
        raise HTTPException(
            status_code=503,
            detail="BERT keyword embeddings not initialized"
        )
    
    if book_id not in bert_rec.book_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Book with id {book_id} not found"
        )
    
    items = bert_rec.recommend_by_keywords(book_id, top_k)
    selected_book = {
        "book_id": book_id,
        "book_name": bert_rec.book_names.get(book_id, ""),
        "keywords": bert_rec.book_keywords.get(book_id, "")
    }
    
    return {
        "selected_book": selected_book,
        "recommendations": items,
        "method": "bert_keywords"
    }

# 여러 책 기반 BERT 키워드 추천
@router.get("/recommend/bert-keywords-multi")
def recommend_bert_keywords_multi(book_ids: str = Query(..., description="추천 기준 책 ID, 콤마로 구분"), top_k: int = Query(10, ge=1, le=50)):
    """여러 책을 기반으로 한 BERT 키워드 추천"""
    if bert_rec.keyword_embeddings is None:
        raise HTTPException(
            status_code=503,
            detail="BERT keyword embeddings not initialized"
        )
    
    try:
        ids = [int(x) for x in book_ids.split(",")]
        
        # 유효하지 않은 ID 체크
        invalid_ids = [id for id in ids if id not in bert_rec.book_ids]
        if invalid_ids:
            raise HTTPException(
                status_code=404,
                detail=f"Books with ids {invalid_ids} not found"
            )
        
        items = bert_rec.recommend_by_multiple_books_keywords(ids, top_k)
        selected_books = [{
            "book_id": bid,
            "book_name": bert_rec.book_names.get(bid, ""),
            "keywords": bert_rec.book_keywords.get(bid, "")
        } for bid in ids]
        
        return {
            "selected_books": selected_books,
            "recommendations": items,
            "method": "bert_keywords_multi"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# 키워드 + 리뷰 결합 BERT 추천 (단일 책)
@router.get("/recommend/bert-combined")
def recommend_bert_combined(book_id: int = Query(...), top_k: int = Query(10, ge=1, le=50)):
    """BERT 임베딩을 사용한 키워드+리뷰 결합 추천"""
    if bert_rec.combined_embeddings is None:
        raise HTTPException(
            status_code=503,
            detail="BERT combined embeddings not initialized (no reviews found)"
        )
    
    if book_id not in bert_rec.book_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Book with id {book_id} not found"
        )
    
    items = bert_rec.recommend_combined(book_id, top_k)
    selected_book = {
        "book_id": book_id,
        "book_name": bert_rec.book_names.get(book_id, ""),
        "keywords": bert_rec.book_keywords.get(book_id, ""),
        "has_review": bool(bert_rec.book_reviews.get(book_id, ""))
    }
    
    return {
        "selected_book": selected_book,
        "recommendations": items,
        "method": "bert_combined"
    }

# 여러 책 기반 BERT 키워드+리뷰 결합 추천
@router.get("/recommend/bert-combined-multi")
def recommend_bert_combined_multi(book_ids: str = Query(..., description="추천 기준 책 ID, 콤마로 구분"), top_k: int = Query(10, ge=1, le=50)):
    """여러 책을 기반으로 한 BERT 키워드+리뷰 결합 추천"""
    if bert_rec.combined_embeddings is None:
        raise HTTPException(
            status_code=503,
            detail="BERT combined embeddings not initialized (no reviews found)"
        )
    
    try:
        ids = [int(x) for x in book_ids.split(",")]
        
        # 유효하지 않은 ID 체크
        invalid_ids = [id for id in ids if id not in bert_rec.book_ids]
        if invalid_ids:
            raise HTTPException(
                status_code=404,
                detail=f"Books with ids {invalid_ids} not found"
            )
        
        items = bert_rec.recommend_by_multiple_books_combined(ids, top_k)
        selected_books = [{
            "book_id": bid,
            "book_name": bert_rec.book_names.get(bid, ""),
            "keywords": bert_rec.book_keywords.get(bid, ""),
            "has_review": bool(bert_rec.book_reviews.get(bid, ""))
        } for bid in ids]
        
        return {
            "selected_books": selected_books,
            "recommendations": items,
            "method": "bert_combined_multi"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# 텍스트 검색
@router.post("/search/bert")
def search_bert(request: TextSearchRequest):
    """텍스트 쿼리로 BERT 검색"""
    if request.use_combined and bert_rec.combined_embeddings is None:
        # combined가 없으면 keyword만 사용
        request.use_combined = False
    
    if not request.use_combined and bert_rec.keyword_embeddings is None:
        raise HTTPException(
            status_code=503,
            detail="BERT embeddings not initialized"
        )
    
    items = bert_rec.search_by_text(request.query, request.top_k, request.use_combined)
    
    return {
        "query": request.query,
        "results": items,
        "method": "bert_combined" if request.use_combined else "bert_keywords"
    }