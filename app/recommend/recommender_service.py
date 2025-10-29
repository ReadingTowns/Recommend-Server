import os
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from app.db.models import Book


load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set. Put it in .env")

# engine과 session 정의
engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

def build_recommendation_response(recommended_books_with_scores):
    # 1. book_id 목록 추출
    book_ids = [b['book_id'] for b, _, _ in recommended_books_with_scores]

    # 2. DB에서 한 번에 조회
    with SessionLocal() as session:
        books_in_db = session.execute(
            select(Book.book_id, Book.book_image).where(Book.book_id.in_(book_ids))
        ).all()
        book_images_map = {bid: img for bid, img in books_in_db}

    # 3. 결과 리스트 생성
    result = [
        {
            "bookId": b['book_id'],
            "bookImage": book_images_map.get(b['book_id'], ""),
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