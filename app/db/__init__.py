import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set. Put it in .env")

# 예: postgresql+psycopg://user:pass@host:5432/db
#    mysql+pymysql://user:pass@host:3306/db
engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)

def fetch_books():
    """
    books 테이블에서 (book_id, book_name, keywords)를 읽어온다.
    컬럼명이 환경마다 다를 수 있어 몇 가지 후보를 시도한다.
    keywords 컬럼은 NULL 가능.
    """
    candidates = [
        "SELECT id AS book_id, book_name, keywords FROM books",
        "SELECT book_id, book_name, keywords FROM books",
        "SELECT id AS book_id, book_name, keyword AS keywords FROM books",
        "SELECT book_id, book_name, keyword AS keywords FROM books",
    ]

    with engine.begin() as conn:
        last_err = None
        for sql in candidates:
            try:
                rows = conn.execute(text(sql)).mappings().all()
                return [(int(r["book_id"]), r["book_name"], r["keywords"]) for r in rows]
            except Exception as e:
                last_err = e
                continue

    # 전부 실패하면 한 번에 터뜨린다.
    raise RuntimeError(
        "Could not query books. Adjust SQL in app/db.py to your schema."
    ) from last_err

def fetch_book_reviews():
    """
    books 테이블에서 (book_id, review_json)을 읽어온다.
    review 컬럼은 JSON 형태로 저장됨
    """
    candidates = [
        "SELECT id AS book_id, review FROM books WHERE review IS NOT NULL",
        "SELECT book_id, review FROM books WHERE review IS NOT NULL",
        "SELECT id AS book_id, reviews AS review FROM books WHERE reviews IS NOT NULL",
        "SELECT book_id, reviews AS review FROM books WHERE reviews IS NOT NULL",
    ]
    
    with engine.begin() as conn:
        last_err = None
        for sql in candidates:
            try:
                rows = conn.execute(text(sql)).mappings().all()
                return [(int(r["book_id"]), r["review"]) for r in rows]
            except Exception as e:
                last_err = e
                continue
    
    # 전부 실패하면 빈 리스트 반환 (리뷰는 선택사항)
    return []