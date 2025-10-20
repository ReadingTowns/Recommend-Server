# app/recommend/models.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.recommend.utils import extract_top_nouns
from app.db.database import SessionLocal
from app.db import models
import numpy as np

def recommend_books_by_user_books(
        book_ids: list[int],
        user_preference_keywords: list[str]):

    top_n = 10

    db = SessionLocal()
    try:
        # 1. 사용자의 책 가져오기
        target_books = db.query(models.Book).filter(models.Book.book_id.in_(book_ids)).all()
        if not target_books:
            raise ValueError(f"No books found for IDs: {book_ids}")

        # 2. 사용자의 책 제외한 책들 가져오기 (추천 후보 책)
        candidate_books = db.query(models.Book).filter(~models.Book.book_id.in_(book_ids)).all()
        if not candidate_books:
            return []

        # 3. 후보 책들 TF-IDF 벡터 생성
        corpus = [(b.keyword or "") + " " + extract_top_nouns(b.review) for b in candidate_books]
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)

        # 4. 사용자 취향 벡터 생성
        user_texts = [(b.keyword or "") + " " + extract_top_nouns(b.review) for b in target_books]
        user_book_vecs = vectorizer.transform(user_texts)
        book_vec = np.asarray(user_book_vecs.mean(axis=0)).ravel()

        if user_preference_keywords:
            preference_text = " ".join(user_preference_keywords)
            keyword_vec = vectorizer.transform([preference_text])
            keyword_vec = np.asarray(keyword_vec.toarray()).ravel()
        else:
            keyword_vec = np.zeros_like(book_vec)

        # 5. 코사인 유사도 계산
        # 키워드 가중치 0.7, 책 소유 가중치 0.3
        final_user_vec = (1 - 0.7) * book_vec + 0.7 * keyword_vec
        user_vec_2d = final_user_vec.reshape(1, -1)
        sim_scores = cosine_similarity(user_vec_2d, X).flatten()
        top_indices = sim_scores.argsort()[::-1][:top_n]

        # 6. 책과 유사도 + 리뷰 키워드 반환
        keyword_top_n = 10
        recommended_books_with_scores = []
        for i in top_indices:
            book = candidate_books[i]
            score = sim_scores[i]

            # 리뷰에서 키워드 뽑기 (최대 10개)
            review_keywords = extract_top_nouns(book.review)
            if isinstance(review_keywords, str):
                # 문자열이면 공백 기준 split 후 상위 10개
                review_keywords = review_keywords.split()[:keyword_top_n]
            else:
                # 리스트라면 그냥 상위 10개
                review_keywords = review_keywords[:keyword_top_n]

            recommended_books_with_scores.append((book, score, review_keywords))

        return recommended_books_with_scores
    finally:
        db.close()
