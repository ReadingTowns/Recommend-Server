from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.db.database import SessionLocal
from app.db import models
import numpy as np
from typing import List, Tuple, Dict

def recommend_users_by_keywords(member_id: int, top_n: int = 10) -> List[Dict]:
    """
    사용자가 선택한 키워드만을 기반으로 유사한 사용자 추천
    """
    db = SessionLocal()
    try:
        # 1. 대상 사용자의 키워드 가져오기
        target_member = db.query(models.Member).filter(
            models.Member.member_id == member_id
        ).first()
        
        if not target_member:
            raise ValueError(f"Member {member_id} not found")
        
        target_keywords = [kw.content for kw in target_member.keywords]
        if not target_keywords:
            return []
        
        # 2. 다른 사용자들의 키워드 가져오기
        other_members = db.query(models.Member).filter(
            models.Member.member_id != member_id
        ).all()
        
        if not other_members:
            return []
        
        # 3. TF-IDF 벡터화
        member_keyword_texts = []
        valid_members = []
        
        for member in other_members:
            member_keywords = [kw.content for kw in member.keywords]
            if member_keywords:
                member_keyword_texts.append(" ".join(member_keywords))
                valid_members.append(member)
        
        if not valid_members:
            return []
        
        # 대상 사용자 키워드 텍스트
        target_text = " ".join(target_keywords)
        
        # TF-IDF 벡터화
        vectorizer = TfidfVectorizer()
        all_texts = member_keyword_texts + [target_text]
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # 타겟 벡터와 다른 사용자들 벡터 분리
        target_vector = tfidf_matrix[-1]
        other_vectors = tfidf_matrix[:-1]
        
        # 4. 코사인 유사도 계산
        similarities = cosine_similarity(target_vector, other_vectors).flatten()
        
        # 5. 상위 N명 선택
        top_indices = similarities.argsort()[::-1][:top_n]
        
        # 6. 결과 구성
        recommendations = []
        for idx in top_indices:
            if similarities[idx] > 0:  # 유사도가 0보다 큰 경우만
                member = valid_members[idx]
                member_keywords = [kw.content for kw in member.keywords]
                
                # 매칭된 키워드 찾기
                matched_keywords = list(set(target_keywords) & set(member_keywords))
                
                recommendations.append({
                    "memberId": int(member.member_id),
                    "similarity": float(similarities[idx]),
                    "matchedKeywords": matched_keywords,
                    "matchType": "KEYWORD_ONLY"
                })
        
        return recommendations
        
    finally:
        db.close()


def recommend_users_by_keywords_and_books(member_id: int, top_n: int = 10) -> List[Dict]:
    """
    사용자의 키워드 + 서재 책 정보를 기반으로 유사한 사용자 추천
    """
    db = SessionLocal()
    try:
        # 1. 대상 사용자의 키워드 및 서재 책 정보 가져오기
        target_member = db.query(models.Member).filter(
            models.Member.member_id == member_id
        ).first()
        
        if not target_member:
            raise ValueError(f"Member {member_id} not found")
        
        # 키워드
        target_keywords = [kw.content for kw in target_member.keywords]
        
        # 서재 책 정보
        target_bookhouses = db.query(models.Bookhouse).filter(
            models.Bookhouse.member_id == member_id
        ).all()
        
        target_books = []
        for bookhouse in target_bookhouses:
            book = db.query(models.Book).filter(
                models.Book.book_id == bookhouse.book_id
            ).first()
            if book:
                target_books.append(book)
        
        if not target_keywords and not target_books:
            return []
        
        # 2. 다른 사용자들의 정보 가져오기
        other_members = db.query(models.Member).filter(
            models.Member.member_id != member_id
        ).all()
        
        if not other_members:
            return []
        
        # 3. 각 사용자의 특징 벡터 생성
        member_feature_texts = []
        valid_members = []
        member_book_ids = {}
        member_keyword_list = {}
        
        for member in other_members:
            # 키워드
            member_keywords = [kw.content for kw in member.keywords]
            
            # 서재 책 정보
            member_bookhouses = db.query(models.Bookhouse).filter(
                models.Bookhouse.member_id == member.member_id
            ).all()
            
            member_books = []
            book_ids = []
            for bookhouse in member_bookhouses:
                book = db.query(models.Book).filter(
                    models.Book.book_id == bookhouse.book_id
                ).first()
                if book:
                    member_books.append(book)
                    book_ids.append(book.book_id)
            
            if member_keywords or member_books:
                # 키워드 텍스트
                keyword_text = " ".join(member_keywords) if member_keywords else ""
                
                # 책 관련 텍스트 (책 키워드 + 제목)
                book_texts = []
                for book in member_books:
                    if book.keyword:
                        book_texts.append(book.keyword)
                    book_texts.append(book.book_name or "")
                book_text = " ".join(book_texts)
                
                # 통합 텍스트
                combined_text = f"{keyword_text} {book_text}".strip()
                if combined_text:
                    member_feature_texts.append(combined_text)
                    valid_members.append(member)
                    member_book_ids[member.member_id] = book_ids
                    member_keyword_list[member.member_id] = member_keywords
        
        if not valid_members:
            return []
        
        # 대상 사용자의 특징 텍스트
        target_keyword_text = " ".join(target_keywords) if target_keywords else ""
        target_book_texts = []
        target_book_ids = []
        for book in target_books:
            if book.keyword:
                target_book_texts.append(book.keyword)
            target_book_texts.append(book.book_name or "")
            target_book_ids.append(book.book_id)
        target_book_text = " ".join(target_book_texts)
        target_text = f"{target_keyword_text} {target_book_text}".strip()
        
        # 4. TF-IDF 벡터화
        vectorizer = TfidfVectorizer()
        all_texts = member_feature_texts + [target_text]
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # 타겟 벡터와 다른 사용자들 벡터 분리
        target_vector = tfidf_matrix[-1]
        other_vectors = tfidf_matrix[:-1]
        
        # 5. 코사인 유사도 계산
        similarities = cosine_similarity(target_vector, other_vectors).flatten()
        
        # 6. 상위 N명 선택
        top_indices = similarities.argsort()[::-1][:top_n]
        
        # 7. 결과 구성
        recommendations = []
        for idx in top_indices:
            if similarities[idx] > 0:  # 유사도가 0보다 큰 경우만
                member = valid_members[idx]
                
                # 매칭된 키워드 찾기
                matched_keywords = list(set(target_keywords) & set(member_keyword_list[member.member_id]))
                
                # 매칭된 책 찾기
                matched_book_ids = list(set(target_book_ids) & set(member_book_ids[member.member_id]))
                matched_books = []
                for book_id in matched_book_ids:
                    book = db.query(models.Book).filter(
                        models.Book.book_id == book_id
                    ).first()
                    if book:
                        matched_books.append({
                            "bookId": book.book_id,
                            "bookName": book.book_name
                        })
                
                recommendations.append({
                    "memberId": int(member.member_id),
                    "similarity": float(similarities[idx]),
                    "matchedKeywords": matched_keywords,
                    "matchedBooks": matched_books,
                    "matchType": "KEYWORD_AND_BOOK"
                })
        
        return recommendations
        
    finally:
        db.close()