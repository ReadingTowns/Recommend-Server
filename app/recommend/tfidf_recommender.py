from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.recommend.utils import extract_top_nouns
import numpy as np
from typing import List, Tuple, Dict, Optional

class TfidfRecommender:
    def __init__(self):
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.book_ids: List[int] = []
        self.book_names: Dict[int, str] = {}
        self.book_keywords: Dict[int, str] = {}
        self.book_authors: Dict[int, str] = {}
        self.book_publishers: Dict[int, str] = {}
        self.book_reviews: Dict[int, str] = {}
        self.corpus: List[str] = []
        
    def build_tfidf_matrix(self, records: List[tuple], review_records: Optional[List[tuple]] = None):
        """서버 시작 시 TF-IDF 매트릭스를 미리 계산"""
        if not records:
            print("No records to build TF-IDF matrix")
            return
            
        # 리뷰 데이터를 book_id로 매핑 (튜플 형식: (book_id, review))
        reviews_by_book = {}
        if review_records:
            for book_id, review_text in review_records:
                if book_id and review_text:
                    if book_id not in reviews_by_book:
                        reviews_by_book[book_id] = []
                    reviews_by_book[book_id].append(review_text)
        
        # 책 데이터 처리 (튜플 형식: (book_id, book_name, keywords, author, publisher))
        self.book_ids = []
        self.corpus = []
        
        for record in records:
            if len(record) == 5:
                book_id, book_name, keywords, author, publisher = record
            else:  # 이전 형식 호환성
                book_id, book_name, keywords = record
                author, publisher = '', ''
            
            if not book_id:
                continue
                
            self.book_ids.append(book_id)
            self.book_names[book_id] = book_name or ''
            self.book_keywords[book_id] = keywords or ''
            self.book_authors[book_id] = author or ''
            self.book_publishers[book_id] = publisher or ''
            
            # 키워드와 리뷰 결합
            keyword_text = keywords or ''
            review_texts = reviews_by_book.get(book_id, [])
            
            if review_texts:
                combined_review = ' '.join(review_texts)
                self.book_reviews[book_id] = combined_review
                review_nouns = extract_top_nouns(combined_review)
                text = keyword_text + " " + review_nouns
            else:
                text = keyword_text
                
            self.corpus.append(text)
        
        # TF-IDF 벡터화
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
        
        print(f"TF-IDF matrix built: {len(self.book_ids)} books, "
              f"{len(self.book_reviews)} with reviews")
    
    def recommend_by_user_books(
        self,
        book_ids: List[int],
        user_preference_keywords: List[str] = None,
        top_n: int = 10
    ) -> List[Tuple]:
        """사용자가 소유한 책들과 선호 키워드를 기반으로 추천"""
        
        if self.tfidf_matrix is None:
            raise ValueError("TF-IDF matrix not initialized")
        
        # 입력 책 ID들의 인덱스 찾기
        target_indices = []
        for book_id in book_ids:
            if book_id in self.book_ids:
                idx = self.book_ids.index(book_id)
                target_indices.append(idx)
        
        if not target_indices:
            raise ValueError(f"No books found for IDs: {book_ids}")
        
        # 추천 후보 인덱스 (입력 책 제외)
        candidate_indices = [i for i in range(len(self.book_ids)) 
                           if self.book_ids[i] not in book_ids]
        
        if not candidate_indices:
            return []
        
        # 사용자 벡터 생성
        # 1. 소유한 책들의 평균 벡터
        user_book_vecs = self.tfidf_matrix[target_indices]
        book_vec = np.asarray(user_book_vecs.mean(axis=0)).ravel()
        
        # 2. 선호 키워드 벡터
        if user_preference_keywords:
            preference_text = " ".join(user_preference_keywords)
            keyword_vec = self.vectorizer.transform([preference_text])
            keyword_vec = np.asarray(keyword_vec.toarray()).ravel()
        else:
            keyword_vec = np.zeros_like(book_vec)
        
        # 3. 최종 사용자 벡터 (가중치: 키워드 0.7, 책 0.3)
        final_user_vec = 0.3 * book_vec + 0.7 * keyword_vec
        user_vec_2d = final_user_vec.reshape(1, -1)
        
        # 후보 책들과의 코사인 유사도 계산
        candidate_matrix = self.tfidf_matrix[candidate_indices]
        sim_scores = cosine_similarity(user_vec_2d, candidate_matrix).flatten()
        
        # 상위 N개 선택
        top_indices_in_candidates = sim_scores.argsort()[::-1][:top_n]
        
        # 결과 생성
        keyword_top_n = 10
        recommended_books_with_scores = []
        
        for idx_in_candidates in top_indices_in_candidates:
            actual_idx = candidate_indices[idx_in_candidates]
            book_id = self.book_ids[actual_idx]
            score = sim_scores[idx_in_candidates]
            
            # 리뷰 키워드 추출
            review_text = self.book_reviews.get(book_id, '')
            if review_text:
                review_keywords = extract_top_nouns(review_text)
                if isinstance(review_keywords, str):
                    review_keywords = review_keywords.split()[:keyword_top_n]
                else:
                    review_keywords = review_keywords[:keyword_top_n]
            else:
                review_keywords = []
            
            # 책 정보를 딕셔너리로 반환
            book_info = {
                'book_id': book_id,
                'book_name': self.book_names.get(book_id, ''),
                'keyword': self.book_keywords.get(book_id, ''),
                'author': self.book_authors.get(book_id, ''),
                'publisher': self.book_publishers.get(book_id, '')
            }
            
            recommended_books_with_scores.append((book_info, score, review_keywords))
        
        return recommended_books_with_scores