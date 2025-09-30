import threading
from typing import List, Tuple, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import json

class BertRecommender:
    def __init__(self, model_name: str = "jhgan/ko-sbert-nli"):
        self._lock = threading.RLock()
        self.book_ids: List[int] = []
        self.book_names: Dict[int, str] = {}
        self.book_keywords: Dict[int, str] = {}
        self.book_reviews: Dict[int, str] = {}
        
        # 임베딩 저장
        self.keyword_embeddings = None
        self.review_embeddings = None
        self.combined_embeddings = None
        
        # 한국어 BERT 모델 (영어는 'all-MiniLM-L6-v2' 사용)
        self.model = SentenceTransformer(model_name)
        
        # GPU 사용 가능 시 GPU로 이동
        if torch.cuda.is_available():
            self.model = self.model.cuda()
    
    def parse_keywords(self, raw) -> str:
        """키워드 파싱 (기존 recommender와 동일)"""
        if raw is None:
            return ""
        s = str(raw).strip()
        
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        
        parts = [p.strip() for p in s.split(",")]
        tokens = []
        for p in parts:
            p = p.strip().strip("'").strip('"')
            if p:
                tokens.append(p)
        return " ".join(tokens)
    
    def parse_reviews(self, review_json) -> str:
        """JSON 형태의 리뷰 데이터 파싱"""
        if not review_json or review_json == 'NULL':
            return ""
        
        try:
            # JSON 문자열을 파싱
            if isinstance(review_json, str):
                data = json.loads(review_json)
            else:
                data = review_json
            
            # reviews 리스트에서 텍스트만 추출
            reviews = data.get('reviews', [])
            review_texts = []
            for review in reviews:
                text = review.get('text', '').strip()
                if text:
                    review_texts.append(text)
            
            return " ".join(review_texts)
        except (json.JSONDecodeError, KeyError, TypeError):
            return ""
    
    def build_keyword_embeddings(self, records: List[Tuple[int, str, str]]):
        """키워드만 사용한 임베딩 구축"""
        with self._lock:
            self.book_ids = [int(bid) for bid, _, _ in records]
            self.book_names = {int(bid): name for bid, name, _ in records}
            self.book_keywords = {int(bid): self.parse_keywords(kw) for bid, _, kw in records}
            
            if len(records) == 0:
                self.keyword_embeddings = None
                return
            
            # 키워드 텍스트 준비
            keyword_texts = []
            for bid in self.book_ids:
                keywords = self.book_keywords.get(bid, "")
                if not keywords:
                    keywords = "키워드 없음"
                keyword_texts.append(keywords)
            
            # BERT 임베딩 생성
            print(f"Creating keyword embeddings for {len(keyword_texts)} books...")
            self.keyword_embeddings = self.model.encode(
                keyword_texts,
                convert_to_tensor=False,
                show_progress_bar=True,
                batch_size=32
            )
            print("Keyword embedding creation completed.")
    
    def build_combined_embeddings(self, book_records: List[Tuple[int, str, str]], 
                                 review_records: List[Tuple[int, str]]):
        """키워드 + 리뷰 결합 임베딩 구축"""
        with self._lock:
            # 기본 정보 저장
            self.book_ids = [int(bid) for bid, _, _ in book_records]
            self.book_names = {int(bid): name for bid, name, _ in book_records}
            self.book_keywords = {int(bid): self.parse_keywords(kw) for bid, _, kw in book_records}
            
            # 리뷰 정보 저장 (review_records: [(book_id, review_json)])
            review_dict = {int(bid): self.parse_reviews(review) for bid, review in review_records}
            self.book_reviews = {bid: review_dict.get(bid, "") for bid in self.book_ids}
            
            if len(book_records) == 0:
                self.combined_embeddings = None
                return
            
            # 키워드 + 리뷰 텍스트 결합
            combined_texts = []
            for bid in self.book_ids:
                keywords = self.book_keywords.get(bid, "")
                reviews = self.book_reviews.get(bid, "")
                
                # 키워드와 리뷰 결합
                combined = ""
                if keywords:
                    combined = f"키워드: {keywords}"
                if reviews:
                    if combined:
                        combined += f" | 리뷰: {reviews}"
                    else:
                        combined = f"리뷰: {reviews}"
                
                if not combined:
                    combined = "정보 없음"
                    
                combined_texts.append(combined)
            
            # BERT 임베딩 생성
            print(f"Creating combined embeddings for {len(combined_texts)} books...")
            self.combined_embeddings = self.model.encode(
                combined_texts,
                convert_to_tensor=False,
                show_progress_bar=True,
                batch_size=32
            )
            print("Combined embedding creation completed.")
    
    def recommend_by_keywords(self, book_id: int, top_k: int = 10) -> List[Dict]:
        """키워드만 사용한 BERT 추천"""
        with self._lock:
            if self.keyword_embeddings is None or book_id not in self.book_ids:
                return []
            
            idx = self.book_ids.index(book_id)
            query_embedding = self.keyword_embeddings[idx].reshape(1, -1)
            
            # 코사인 유사도 계산
            similarities = cosine_similarity(query_embedding, self.keyword_embeddings).ravel()
            similarities[idx] = -1.0  # 자기 자신 제외
            
            # 상위 k개 선택
            top_indices = np.argsort(-similarities)[:top_k]
            
            results = []
            for i in top_indices:
                score = float(similarities[i])
                if score < 0.3:  # 유사도 임계값
                    continue
                    
                bid = int(self.book_ids[i])
                results.append({
                    "book_id": bid,
                    "book_name": self.book_names.get(bid, ""),
                    "keywords": self.book_keywords.get(bid, ""),
                    "similarity_score": round(score, 4),
                    "method": "keyword_only"
                })
            
            return results
    
    def recommend_combined(self, book_id: int, top_k: int = 10) -> List[Dict]:
        """키워드 + 리뷰 결합 BERT 추천"""
        with self._lock:
            if self.combined_embeddings is None or book_id not in self.book_ids:
                return []
            
            idx = self.book_ids.index(book_id)
            query_embedding = self.combined_embeddings[idx].reshape(1, -1)
            
            # 코사인 유사도 계산
            similarities = cosine_similarity(query_embedding, self.combined_embeddings).ravel()
            similarities[idx] = -1.0  # 자기 자신 제외
            
            # 상위 k개 선택
            top_indices = np.argsort(-similarities)[:top_k]
            
            results = []
            for i in top_indices:
                score = float(similarities[i])
                if score < 0.3:  # 유사도 임계값
                    continue
                    
                bid = int(self.book_ids[i])
                review_preview = self.book_reviews.get(bid, "")
                if review_preview and len(review_preview) > 200:
                    review_preview = review_preview[:200] + "..."
                    
                results.append({
                    "book_id": bid,
                    "book_name": self.book_names.get(bid, ""),
                    "keywords": self.book_keywords.get(bid, ""),
                    "review_preview": review_preview,
                    "similarity_score": round(score, 4),
                    "method": "keyword_and_review"
                })
            
            return results
    
    def search_by_text(self, query_text: str, top_k: int = 10, use_combined: bool = True) -> List[Dict]:
        """텍스트 쿼리로 유사한 책 검색"""
        with self._lock:
            embeddings = self.combined_embeddings if use_combined else self.keyword_embeddings
            
            if embeddings is None:
                return []
            
            # 쿼리 텍스트 임베딩
            query_embedding = self.model.encode([query_text], convert_to_tensor=False)
            
            # 유사도 계산
            similarities = cosine_similarity(query_embedding, embeddings).ravel()
            
            # 상위 k개 선택
            top_indices = np.argsort(-similarities)[:top_k]
            
            results = []
            for i in top_indices:
                score = float(similarities[i])
                if score < 0.2:  # 검색은 좀 더 낮은 임계값
                    continue
                    
                bid = int(self.book_ids[i])
                result = {
                    "book_id": bid,
                    "book_name": self.book_names.get(bid, ""),
                    "keywords": self.book_keywords.get(bid, ""),
                    "similarity_score": round(score, 4),
                    "method": "text_search"
                }
                
                if use_combined:
                    review_preview = self.book_reviews.get(bid, "")
                    if review_preview and len(review_preview) > 200:
                        review_preview = review_preview[:200] + "..."
                    result["review_preview"] = review_preview
                
                results.append(result)
            
            return results
    
    def recommend_by_multiple_books_keywords(self, book_ids: List[int], top_k: int = 10) -> List[Dict]:
        """여러 책의 키워드를 기반으로 한 BERT 추천"""
        with self._lock:
            if self.keyword_embeddings is None or not book_ids:
                return []
            
            # 여러 책의 임베딩 평균 계산
            embeddings_list = []
            for book_id in book_ids:
                if book_id in self.book_ids:
                    idx = self.book_ids.index(book_id)
                    embeddings_list.append(self.keyword_embeddings[idx])
            
            if not embeddings_list:
                return []
            
            # 평균 임베딩 계산
            avg_embedding = np.mean(embeddings_list, axis=0).reshape(1, -1)
            
            # 코사인 유사도 계산
            similarities = cosine_similarity(avg_embedding, self.keyword_embeddings).ravel()
            
            # 입력 책들은 제외
            for book_id in book_ids:
                if book_id in self.book_ids:
                    idx = self.book_ids.index(book_id)
                    similarities[idx] = -1.0
            
            # 상위 k개 선택
            top_indices = np.argsort(-similarities)[:top_k]
            
            results = []
            for i in top_indices:
                score = float(similarities[i])
                if score < 0.3:  # 유사도 임계값
                    continue
                    
                bid = int(self.book_ids[i])
                results.append({
                    "book_id": bid,
                    "book_name": self.book_names.get(bid, ""),
                    "keywords": self.book_keywords.get(bid, ""),
                    "similarity_score": round(score, 4),
                    "method": "keyword_multi"
                })
            
            return results
    
    def recommend_by_multiple_books_combined(self, book_ids: List[int], top_k: int = 10) -> List[Dict]:
        """여러 책의 키워드+리뷰를 기반으로 한 BERT 추천"""
        with self._lock:
            if self.combined_embeddings is None or not book_ids:
                return []
            
            # 여러 책의 임베딩 평균 계산
            embeddings_list = []
            for book_id in book_ids:
                if book_id in self.book_ids:
                    idx = self.book_ids.index(book_id)
                    embeddings_list.append(self.combined_embeddings[idx])
            
            if not embeddings_list:
                return []
            
            # 평균 임베딩 계산
            avg_embedding = np.mean(embeddings_list, axis=0).reshape(1, -1)
            
            # 코사인 유사도 계산
            similarities = cosine_similarity(avg_embedding, self.combined_embeddings).ravel()
            
            # 입력 책들은 제외
            for book_id in book_ids:
                if book_id in self.book_ids:
                    idx = self.book_ids.index(book_id)
                    similarities[idx] = -1.0
            
            # 상위 k개 선택
            top_indices = np.argsort(-similarities)[:top_k]
            
            results = []
            for i in top_indices:
                score = float(similarities[i])
                if score < 0.3:  # 유사도 임계값
                    continue
                    
                bid = int(self.book_ids[i])
                review_preview = self.book_reviews.get(bid, "")
                if review_preview and len(review_preview) > 200:
                    review_preview = review_preview[:200] + "..."
                    
                results.append({
                    "book_id": bid,
                    "book_name": self.book_names.get(bid, ""),
                    "keywords": self.book_keywords.get(bid, ""),
                    "review_preview": review_preview,
                    "similarity_score": round(score, 4),
                    "method": "combined_multi"
                })
            
            return results