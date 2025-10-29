def build_recommendation_response(recommended_books_with_scores):
    result = [
        {
            "bookId": b['book_id'],
            "bookImage": b.get('image', ''),
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