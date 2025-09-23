from sqlalchemy import Column, Integer, String, Text, TIMESTAMP, Enum
from sqlalchemy.sql import func
from app.db.database import Base
import enum

class SourceFieldEnum(enum.Enum):
    CRAWLING = "CRAWLING"
    MANUAL = "MANUAL"

class Book(Base):
    __tablename__ = "books"
    __table_args__ = {"mysql_charset": "utf8mb4"}
    
    book_id = Column(Integer, primary_key=True, index=True)
    book_name = Column(String(200))
    book_image = Column(String(300))
    author = Column(String(100))
    publisher = Column(String(100))
    summary = Column(Text)
    isbn = Column(String(20))
    keyword = Column(Text)
    review = Column(Text)
    source_field = Column(Enum(SourceFieldEnum), default=SourceFieldEnum.CRAWLING)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())