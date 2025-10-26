from sqlalchemy import Column, Integer, String, Text, TIMESTAMP, Enum, ForeignKey, Table, BigInteger
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.database import Base
import enum

class SourceFieldEnum(enum.Enum):
    CRAWLING = "CRAWLING"
    MANUAL = "MANUAL"

class KeywordTypeEnum(enum.Enum):
    MOOD = "MOOD"
    GENRE = "GENRE"
    CONTENT = "CONTENT"

member_keywords = Table(
    'member_keywords',
    Base.metadata,
    Column('member_id', BigInteger, ForeignKey('members.member_id'), primary_key=True),
    Column('keyword_id', BigInteger, ForeignKey('keywords.keyword_id'), primary_key=True)
)

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
    
    bookhouses = relationship("Bookhouse", back_populates="book")

class Member(Base):
    __tablename__ = "members"
    __table_args__ = {"mysql_charset": "utf8mb4"}
    
    member_id = Column(BigInteger, primary_key=True, index=True)
    
    keywords = relationship("Keyword", secondary=member_keywords, back_populates="members")
    bookhouses = relationship("Bookhouse", back_populates="member")

class Keyword(Base):
    __tablename__ = "keywords"
    __table_args__ = {"mysql_charset": "utf8mb4"}
    
    keyword_id = Column(BigInteger, primary_key=True, index=True)
    content = Column(String(100), nullable=False)
    type = Column(Enum(KeywordTypeEnum))
    
    members = relationship("Member", secondary=member_keywords, back_populates="keywords")

class Bookhouse(Base):
    __tablename__ = "bookhouses"
    __table_args__ = {"mysql_charset": "utf8mb4"}
    
    bookhouse_id = Column(BigInteger, primary_key=True, index=True)
    member_id = Column(BigInteger, ForeignKey('members.member_id'))
    book_id = Column(Integer, ForeignKey('books.book_id'))
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    
    member = relationship("Member", back_populates="bookhouses")
    book = relationship("Book", back_populates="bookhouses")