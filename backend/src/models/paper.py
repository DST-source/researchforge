from sqlalchemy import Column, Integer, String, Text, ARRAY, DateTime, Boolean
from sqlalchemy.sql import func
from .base import Base

class Paper(Base):
    __tablename__ = "papers"
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    title = Column(String(500))
    authors = Column(ARRAY(String))  # PostgreSQL array
    year = Column(Integer)
    abstract = Column(Text)
    num_pages = Column(Integer)
    file_path = Column(String(500), nullable=False)
    uploaded_at = Column(DateTime, default=func.now())
    indexed = Column(Boolean, default=False)
    indexed_at = Column(DateTime)