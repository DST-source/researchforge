from sqlalchemy import Column, Integer, String, Text, ForeignKey
from .base import Base

class Section(Base):
    __tablename__ = "sections"
    
    id = Column(Integer, primary_key=True)
    paper_id = Column(Integer, ForeignKey("papers.id", ondelete="CASCADE"))
    section_type = Column(String(50))  # abstract, intro, methods, etc.
    title = Column(String(500))
    start_page = Column(Integer)
    end_page = Column(Integer)
    text_content = Column(Text)