"""
Database module: support SQLite and any SQL DB via SQLAlchemy.
"""
from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from config import DB_URL, SQLITE_PATH

Base = declarative_base()

class OcrResult(Base):
    __tablename__ = 'ocr_results'
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(Float, nullable=False)
    ocr_text = Column(String)
    ocr_confidence = Column(Float)

# engine and session
def get_engine():
    if DB_URL:
        return create_engine(DB_URL)
    else:
        # ensure directory exists
        os.makedirs(os.path.dirname(SQLITE_PATH), exist_ok=True)
        return create_engine(f"sqlite:///{SQLITE_PATH}")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())

def init_db():
    engine = get_engine()
    Base.metadata.create_all(bind=engine)

def save_results(results_list):
    """Bulk insert list of dicts with keys matching OcrResult columns."""
    db = SessionLocal()
    try:
        objs = [
            OcrResult(
                timestamp=r.get('timestamp'),
                ocr_text=r.get('ocr_text'),
                ocr_confidence=r.get('ocr_confidence')
            )
            for r in results_list
        ]
        db.bulk_save_objects(objs)
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
