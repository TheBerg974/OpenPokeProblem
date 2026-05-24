"""
database.py — SQLAlchemy (asyncpg) models, Pydantic schemas, and async CRUD
for the Meta-Registry and Thread State persistence layer.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from pydantic import BaseModel
from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, relationship, selectinload
from sqlalchemy.future import select

# ---------------------------------------------------------------------------
# Engine & session factory
# ---------------------------------------------------------------------------

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://admin:password@localhost:5432/open_poke",
)

engine = create_async_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


# ---------------------------------------------------------------------------
# ORM models
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    __allow_unmapped__ = True


class ThreadMeta(Base):
    """High-level directory entry for a conversation thread."""

    __tablename__ = "thread_meta"

    thread_id: str = Column(String, primary_key=True, index=True)
    title: str = Column(String, nullable=False)
    semantic_summary: str = Column(Text, nullable=True)
    updated_at: datetime = Column(
        DateTime(timezone=True),
        default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    history: list["ThreadHistory"] = relationship(
        "ThreadHistory",
        back_populates="thread",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class ThreadHistory(Base):
    """Individual message turn within a thread."""

    __tablename__ = "thread_history"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    thread_id: str = Column(
        String, ForeignKey("thread_meta.thread_id", ondelete="CASCADE"), nullable=False
    )
    role: str = Column(String, nullable=False)      # "user" | "assistant" | "tool"
    content: str = Column(Text, nullable=False)
    created_at: datetime = Column(
        DateTime(timezone=True), default=func.now(), nullable=False
    )

    thread: ThreadMeta = relationship("ThreadMeta", back_populates="history")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class ThreadMetaSchema(BaseModel):
    thread_id: str
    title: str
    semantic_summary: Optional[str] = None
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class ThreadHistorySchema(BaseModel):
    id: Optional[int] = None
    thread_id: str
    role: str
    content: str
    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Database initialisation
# ---------------------------------------------------------------------------


async def init_db() -> None:
    """Create all tables if they do not already exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------


async def upsert_thread_meta(
    thread_id: str,
    title: str,
    summary: Optional[str] = None,
    session: Optional[AsyncSession] = None,
) -> ThreadMetaSchema:
    """Insert or update a ThreadMeta row and return the validated schema."""

    async def _run(db: AsyncSession) -> ThreadMetaSchema:
        result = await db.execute(
            select(ThreadMeta).where(ThreadMeta.thread_id == thread_id)
        )
        existing = result.scalar_one_or_none()

        if existing:
            existing.title = title
            existing.semantic_summary = summary
            existing.updated_at = datetime.utcnow()
            row = existing
        else:
            row = ThreadMeta(
                thread_id=thread_id,
                title=title,
                semantic_summary=summary,
                updated_at=datetime.utcnow(),
            )
            db.add(row)

        await db.commit()
        await db.refresh(row)
        return ThreadMetaSchema.model_validate(row)

    if session:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)


async def fetch_thread_history(
    thread_id: str,
    session: Optional[AsyncSession] = None,
) -> list[ThreadHistorySchema]:
    """Return all history rows for a thread, ordered by creation time."""

    async def _run(db: AsyncSession) -> list[ThreadHistorySchema]:
        result = await db.execute(
            select(ThreadHistory)
            .where(ThreadHistory.thread_id == thread_id)
            .order_by(ThreadHistory.created_at.asc())
        )
        rows = result.scalars().all()
        return [ThreadHistorySchema.model_validate(r) for r in rows]

    if session:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)


async def append_thread_history(
    thread_id: str,
    role: str,
    content: str,
    session: Optional[AsyncSession] = None,
) -> ThreadHistorySchema:
    """Append a new message turn to the thread history."""

    async def _run(db: AsyncSession) -> ThreadHistorySchema:
        row = ThreadHistory(
            thread_id=thread_id,
            role=role,
            content=content,
            created_at=datetime.utcnow(),
        )
        db.add(row)
        await db.commit()
        await db.refresh(row)
        return ThreadHistorySchema.model_validate(row)

    if session:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)


async def fetch_all_thread_metas(
    session: Optional[AsyncSession] = None,
) -> list[ThreadMetaSchema]:
    """Return all ThreadMeta rows — used to warm the Redis meta-registry."""

    async def _run(db: AsyncSession) -> list[ThreadMetaSchema]:
        result = await db.execute(
            select(ThreadMeta).order_by(ThreadMeta.updated_at.desc())
        )
        rows = result.scalars().all()
        return [ThreadMetaSchema.model_validate(r) for r in rows]

    if session:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)
