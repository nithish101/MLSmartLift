"""SQLAlchemy engine, session factory and ORM tables (PostgreSQL)."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, create_engine, func
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker


class Base(DeclarativeBase):
    pass


class Workout(Base):
    __tablename__ = "workouts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    exercise: Mapped[str] = mapped_column(String(32), nullable=False)
    bodyweight: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    sets: Mapped[list[WorkoutSet]] = relationship(
        back_populates="workout",
        cascade="all, delete-orphan",
        order_by="WorkoutSet.set_number",
        lazy="selectin",
    )

    __table_args__ = (Index("ix_workouts_exercise_created_at", "exercise", "created_at"),)


class WorkoutSet(Base):
    __tablename__ = "sets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    workout_id: Mapped[int] = mapped_column(
        ForeignKey("workouts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    set_number: Mapped[int] = mapped_column(Integer, nullable=False)
    weight: Mapped[float] = mapped_column(Float, nullable=False)
    reps: Mapped[int] = mapped_column(Integer, nullable=False)
    rir: Mapped[int] = mapped_column(Integer, nullable=False)

    workout: Mapped[Workout] = relationship(back_populates="sets")


class Prediction(Base):
    """Predicted vs. actual reps for the top set of each logged workout."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    workout_id: Mapped[int] = mapped_column(
        ForeignKey("workouts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    exercise: Mapped[str] = mapped_column(String(32), nullable=False)
    predicted_reps: Mapped[int] = mapped_column(Integer, nullable=False)
    actual_reps: Mapped[int] = mapped_column(Integer, nullable=False)
    weight: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (Index("ix_predictions_exercise_created_at", "exercise", "created_at"),)


def make_engine(url: str, pool_size: int = 10, max_overflow: int = 20) -> Engine:
    return create_engine(url, pool_pre_ping=True, pool_size=pool_size, max_overflow=max_overflow)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


def init_db(engine: Engine) -> None:
    """Create any missing tables."""
    Base.metadata.create_all(engine)
