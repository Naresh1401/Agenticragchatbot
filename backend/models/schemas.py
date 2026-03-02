"""
Pydantic request/response models for the API.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Ingestion ────────────────────────────────────────────────────────────────

class IngestURLRequest(BaseModel):
    url: str = Field(..., min_length=10, max_length=2048)

    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://example.com/report.pdf",
            }
        }


class ScrapeRequest(BaseModel):
    url: str = Field(..., min_length=10, max_length=2048)

    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://example.com/about",
            }
        }


class UploadResponse(BaseModel):
    status: str
    filename: str
    source_id: str
    chunks_indexed: int
    file_type: str
    message: str


class SourceInfo(BaseModel):
    source_id: str
    filename: str
    file_type: str
    chunks: int
    uploaded_at: str


# ── Chat ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096)
    session_id: str = Field(default="default", min_length=1, max_length=64)
    stream: bool = False
    active_sources: List[str] = Field(
        default_factory=list,
        description="Filenames of recently uploaded/ingested files in this session",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "message": "What are the main findings in the uploaded report?",
                "session_id": "user-123",
            }
        }


class Citation(BaseModel):
    source: str          # filename
    chunk_id: str
    page_number: Optional[int] = None
    score: float
    snippet: str         # first 200 chars of chunk


class ToolCall(BaseModel):
    tool: str
    input: str
    output_summary: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: List[Citation] = []
    tool_calls: List[ToolCall] = []
    follow_up_suggestions: List[str] = Field(
        default_factory=list,
        description="Contextual follow-up questions the user might want to ask",
    )
    confidence: float = 1.0
    pii_detected: bool = False
    injection_detected: bool = False
    low_confidence: bool = False
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# ── Session ──────────────────────────────────────────────────────────────────

class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class ConversationMessage(BaseModel):
    role: MessageRole
    content: str
    created_at: str


class SessionHistory(BaseModel):
    session_id: str
    messages: List[ConversationMessage]
    created_at: str
    last_active: str
