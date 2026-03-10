from datetime import datetime
from enum import StrEnum
from typing import List, Optional

from pydantic import BaseModel, Field


class Status(StrEnum):
    UPLOADING = "UPLOADING"
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    FINISHED = "FINISHED"
    ERROR = "ERROR"


class WMRemoveResults(BaseModel):
    percentage: int
    status: Status
    download_url: str | None = None


class QueueSummary(BaseModel):
    is_busy: bool = Field(..., description="Whether a task is currently being processed")
    queue_length: int = Field(..., description="Number of tasks waiting in the queue")
    total_active: int = Field(..., description="Total active tasks (processing + queued)")


class QueueTaskInfo(BaseModel):
    id: str
    status: str
    percentage: int
    video_path: str
    created_at: Optional[datetime] = Field(None, description="Task creation time")


class QueueStatusResponse(BaseModel):
    summary: QueueSummary
    current_task_id: Optional[str] = Field(None, description="Currently running task ID")
    waiting_queue: List[QueueTaskInfo] = Field(
        default_factory=list, description="List of queued tasks"
    )
