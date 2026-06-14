from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    industry_key: str
    scenario_key: str
    message: str
    session_id: Optional[str] = None
    provider: Optional[str] = "mimo"
    model: Optional[str] = None
    stream: Optional[bool] = True


class ChatResponse(BaseModel):
    session_id: str
    content: str
    industry_key: str
    scenario_key: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


class IndustrySummary(BaseModel):
    key: str
    name: str
    icon: str
    description: str
    scenario_count: int


class ScenarioSummary(BaseModel):
    key: str
    description: str
    examples: list[str]


class IndustryDetail(BaseModel):
    key: str
    name: str
    icon: str
    description: str
    scenarios: dict[str, ScenarioSummary]


class ScenarioDetail(BaseModel):
    key: str
    description: str
    prompt: str
    examples: list[str]
    industry_key: str
    industry_name: str


class APIInfo(BaseModel):
    name: str = "AI 创业教练 API"
    version: str = "1.0.0"
    providers: list[str]
    endpoints: list[str]
