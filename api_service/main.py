"""AI 创业教练 API 服务 - 可部署到 Railway / Render / HuggingFace Spaces"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from contextlib import asynccontextmanager
from typing import Optional
import json

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from agent import PROVIDERS, chat, get_client
from industries_data import INDUSTRIES
from schemas import (
    ChatRequest, IndustrySummary, IndustryDetail,
    ScenarioSummary, ScenarioDetail, APIInfo,
)
from sessions import get_or_create, add_message, clear as clear_session


@asynccontextmanager
async def lifespan(app: FastAPI):
    available = [k for k, v in PROVIDERS.items() if os.getenv(v["env_key"])]
    if not available:
        keys = {v["env_key"] for v in PROVIDERS.values()}
        print(f"[WARN] 未设置任何 API Key: {', '.join(keys)}")
    else:
        print(f"[INFO] 可用 Providers: {available}")
    yield


app = FastAPI(
    title="AI 创业教练 API",
    version="1.0.0",
    description="八大产业、数十个创业场景的 AI 教练 API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件（手机端 Web App）
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/chat", include_in_schema=False)
def chat_page():
    return FileResponse(os.path.join(static_dir, "index.html"))


# ── 健康检查 ──────────────────────────────────────────────


@app.get("/health")
def health():
    return {"status": "ok"}


# ── API 信息 ──────────────────────────────────────────────


@app.get("/")
def api_info():
    return APIInfo(
        providers=list(PROVIDERS.keys()),
        endpoints=[
            "GET  /",
            "GET  /health",
            "GET  /industries",
            "GET  /industries/{key}",
            "GET  /industries/{industry_key}/{scenario_key}",
            "POST /chat",
            "DELETE /sessions/{session_id}",
        ],
    )


# ── 产业 & 场景查询 ──────────────────────────────────────


@app.get("/industries", summary="列出所有产业")
def list_industries():
    items = []
    for key, val in INDUSTRIES.items():
        items.append(IndustrySummary(
            key=key,
            name=val["name"],
            icon=val.get("icon", ""),
            description=val.get("description", ""),
            scenario_count=len(val.get("scenarios", {})),
        ))
    return items


@app.get("/industries/{industry_key}", summary="获取产业详情")
def get_industry(industry_key: str):
    ind = INDUSTRIES.get(industry_key)
    if not ind:
        raise HTTPException(404, f"产业 '{industry_key}' 不存在")
    scenarios = {}
    for sk, sv in ind.get("scenarios", {}).items():
        scenarios[sk] = ScenarioSummary(
            key=sk,
            description=sv.get("description", ""),
            examples=sv.get("examples", []),
        )
    return IndustryDetail(
        key=industry_key,
        name=ind["name"],
        icon=ind.get("icon", ""),
        description=ind.get("description", ""),
        scenarios=scenarios,
    )


@app.get("/industries/{industry_key}/{scenario_key}", summary="获取场景详情")
def get_scenario(industry_key: str, scenario_key: str):
    ind = INDUSTRIES.get(industry_key)
    if not ind:
        raise HTTPException(404, f"产业 '{industry_key}' 不存在")
    sc = ind.get("scenarios", {}).get(scenario_key)
    if not sc:
        raise HTTPException(404, f"场景 '{scenario_key}' 不存在于 {industry_key}")
    return ScenarioDetail(
        key=scenario_key,
        description=sc.get("description", ""),
        prompt=sc.get("prompt", ""),
        examples=sc.get("examples", []),
        industry_key=industry_key,
        industry_name=ind["name"],
    )


# ── 对话 ──────────────────────────────────────────────────


def resolve_provider(provider: str):
    cfg = PROVIDERS.get(provider)
    if not cfg:
        raise HTTPException(400, f"不支持的 provider: {provider}，可选: {', '.join(PROVIDERS.keys())}")
    api_key = os.getenv(cfg["env_key"])
    if not api_key:
        raise HTTPException(400, f"{cfg['name']} API Key 未配置，请设置环境变量 {cfg['env_key']}")
    return cfg, api_key


def resolve_scenario(industry_key: str, scenario_key: str):
    ind = INDUSTRIES.get(industry_key)
    if not ind:
        raise HTTPException(404, f"产业 '{industry_key}' 不存在")
    sc = ind.get("scenarios", {}).get(scenario_key)
    if not sc:
        raise HTTPException(404, f"场景 '{scenario_key}' 不存在于 {industry_key}")
    return sc, ind["name"]


# 非流式响应
async def chat_sync(req: ChatRequest):
    cfg, api_key = resolve_provider(req.provider or "mimo")
    sc, _ = resolve_scenario(req.industry_key, req.scenario_key)

    model = req.model or cfg["models"][0]
    sid, history = get_or_create(req.session_id)
    add_message(sid, "user", req.message)

    msgs = [{"role": m["role"], "content": m["content"]} for m in history]

    full = ""
    for chunk in chat(req.provider or "mimo", api_key, sc["prompt"], msgs, model):
        full += chunk

    add_message(sid, "assistant", full)
    return {"session_id": sid, "content": full, "industry_key": req.industry_key, "scenario_key": req.scenario_key}


# 流式 SSE 响应
async def chat_stream(req: ChatRequest):
    cfg, api_key = resolve_provider(req.provider or "mimo")
    sc, _ = resolve_scenario(req.industry_key, req.scenario_key)

    model = req.model or cfg["models"][0]
    sid, history = get_or_create(req.session_id)
    add_message(sid, "user", req.message)

    msgs = [{"role": m["role"], "content": m["content"]} for m in history]

    async def event_generator():
        full = ""
        yield {"event": "meta", "data": json.dumps({"session_id": sid})}
        for chunk in chat(req.provider or "mimo", api_key, sc["prompt"], msgs, model):
            full += chunk
            yield {"event": "chunk", "data": json.dumps({"content": chunk})}
        add_message(sid, "assistant", full)
        yield {"event": "done", "data": json.dumps({"session_id": sid})}

    return EventSourceResponse(event_generator())


@app.post("/chat", summary="AI 对话（支持流式 SSE 和非流式 JSON）")
async def chat_endpoint(req: ChatRequest):
    if req.stream:
        return await chat_stream(req)
    return await chat_sync(req)


# ── 会话管理 ──────────────────────────────────────────────


@app.delete("/sessions/{session_id}", summary="清除会话历史")
def delete_session(session_id: str):
    if clear_session(session_id):
        return {"status": "deleted", "session_id": session_id}
    raise HTTPException(404, f"会话 '{session_id}' 不存在")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)
