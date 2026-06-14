"""AI Agent 核心 - 多模型支持"""

import os
from openai import OpenAI

# 供应商配置
PROVIDERS = {
    "deepseek": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "env_key": "DEEPSEEK_API_KEY",
        "models": ["deepseek-chat", "deepseek-reasoner"]
    },
    "mimo": {
        "name": "MiMo",
        "base_url": "https://api.xiaomimimo.com/v1",
        "env_key": "MIMO_API_KEY",
        "models": ["mimo-v2-flash", "mimo-v2-pro", "mimo-v2-omni", "mimo-v2.5", "mimo-v2.5-pro"],
        "key_hint": "sk-..."
    },
    "mimo_openrouter": {
        "name": "MiMo (OpenRouter)",
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
        "models": ["xiaomi/mimo-v2.5", "xiaomi/mimo-v2.5-pro"],
        "key_hint": "sk-or-v1-..."
    }
}


def get_client(provider: str, api_key: str = None):
    """获取对应供应商的客户端"""
    cfg = PROVIDERS.get(provider)
    if not cfg:
        return None

    key = api_key or os.getenv(cfg["env_key"]) or cfg.get("default_key")
    if not key:
        return None

    return OpenAI(api_key=key, base_url=cfg["base_url"])


def chat(provider: str, api_key: str, system_prompt: str, messages: list, model: str):
    """调用模型进行流式对话"""
    client = get_client(provider, api_key)
    if not client:
        yield f"请先设置 {PROVIDERS[provider]['name']} API Key"
        return

    msgs = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        msgs.append({"role": msg["role"], "content": msg["content"]})

    try:
        response = client.chat.completions.create(
            model=model,
            messages=msgs,
            max_tokens=4096,
            stream=True
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"调用出错: {str(e)}"
