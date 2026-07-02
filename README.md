# gemini_proxy

`gemini_proxy` 是一个轻量级协议转换网关，用于将 **Gemini API / Google GenAI SDK** 风格的请求转换为 **OpenAI Chat Completions 兼容接口**请求，并将上游 OpenAI-compatible 响应转换回 Gemini 兼容响应。

它适合部署在 Gemini 客户端与 OpenAI 兼容模型服务之间，例如：LiteLLM、自建 OpenAI-compatible 网关，或厂商提供的 `/v1/chat/completions` 兼容端点。

## 功能特性

- 暴露 Gemini 风格接口：
  - `POST /v1beta/models/{model}:generateContent`
  - `POST /v1beta/models/{model}:streamGenerateContent`
  - `POST /v1beta/models/{model}:countTokens`
  - `GET /v1beta/models`
  - `GET /v1beta/models/{model}`
- 上游统一转发到 OpenAI-compatible `/v1/chat/completions`。
- 支持非流式与流式响应转换。
- 支持函数调用 / tool calls 的 Gemini ↔ OpenAI 字段映射。
- 支持 `x-goog-api-key`、`Authorization: Bearer`、`?key=` 三种入站鉴权方式。
- 支持强制覆盖上游模型：客户端传入的 `{model}` 只用于日志，不影响实际出站模型。
- 支持上游非流式请求重试：429、5xx、连接错误、超时等可重试错误会按指数退避重试。
- 支持模型发现：从上游 `/v1/models` 拉取并转换为 Gemini `models[]` 格式。
- 支持 reasoning / thinking 适配：可将 `<think>`、`reasoning_content` 等上游推理内容转换为 Gemini `thought: true` part。

## 当前实现范围

当前项目主要实现 **Gemini 协议入口 → OpenAI-compatible 上游**。

已实现：

```text
Gemini client
   ↓
/v1beta/models/{model}:generateContent
/v1beta/models/{model}:streamGenerateContent
/v1beta/models/{model}:countTokens
/v1beta/models
   ↓
gemini_proxy
   ↓
OpenAI-compatible /v1/chat/completions 或 /v1/models
```

未实现或非当前重点：

- OpenAI 协议入站 `/v1/chat/completions`。
- Embeddings、Images、Audio 等 OpenAI / Gemini 扩展接口。
- 精确 token counting；当前 `countTokens` 是启发式估算。

## 目录结构

```text
gemini_proxy/
├── app/
│   ├── main.py                         # FastAPI 应用入口
│   ├── api/
│   │   └── gemini.py                   # Gemini 协议路由
│   ├── core/
│   │   ├── auth.py                     # 入站 API key 鉴权
│   │   ├── config.py                   # pydantic-settings 配置
│   │   └── logging.py                  # 日志配置
│   ├── services/
│   │   ├── discovery.py                # 上游模型发现
│   │   ├── stream/processor.py         # OpenAI SSE → Gemini SSE
│   │   └── transformer/
│   │       ├── to_openai.py            # Gemini 请求 → OpenAI 请求
│   │       ├── from_openai.py          # OpenAI 响应 → Gemini 响应
│   │       └── fields.py               # 字段/枚举映射
│   └── utils/
│       ├── error_handler.py            # 错误归一化
│       ├── retry.py                    # 上游重试
│       └── thinking.py                 # thinking / reasoning 适配
├── docs/                               # 需求与设计文档
├── tests/                              # 单元测试与 E2E 测试
├── .env.example                        # 环境变量模板
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── requirements-dev.txt
```

## 环境要求

- Python 3.10+
- 一个 OpenAI-compatible 上游服务，至少需要支持：
  - `POST /v1/chat/completions`
  - `GET /v1/models`（可选；失败时会回退到 `UPSTREAM_MODEL`）

## 快速开始

### 1. 安装依赖

建议使用虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果需要运行测试：

```bash
pip install -r requirements-dev.txt
```

### 2. 配置环境变量

复制示例配置：

```bash
cp .env.example .env
```

编辑 `.env`：

```env
UPSTREAM_OPENAI_URL=http://localhost:4000
UPSTREAM_API_KEY=sk-empty
UPSTREAM_MODEL=gpt-4o

PROXY_API_KEY=your-secret-proxy-key

HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
REQUEST_TIMEOUT=600
RETRY_MAX_ATTEMPTS=3
RETRY_BASE_DELAY=1.0
MODEL_DISCOVERY_CACHE_TTL=300
```

### 3. 启动服务

```bash
python -m app.main
```

默认监听：

```text
http://0.0.0.0:8000
```

健康检查：

```bash
curl http://localhost:8000/health
```

期望返回：

```json
{"status":"healthy"}
```

## Docker 使用

### Docker Compose

```bash
docker compose up -d --build
```

查看日志：

```bash
docker compose logs -f
```

停止：

```bash
docker compose down
```

### Docker

```bash
docker build -t gemini_proxy .
docker run --env-file .env -p 8000:8000 gemini_proxy
```

## API 使用示例

### 认证方式

网关支持以下三种方式之一：

```http
x-goog-api-key: your-secret-proxy-key
```

或：

```http
Authorization: Bearer your-secret-proxy-key
```

或：

```text
?key=your-secret-proxy-key
```

### 非流式 generateContent

```bash
curl -s http://localhost:8000/v1beta/models/gemini-2.5-flash:generateContent \
  -H 'Content-Type: application/json' \
  -H 'x-goog-api-key: your-secret-proxy-key' \
  -d '{
    "contents": [
      {
        "role": "user",
        "parts": [
          {"text": "你好，请用一句话介绍你自己。"}
        ]
      }
    ]
  }'
```

> 注意：URL 中的 `gemini-2.5-flash` 不会决定上游模型。实际请求上游时始终使用 `.env` 中的 `UPSTREAM_MODEL`。

### 流式 streamGenerateContent

```bash
curl -N http://localhost:8000/v1beta/models/gemini-2.5-flash:streamGenerateContent \
  -H 'Content-Type: application/json' \
  -H 'x-goog-api-key: your-secret-proxy-key' \
  -d '{
    "contents": [
      {
        "role": "user",
        "parts": [
          {"text": "写一个三行短诗。"}
        ]
      }
    ]
  }'
```

### 模型列表

```bash
curl -s http://localhost:8000/v1beta/models \
  -H 'x-goog-api-key: your-secret-proxy-key'
```

### token 估算

```bash
curl -s http://localhost:8000/v1beta/models/gemini-2.5-flash:countTokens \
  -H 'Content-Type: application/json' \
  -H 'x-goog-api-key: your-secret-proxy-key' \
  -d '{
    "contents": [
      {"role": "user", "parts": [{"text": "hello"}]}
    ]
  }'
```

## 字段转换说明

### 请求方向：Gemini → OpenAI

主要转换规则：

| Gemini | OpenAI-compatible |
|--------|-------------------|
| `contents[].role=user` | `messages[].role=user` |
| `contents[].role=model` | `messages[].role=assistant` |
| `systemInstruction` | 首条 `system` message |
| `parts[].text` | `content` 文本 |
| `parts[].inline_data` | `image_url` data URL |
| `functionCall` | `tool_calls` |
| `functionResponse` | `role=tool` message |
| `generationConfig.temperature` | `temperature` |
| `generationConfig.maxOutputTokens` | `max_tokens` |
| `generationConfig.topP` | `top_p` |
| `tools[].functionDeclarations` | OpenAI `tools[]` |

### 响应方向：OpenAI → Gemini

| OpenAI-compatible | Gemini |
|-------------------|--------|
| `message.content` | `candidates[].content.parts[].text` |
| `message.tool_calls` | `parts[].functionCall` |
| `finish_reason=stop` | `finishReason=STOP` |
| `finish_reason=length` | `finishReason=MAX_TOKENS` |
| `finish_reason=content_filter` | `finishReason=SAFETY` |
| `usage` | `usageMetadata` |
| `reasoning_content` | `parts[].thought=true` |

## Thinking / Reasoning 处理

部分上游 reasoning 模型会暴露推理内容，但格式与 Gemini 的 `thought` part 不一致。项目当前支持以下格式：

- 文本标签：
  - `<think>...</think>`
  - `<thinking>...</thinking>`
  - `<reflection>...</reflection>`
  - `<reasoning>...</reasoning>`
  - `<antml:thinking>...</antml:thinking>`
- OpenAI-compatible 扩展字段：
  - `message.reasoning_content`
  - `delta.reasoning_content`
- Gemini 原生：
  - `parts[].thought = true`

当前策略是：**将可识别的 reasoning 内容转换为 Gemini `thought: true` part**，而不是简单丢弃。

示例：

```json
{
  "message": {
    "reasoning_content": "先分析问题。",
    "content": "最终答案。"
  }
}
```

会转换为：

```json
{
  "parts": [
    {"thought": true, "text": "先分析问题。"},
    {"text": "最终答案。"}
  ]
}
```

## 重试策略

非流式 `generateContent` 对以下错误进行重试：

- HTTP 408
- HTTP 429
- HTTP 500 / 502 / 503 / 504
- 网络连接错误
- 超时错误

默认配置：

```env
RETRY_MAX_ATTEMPTS=3
RETRY_BASE_DELAY=1.0
```

流式接口不做重试，避免客户端收到重复 chunk。

## 测试

安装测试依赖：

```bash
pip install -r requirements-dev.txt
```

运行全部测试：

```bash
pytest tests/ -v
```

在当前开发环境中，也可以使用固定虚拟环境：

```bash
/media/data/venv/bin/python -m pytest tests/ -v
```

当前测试覆盖：

- 协议字段转换
- 流式 SSE 转换
- thinking / reasoning 适配
- 上游重试
- 鉴权失败
- 模型发现
- 端到端 mock 测试

## 配置项

| 环境变量 | 必填 | 默认值 | 说明 |
|----------|------|--------|------|
| `UPSTREAM_OPENAI_URL` | 是 | 无 | 上游 OpenAI-compatible 服务地址，不含 `/v1` 后缀 |
| `UPSTREAM_API_KEY` | 是 | 无 | 发给上游的 Bearer token |
| `UPSTREAM_MODEL` | 是 | 无 | 强制使用的上游模型名 |
| `PROXY_API_KEY` | 是 | 无 | 客户端访问本网关所需的 API key |
| `HOST` | 否 | `0.0.0.0` | 监听地址 |
| `PORT` | 否 | `8000` | 监听端口 |
| `LOG_LEVEL` | 否 | `INFO` | 日志级别 |
| `REQUEST_TIMEOUT` | 否 | `600` | 上游请求超时时间（秒） |
| `RETRY_MAX_ATTEMPTS` | 否 | `3` | 非流式请求最大尝试次数 |
| `RETRY_BASE_DELAY` | 否 | `1.0` | 指数退避基础秒数 |
| `MODEL_DISCOVERY_CACHE_TTL` | 否 | `300` | 模型列表缓存时间（秒），`0` 表示不缓存 |

## 安全注意事项

- 不要提交 `.env` 文件。
- `PROXY_API_KEY` 应使用高强度随机值。
- 上游 `UPSTREAM_API_KEY` 不会返回给客户端，但会用于请求上游。
- 如果部署在公网，建议在反向代理层增加 TLS、访问控制和限流。
- 流式和 thinking 日志可能包含用户内容，生产环境建议避免输出完整请求/响应正文。

## 开发说明

代码风格以清晰、显式、协议严谨为主：

- 协议转换逻辑集中在 `app/services/transformer/`。
- 流式转换逻辑集中在 `app/services/stream/processor.py`。
- 错误归一化集中在 `app/utils/error_handler.py`。
- 配置集中在 `app/core/config.py`。

修改协议映射时，建议同步更新：

1. 对应设计文档：`docs/design/`
2. 测试：`tests/`
3. README 中相关说明

## License

当前仓库未声明许可证。请根据你的发布计划补充 `LICENSE` 文件。
