# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 1. 项目定位

`gemini_proxy` 是一个**协议转换网关**，部署在「使用 Gemini SDK / OpenAI SDK 的客户端」与「原生 OpenAI 兼容模型服务」之间。

核心功能：

- **入站**：同时暴露 **Gemini 协议**端点与 **OpenAI 协议**端点，绑定到**同一个 Base URL**。客户端（无论是 Google GenAI SDK、还是 OpenAI SDK 指向本网关）无需感知底层的协议差异。
- **出站**：统一请求底层的 **OpenAI 兼容模型服务**（如任何提供 `/v1/chat/completions` 的服务，例如通过 LiteLLM / 自建 OpenAI 兼容网关 / 厂商 OpenAI 兼容端点）。
- **适配**：当收到 Gemini 协议请求时，将其**映射**为 OpenAI Chat Completions 请求，发送给上游；当上游返回 OpenAI 响应时，再**重构**为 Gemini 协议响应返回给客户端。
- **模型分发**：入站与出站使用同一 Base URL，使网关可以无缝「混合」调用——例如 `GET /v1beta/models`（Gemini）和 `GET /v1/models`（OpenAI）可由同一进程基于同一上游模型列表派生。

> ⚠️ 本项目目前处于从零起步阶段，目录是空的。下面章节明确标注的"计划"将在实现时落地，**禁止**在文档中伪装尚未存在的文件、命令或测试。

## 2. 目录与模块规划

> 以下是**目标目录结构**（尚未实现时按此规划；落地后按本节执行）。

```
gemini_proxy/
├── app/                        # 应用源代码
│   ├── main.py                 # FastAPI 实例、lifespan、路由挂载
│   ├── api/                    # 路由层（按协议切分）
│   │   ├── gemini.py           # Gemini 协议路由：/v1beta/models, /v1beta/models/{name},
│   │   │                       # /v1beta/models/{name}:generateContent, :streamGenerateContent,
│   │   │                       # /v1beta/models/{name}:countTokens, :embedContent 等
│   │   └── openai.py           # OpenAI 协议路由：/v1/models, /v1/chat/completions,
│   │                           # /v1/embeddings, /v1/images/generations（按需）
│   ├── core/                   # 配置 / 鉴权 / 日志 / 常量
│   │   ├── config.py           # pydantic-settings：上游 OpenAI 兼容 URL 与 API Key、
│   │   │                       # 网关入口 HOST/PORT、PROXY_API_KEY 等
│   │   ├── auth.py             # Gemini 风格 key= 查询参数、OpenAI Bearer、x-api-key
│   │   │                       # 多模式鉴权（参考 claude_proxy）
│   │   └── logging.py
│   ├── services/
│   │   ├── transformer/        # 协议转换引擎（核心）
│   │   │   ├── to_openai.py    # Gemini 请求 → OpenAI 请求
│   │   │   ├── from_openai.py  # OpenAI 响应 → Gemini 响应
│   │   │   └── fields.py       # 字段映射表 / 枚举映射（role、finishReason、
│   │   │                       #   safety、functionCall ↔ tool_calls 等）
│   │   ├── stream/             # Gemini streamGenerateContent ↔ OpenAI SSE
│   │   │   └── processor.py    # 维护状态机，跨分片缓冲并转换 SSE 事件
│   │   └── discovery.py        # 模型发现：向上游 `/v1/models` 拉取，构造两侧
│   │                           # 都兼容的 model 列表
│   ├── adapters/               # 上游差异适配（按上游厂商拆分）
│   │   ├── base.py             # BaseAdapter：注入 system prompt、检测文本内联工具
│   │   ├── factory.py          # 根据模型名路由到具体适配器
│   │   └── default.py          # 默认适配器
│   ├── models/                 # Pydantic 数据模型：Gemini（generateContent 请求/响应、
│   │                           #   candidate / content / part / functionCall 等）
│   └── utils/
│       └── error_handler.py    # 上游错误码 → Gemini/OpenAI 错误对象映射
├── docs/
│   ├── requirements/
│   └── design/
├── tests/                      # 单元 + E2E（geminicli → gemini_proxy → OpenAI 兼容上游）
├── .env.example                # 模板（不含敏感值）
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── TODO.md
```

## 3. 命令速查（待实现）

> 实际命令在依赖与构建文件落地后填入。当前可预见的命令模式：

```bash
# 激活虚拟环境（与全局约定一致，使用 /media/data/venv 下的 venv）
source /media/data/venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 启动开发服务（uvicorn reload 仅用于本地）
python -m app.main

# 运行全部测试
pytest

# 运行单个测试文件
pytest tests/test_transformer.py

# 运行单个测试函数
pytest tests/test_transformer.py::test_gemini_request_to_openai_basic

# docker-compose 启动
docker compose up -d

# 健康检查
curl http://localhost:${PORT:-8000}/health
```

测试矩阵必须覆盖：

- Gemini `generateContent`（非流）→ OpenAI `chat/completions`（非流）→ Gemini 响应回包
- Gemini `streamGenerateContent`（SSE）↔ OpenAI 流式 delta 双向转换
- `functionCalling` / `tool_use` 互转（包括 OpenAI `tools` ↔ Gemini `tools.functionDeclarations`）
- `countTokens` 接口的转发或本地估算
- 模型列表 `GET /v1beta/models` 与 `GET /v1/models` 同源派生

## 4. 双协议边界：核心架构

网关的本质是一个**双入单出**的协议转换器：

```
                       ┌──────────────── gemini_proxy ────────────────┐
                       │                                                  │
[Gemini SDK clients] ──┤  /v1beta/...    (Gemini 原生端点)              │
                       │       │                                          │
                       │       ▼                                          │
                       │  ┌──────────────────┐                           │
                       │  │ Gemini → OpenAI  │                           │
                       │  │ request mapper   │                           │
                       │  └────────┬─────────┘                           │
                       │           ▼                                     │
[OpenAI SDK clients] ──┤  /v1/...      (OpenAI 兼容端点，                │
                       │       │       通常是直转发或仅最小改写)        │
                       │       ▼                                         │
                       │  OpenAI Chat Completions 请求 ───►  [Upstream]   │
                       │       ▲                                         │
                       │       │                                         │
                       │  ┌──────────────────┐                           │
                       │  │ OpenAI → Gemini  │                           │
                       │  │ response mapper  │  (仅 Gemini 路径需要)      │
                       │  └──────────────────┘                           │
                       └──────────────────────────────────────────────────┘
```

### 4.1 入站路由判别

`main.py` 启动时同时挂载 `gemini` 与 `openai` 两组 router，路径分别以 `/v1beta` 与 `/v1` 前缀注册。**关键不变量**：两侧共享同一 `BaseURL`，因此客户端只需将 SDK 的 `baseURL` 指向网关即可，剩余协议识别由网关完成。

每个协议路由内部独立处理鉴权与协议结构，不要在网关层做"协议探测"再分发——这会让 Gemini 的 `?key=` 查询参数和 OpenAI 的 `Authorization: Bearer` 头互相干扰。

### 4.2 协议转换核心差异（Gemini ↔ OpenAI）

实现 `app/services/transformer/` 时，重点覆盖以下字段映射：

| 维度 | Gemini (`generateContent`) | OpenAI (`chat.completions`) |
|------|----------------------------|------------------------------|
| 内容载体 | `contents[].parts[]` | `messages[].content` |
| 角色 | `user` / `model`（外加 `system` 在 `systemInstruction`） | `system` / `user` / `assistant` / `tool` |
| 系统提示 | 顶层 `systemInstruction.parts[].text` | `messages[0].role="system"` |
| 工具 | 顶层 `tools[].functionDeclarations` | 顶层 `tools[].function` |
| 工具选择 | `toolConfig.functionCallingConfig.mode` | `tool_choice` |
| 工具调用结果 | 历史 `contents` 中由 `model` 段发出 `functionCall`，由 `user` 段用 `functionResponse` 回传 | `assistant.tool_calls` + 下一条 `role="tool"` |
| 助手首条内容 | `candidates[0].content.parts[]` | `choices[0].message` |
| 结束原因 | `candidates[0].finishReason`（`STOP`、`MAX_TOKENS`、`SAFETY`、`RECITATION`、`OTHER`） | `choices[0].finish_reason`（`stop`、`length`、`tool_calls`、`content_filter`、`function_call`） |
| 用量 | `usageMetadata`（`promptTokenCount`、`candidatesTokenCount`、`thoughtsTokenCount`、`totalTokenCount`） | `usage`（`prompt_tokens`、`completion_tokens`、`total_tokens`） |
| 安全 | `safetyRatings` / `safetySettings` | 无内建概念（一般映射为 `content_filter`） |
| 思考 | `parts[].thought=true` + `thoughtsTokenCount` | 无内建（可借助 `reasoning_effort`，或经 `extra_body.thinking_config` 转发） |
| 响应格式 | `generationConfig.responseMimeType` / `responseSchema` | `response_format` |
| 参数 | `temperature`、`topP`、`topK`、`maxOutputTokens` 等 | `temperature`、`top_p`、`n`、`max_tokens` 等 |
| 嵌入 | `embedContent` / `batchEmbedContents` | `embeddings.create` |
| 多模态 | `parts[].inline_data`（base64 + `mime_type`） | `content[].image_url`（含 `data:...;base64,` URL）/ `input_audio` |

实现时建议把字段映射集中到 `app/services/transformer/fields.py`，避免在请求/响应两侧散落硬编码。

### 4.3 流式转换（`streamGenerateContent`）

Gemini 的流式响应是分号分隔的 JSON 对象数组（不是 SSE）；OpenAI 的流式是 `data: {...}\n\n` 加 `[DONE]`。两边协议形态不同，必须由 `stream/processor.py` 中的状态机统一翻译为目标协议的事件序列：

- 入站 = Gemini：网关**消费** Gemini JSON 数组流，重组为 OpenAI SSE 发给上游（如果是 Gemini→OpenAI 路径）。
- 入站 = Gemini 路径且直连 OpenAI 时需要回包：网关**消费** OpenAI SSE 流，重新切成 Gemini JSON 对象流（按 `candidates[].content.parts[].text` 增量下发）回给客户端。
- 入站 = OpenAI：原则上**透传**，仅处理必要的错误归一化与 SSE 头兼容。

流处理器的内部状态至少要跟踪：`message_id` / `response_id`、`累积文本 buffer`、`tool_calls 增量`、`finishReason 锁定`、`usage 终结`（OpenAI 在尾部单独 chunk 给 `usage`）。

### 4.4 鉴权（双模式）

`app/core/auth.py` 需支持两种调用方习惯：

- **Gemini 风格**：URL 查询参数 `?key=...`（Google GenAI SDK 默认行为）。
- **OpenAI 风格**：`Authorization: Bearer <key>` 头，**以及** `x-api-key` 头（部分 SDK）。

网关对外只暴露一个 `PROXY_API_KEY`，对内向上游注入上游凭据（`OPENAI_COMPAT_API_KEY`），与 `claude_proxy` 的「双向鉴权」一致。

### 4.5 模型发现（`discovery.py`）

上游 `/v1/models` 拉取 → 缓存为内部列表 → 在两侧端点按各自 schema 暴露：
- `GET /v1beta/models` → Gemini `models[]`，每条含 `name`（如 `models/{id}`）、`displayName`、`supportedGenerationMethods`。
- `GET /v1/models` → OpenAI `{ data: [...], object: "list" }`。

注意 Gemini 的 `name` 必须带 `models/` 前缀，与 OpenAI 的 `id` 不同。

## 5. 部署 / 容器

参照 `claude_proxy` 的 `Dockerfile` 与 `docker-compose.yml` 模式复用：

- 镜像使用 `python:3.12-slim`。
- `HEALTHCHECK` 指向 `/health`。
- 卷挂载 `/etc/localtime` 保证时区。
- **必须将 `.venv/`、`.env`、`.learnings/` 写入 `.gitignore`**。
- **仅提交 `.env.example`**。

## 6. 与 `claude_proxy` 的关系

`claude_proxy`（位于 `/media/data/git/claude_proxy`，已完成）实现了"Claude 协议 ↔ OpenAI 协议"的对称转换。`gemini_proxy` 复用其**架构模式**（双协议入口、转换引擎 + 流处理器 + 适配器工厂、Pydantic 配置、双模式鉴权、TODO/`.learnings` 知识沉淀规范），但**不直接 import**——保持两个项目独立维护，各自演进。

具体可复用借鉴（不是必须一致）：

- `app/core/{config,auth,logging}.py` 的结构与依赖注入模式。
- `app/services/transformer/engine.py` 的「两侧字段映射 + 适配器注入」分层。
- `app/services/stream/processor.py` 的「状态机 + 工具调用中途注入 + finishReason 修正」实现思路。
- `tests/` 中单元 + E2E 的双层覆盖策略。

## 7. 强制行为准则

- **文档优先**：任何功能开发或架构调整，**先**更新 `docs/requirements/`、`docs/design/`，再动代码。
- **协议严谨性**：对外暴露的每一个字节都必须符合目标 SDK 的协议校验（Google GenAI SDK 对 Gemini 字段是严格的；OpenAI SDK 对 `finish_reason`、`tool_calls.id` 唯一性等也敏感）。
- **内容递增**：更新 `.learnings/` 或 `TODO.md` 时严禁删减既有有效内容；过时内容标注 `> ⚠️ 已过时：[原因]`，不删除。
- **Git 安全**：`commit` 前必须确认 `.env`、`.venv/`、`.learnings/` 未被追踪（参考 `claude_proxy/.gitignore`）。
- **知识沉淀**：每发现一个 Gemini ↔ OpenAI 的字段/行为差异，立即写入 `.learnings/knowledge/`；错误修复写入 `experience/`；优化方案写入 `best_practice/`。

## Related

- `/media/data/git/claude_proxy/` — 已完成的镜像参考项目（Claude ↔ OpenAI 协议网关），作为架构与代码风格的参照标杆。
- `/media/data/git/claude_proxy/CLAUDE.md`-待生成版（如后续创建可作为本文件的对照）— 同一作者体系下的协议网关写法范例。
- `.learnings/index.md` — 本地知识库总索引（落地后建立）。
- `TODO.md` — 项目待办事项（落地后建立）。
