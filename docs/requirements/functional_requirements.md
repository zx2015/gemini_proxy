# 功能需求文档 (Functional Requirements)

> 适用版本：v0.1.0
> 最后更新：2026-07-02
>
> 修订记录：
> - 2026-07-02：v0.1.0 初稿（双协议入口、单入单出、字段映射表、错误归一化）
> - 2026-07-02：v0.1.0-r1 新增 §2.7 强制上游模型覆盖、§2.8 上游失败重试机制；删除 `MODEL_MAPPING` 配置项

## 1. 项目背景

`gemini_proxy` 是一个基于 Python（FastAPI）的协议适配网关，部署在 `gemini-cli`（Google Gen AI SDK）与上游 **OpenAI 兼容模型服务**之间。其核心目标：

- 拦截 `gemini-cli` 发出的 **Gemini 协议**（`generateContent` / `streamGenerateContent`）请求；
- 在网关层**重写**请求体为 **OpenAI Chat Completions** 规范；
- 在收到上游响应后，**重写**回 Gemini 规范再返回给客户端；
- 客户端**无需感知**底层协议差异，只需将 SDK 的 `baseURL` / `endpoint` 指向本网关。

`gemini-cli` 默认走的是 Google Gen AI SDK（HTTP 风格），因此网关主要监听 **`/v1beta` 前缀**路径。本项目**暂不**提供 OpenAI 入站协议支持——这是一条**单入单出**的链路：入站 Gemini，出站 OpenAI。

> 📌 长期目标：参考 `claude_proxy` 的双协议入口架构，未来可演进为「同时支持 Gemini 协议与 OpenAI 协议入站」的双入单出门关。**当前版本以单入单出为目标，避免范围蔓延。**

## 2. 核心功能需求

### 2.1 协议拦截与转发

- **技术栈**：Python 3.10+、FastAPI、HTTPX（异步）、Pydantic v2。
- **拦截范围（必做）**：
  - `POST /v1beta/models/{model}:generateContent`（非流式）
  - `POST /v1beta/models/{model}:streamGenerateContent`（SSE/JSON 数组流）
  - `GET /v1beta/models`（模型列表，可选；详见 2.4）
- **目标上游**：`POST {UPSTREAM_OPENAI_URL}/v1/chat/completions`（流式时设置 `stream: true`）。
- **鉴权抓取**：必须同时支持以下三种凭据传递方式：
  1. HTTP Header `x-goog-api-key: <key>`
  2. HTTP Header `Authorization: Bearer <key>`
  3. URL Query 参数 `?key=<key>`（部分老版本 Gen AI SDK 行为）

> 网关对外的入站 `PROXY_API_KEY` 与上游 `UPSTREAM_API_KEY` 解耦——入站凭据用于客户端认证；出站网关向 `Authorization: Bearer <UPSTREAM_API_KEY>` 注入上游凭据。

### 2.2 请求体转换（Gemini → OpenAI）

详见 [docs/design/transformer.md §1](../design/transformer.md)，此处只列强约束：

| Gemini 字段 | OpenAI 字段 | 强约束 |
|-------------|-------------|--------|
| URL `{model}` | `model` | **强制覆盖**：忽略 URL 中的 `{model}`，**始终**使用环境变量 `UPSTREAM_MODEL` 指定的值（详见 [§2.7](#27-强制上游模型覆盖)） |
| `contents[].role="user"` | `messages[].role="user"` | 直接映射 |
| `contents[].role="model"` | `messages[].role="assistant"` | **必须**转换 |
| `contents[].role="function"` | `messages[].role="tool"` | 函数调用回传，详见 2.3 |
| `systemInstruction.parts[].text` | `messages[0].role="system"` | 提取为第一条 system 消息 |
| `contents[].parts[].text` | `messages[].content` | 纯文本直接取 `parts[0].text` |
| `contents[].parts[].inline_data` | `messages[].content[].image_url` | 多模态，按 base64 data URL 形式重写 |
| `generationConfig.temperature` | `temperature` | 直接映射 |
| `generationConfig.maxOutputTokens` | `max_tokens` | 直接映射 |
| `generationConfig.topP` | `top_p` | 直接映射 |
| `generationConfig.topK` | （不支持 → 透传或丢弃） | 默认丢弃，详见设计 |
| `tools[].functionDeclarations` | `tools[].function` | 嵌套结构展平 |
| `toolConfig.functionCallingConfig.mode` | `tool_choice` | `AUTO`→`auto`，`ANY`→`required`，`NONE`→`none` |

### 2.3 响应体转换（OpenAI → Gemini）

| OpenAI 字段 | Gemini 字段 | 强约束 |
|-------------|-------------|--------|
| `choices[0].message.role="assistant"` | `candidates[0].content.role="model"` | **必须**转换 |
| `choices[0].message.content` | `candidates[0].content.parts[0].text` | 文本提取 |
| `choices[0].message.tool_calls` | `candidates[0].content.parts[].functionCall` | 工具调用重写 |
| `choices[0].finish_reason="stop"` | `finishReason="STOP"` | 直接映射 |
| `choices[0].finish_reason="length"` | `finishReason="MAX_TOKENS"` | 直接映射 |
| `choices[0].finish_reason="tool_calls"` | `finishReason="STOP"` + 含 `functionCall` | Gemini 无独立 `TOOL_CALLS`，统一用 `STOP` |
| `choices[0].finish_reason="content_filter"` | `finishReason="SAFETY"` | 语义最近对齐 |
| `usage.prompt_tokens` | `usageMetadata.promptTokenCount` | 字段重命名 |
| `usage.completion_tokens` | `usageMetadata.candidatesTokenCount` | 字段重命名 |
| `usage.total_tokens` | `usageMetadata.totalTokenCount` | 字段重命名 |

### 2.4 辅助功能

- **模型列表端点**：`GET /v1beta/models`（可选实现 v0.1.0），向上游 `GET /v1/models` 拉取后转换为 Gemini `models[]` 格式。
- **Token 计数**：`POST /v1beta/models/{model}:countTokens`，先尝试转发给上游；失败则按字符数启发式估算（`字符数 / 3 + 20`）。
- **多模态**：`inline_data`（含 `mime_type` 与 base64）→ `image_url`（`data:{mime_type};base64,{data}`）。
- **函数调用（Function Calling）**：完整支持 `tools.functionDeclarations` ↔ `tools.function`；`functionResponse` 消息与 OpenAI `role="tool"` 消息互转。
- **响应格式约束**：`generationConfig.responseMimeType` ↔ OpenAI `response_format.type`。

### 2.5 错误处理与对齐

- **入站错误**：HTTP 401（鉴权失败）、HTTP 400（请求体反序列化失败）。
- **上游错误**：捕获 `httpx.HTTPStatusError` 与 `httpx.RequestError`，映射为 Gemini 错误对象：
  ```json
  {
    "error": {
      "code": 400,
      "message": "<reason>",
      "status": "INVALID_ARGUMENT"
    }
  }
  ```
- **流式错误**：在 JSON 数组流模式下，写入 `{"error": {...}}` 作为流的最后一个元素。
- **超时**：上游请求超时（默认 600s）必须按 504 错误回包，避免网关层 hang。

### 2.6 安全与配置

- **双向鉴权**：入站校验 `PROXY_API_KEY`；出站注入 `UPSTREAM_API_KEY`。
- **Header 透传策略**：`x-goog-api-key` / `Authorization` 头在转发到上游时**必须替换**，**严禁**透传。
- **配置项**：
  - `UPSTREAM_OPENAI_URL`（如 `http://localhost:4000`）
  - `UPSTREAM_API_KEY`（如 `sk-empty`）
  - `UPSTREAM_MODEL`（**新增**：上游 OpenAI 兼容服务的目标模型名，如 `gpt-4o`；详见 [§2.7](#27-强制上游模型覆盖)）
  - `PROXY_API_KEY`（客户端访问本网关的凭据）
  - `HOST` / `PORT`（默认 `0.0.0.0` / `8000`）
  - `LOG_LEVEL`（默认 `INFO`）
  - ~~`MODEL_MAPPING`~~（**已移除**：v0.1.0-r1 起不再支持模型名映射，详见 [§2.7](#27-强制上游模型覆盖)）

### 2.7 强制上游模型覆盖

- **入站 model 完全忽略**：客户端在 URL 路径 `models/{model}:generateContent` 或请求体任何位置传入的 `model` 字段均**不**用于出站请求。
- **强制使用 `UPSTREAM_MODEL`**：出站 `POST /v1/chat/completions` 的 `model` 字段**始终**等于 `settings.upstream_model`。
- **不返回错误**：客户端传入不存在的模型名（如 `gemini-3-ultra`）时，网关**不**校验、**不**返回 4xx，照常转发——**避免**误传模型导致的 404 / 计费错乱。
- **日志可观测**：每次请求记录 `inbound_model=<X> → outbound_model=<UPSTREAM_MODEL>` 映射关系，便于排查。

> 📌 实现位置：`app/services/transformer/to_openai.py` 的 `RequestTransformer` 中——**直接**把 `openai_req["model"]` 设为 `settings.upstream_model`，**不要**从入站 payload 提取。

### 2.8 上游失败重试机制

仅对**非流式**请求生效；流式请求**不**重试（详见 [stream_handler.md §6](../design/stream_handler.md#6-错误处理)）。

#### 2.8.1 重试参数

| 行为 | 配置项 | 默认值 | 说明 |
|------|--------|--------|------|
| 最大尝试次数（含首次） | `RETRY_MAX_ATTEMPTS` | `3` | 即"重试 2 次" |
| 退避基数 | `RETRY_BASE_DELAY` | `1.0` 秒 | 实际延迟 = `base * 2^(attempt-1)` |
| 可重试错误码 | — | `429, 5xx, 网络超时/连接错误` | 见下表 |
| 不可重试错误码 | — | `400, 401, 403, 404, 422` | 立即返回，不重试 |

#### 2.8.2 可重试 vs 不可重试

| HTTP 状态 | 错误类型 | 是否重试 |
|-----------|----------|----------|
| `408` Request Timeout | 客户端请求超时 | **可重试** |
| `429` Too Many Requests | 限流 | **可重试** |
| `500` Internal Server Error | 上游内部错误 | **可重试** |
| `502` Bad Gateway | 上游网关错误 | **可重试** |
| `503` Service Unavailable | 上游不可用 | **可重试** |
| `504` Gateway Timeout | 上游网关超时 | **可重试** |
| `400` Bad Request | 请求格式错误 | **不可重试**（重试无意义） |
| `401` Unauthorized | 鉴权失败 | **不可重试**（重试无意义） |
| `403` Forbidden | 权限不足 | **不可重试** |
| `404` Not Found | 模型/资源不存在 | **不可重试** |
| `422` Unprocessable Entity | 参数校验失败 | **不可重试** |
| `httpx.ConnectError` | TCP 连接失败 | **可重试** |
| `httpx.ReadTimeout` | 读超时 | **可重试** |
| `httpx.WriteTimeout` | 写超时 | **可重试** |
| `httpx.PoolTimeout` | 连接池超时 | **可重试** |
| `httpx.RemoteProtocolError` | 协议错误 | **可重试** |

#### 2.8.3 退避策略

采用**指数退避**（Exponential Backoff），**不**加 jitter（v0.1.0 范围内保持简单）：

| 第 N 次重试 | 距上次失败的时间 |
|------------|------------------|
| 第 1 次重试 | `RETRY_BASE_DELAY * 2^0` = 1.0s |
| 第 2 次重试 | `RETRY_BASE_DELAY * 2^1` = 2.0s |
| （如配置为 4 次）第 3 次重试 | `RETRY_BASE_DELAY * 2^2` = 4.0s |

> 📌 v0.1.0-r1 默认 `RETRY_MAX_ATTEMPTS=3`、`RETRY_BASE_DELAY=1.0`，总退避最坏情况 1s+2s=3s。
> **不引入** jitter 抖动——单实例部署时雪崩风险低；后续如需横向扩容再加入。

#### 2.8.4 日志与可观测性

- 每次重试记录 WARN 日志：`Upstream request failed (attempt 1/3, status=503), retrying in 1.0s`。
- 最终失败记录 ERROR 日志：`Upstream request failed after 3 attempts, last_status=503`。
- 重试**不**对客户端暴露——客户端仍只看到一次最终响应（成功或失败）。
- 建议在 metrics 中暴露 `upstream_request_retries_total`（按状态码分组），便于运维。

#### 2.8.5 重试边界

- **重试内容相同**：每次重试使用**同一份** OpenAI 请求体（`model`、`messages`、`tools` 等），**不**重算、不重写。
- **请求幂等性**：非流式 `chat/completions` 在主流 OpenAI 兼容服务上**幂等**——同一请求返回同一结果，重试安全。
- **超时叠加**：每次重试都有独立的 `REQUEST_TIMEOUT`（默认 600s），总耗时上限 ≈ `3 * 600 + 3` = 1803s，**应**在客户端层设置更短的总超时。

> ⚠️ 已过时：v0.1.0 初稿 §2.5 仅要求"超时 504 回包"，未规定重试。v0.1.0-r1 引入重试机制后，**重试耗尽**的最终错误仍按 §2.5 规则返回 Gemini `error` 包装。

## 3. 非功能需求

- **协议完整性**：返回给 `gemini-cli` 的每一个字节都必须通过 Google Gen AI SDK 的字段校验。
- **低延迟转发**：网关层引入的额外延迟应控制在 50ms 以内（不含模型生成时间）。
- **流式打字机体验**：流式转换不应造成明显的文本输出卡顿或大块弹出，建议**先转发**文本增量，只在流末尾注入 `finishReason` 与 `usageMetadata`。
- **稳定性**：上游 429 / 500 / 502 / 503 / 504 必须能被识别并转换为合规的 Gemini 错误响应，**严禁**将上游原始 HTML / 堆栈泄漏给客户端。

## 4. 详细配置需求

| 变量 | 必填 | 默认 | 说明 |
|------|------|------|------|
| `UPSTREAM_OPENAI_URL` | 是 | — | 上游 OpenAI 兼容服务地址（不含 `/v1`） |
| `UPSTREAM_API_KEY` | 是 | — | 注入到上游 `Authorization: Bearer` |
| `UPSTREAM_MODEL` | 是 | — | **强制**使用的上游模型名（如 `gpt-4o`），忽略客户端传入的 `model`（详见 [§2.7](#27-强制上游模型覆盖)） |
| `PROXY_API_KEY` | 是 | — | 客户端访问本网关的凭据 |
| `HOST` | 否 | `0.0.0.0` | 监听地址 |
| `PORT` | 否 | `8000` | 监听端口 |
| `LOG_LEVEL` | 否 | `INFO` | 日志级别 |
| `REQUEST_TIMEOUT` | 否 | `600` | 上游请求超时（秒） |
| `RETRY_MAX_ATTEMPTS` | 否 | `3` | 上游非流式请求的最大尝试次数（含首次），详见 [§2.8](#28-上游失败重试机制) |
| `RETRY_BASE_DELAY` | 否 | `1.0` | 重试退避基数（秒），实际延迟 = `base * 2^(attempt-1)`，详见 [§2.8](#28-上游失败重试机制) |

> ⚠️ 已过时：v0.1.0 初稿中规划的 `MODEL_MAPPING`（JSON 字符串模型名映射）已于 v0.1.0-r1 起移除。`UPSTREAM_MODEL` 直接指定单一目标模型，**不再**做客户端透明映射。

## 5. 出域边界（Out of Scope for v0.1.0）

为避免范围蔓延，以下功能**不在** v0.1.0 范围内，留待后续版本：

- OpenAI 协议入站（双入单出模式）
- Embeddings、Batch API、Image/Video Generation
- 适配器插件系统（多厂商上游 OpenAI 兼容服务适配）
- Token 用量计费、限流、审计日志
- Web 管理界面

## Related

- [架构设计](../design/architecture.md) — 整体技术栈、拓扑、组件职责
- [转换引擎详细设计](../design/transformer.md) — 字段映射规则、边缘情况
- [流处理详细设计](../design/stream_handler.md) — 流式状态机
- [设计文档总索引](../design/index.md)
- [CLAUDE.md](../../CLAUDE.md) — 项目行为准则
