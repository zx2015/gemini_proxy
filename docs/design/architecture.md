# 整体架构设计 (Architecture)

> 适用版本：v0.1.0
> 最后更新：2026-07-02
>
> 修订记录：
> - 2026-07-02：v0.1.0 初稿
> - 2026-07-02：v0.1.0-r1 新增 §3.5 上游重试机制、§6.5 强制模型覆盖的设计取舍；删除 `MODEL_MAPPING` 设计

## 1. 技术栈

- **核心框架**：FastAPI（Python 3.10+）
- **异步 HTTP 客户端**：HTTPX
- **配置管理**：Pydantic Settings（`pydantic-settings`）
- **数据模型**：Pydantic v2
- **测试**：pytest + pytest-asyncio + respx（mock HTTPX）
- **部署**：uvicorn + Docker

## 2. 系统拓扑

```text
[gemini-cli / GenAI SDK] 
        │
        │ (Gemini 协议：generateContent / streamGenerateContent)
        │  Header: x-goog-api-key / Authorization: Bearer / ?key=
        ▼
┌────────────────────────────────────────────────────────┐
│                    gemini_proxy                         │
│                                                        │
│  1. AuthMiddleware   ── 校验 PROXY_API_KEY            │
│  2. Gemini Router    ── /v1beta/models/{m}:*          │
│  3. Transformer      ── Gemini ⇄ OpenAI              │
│  4. Stream Processor ── SSE/JSON 数组流 状态机        │
│  5. ErrorHandler     ── 上游异常 → Gemini 错误对象     │
│                                                        │
└────────────────────────┬───────────────────────────────┘
                         │
                         │ (OpenAI 协议：chat/completions)
                         │  Header: Authorization: Bearer <UPSTREAM_API_KEY>
                         ▼
              [Upstream OpenAI-Compatible Service]
                  （LiteLLM / 厂商 OpenAI 兼容端点）
```

## 3. 数据生命周期与处理逻辑

### 3.1 请求阶段 (Request Phase)

1. **鉴权 (AuthMiddleware)**：从 `x-goog-api-key` / `Authorization: Bearer` / `?key=` 三个位置之一提取凭据，比对 `settings.proxy_api_key`。失败 → 401。
2. **路由匹配**：FastAPI 路径 `POST /v1beta/models/{model}:generateContent` 或 `:streamGenerateContent`，从 URL 提取 `{model}`。
3. **Header 重写**：
   - **剥离**所有 Gemini 专属头（`x-goog-api-key`、`x-goog-api-client`、`x-request-id` 等）。
   - **注入** `Authorization: Bearer <UPSTREAM_API_KEY>`、`Content-Type: application/json`。
   - **记录**剥离动作到日志（仅记录头名，不记录值，避免泄漏）。
4. **请求体转换**（详见 [transformer.md §1](transformer.md)）：
   - `contents[]` → `messages[]`
   - `systemInstruction` → 首条 system 消息
   - `tools[].functionDeclarations` → `tools[].function`
   - `generationConfig.*` → `temperature` / `max_tokens` / `top_p`
   - 应用 `MODEL_MAPPING`（如 `gemini-2.5-pro` → `gpt-4o`）。
5. **上游调用**：
   - 非流式：`POST {UPSTREAM_OPENAI_URL}/v1/chat/completions`
   - 流式：同上，body 中 `stream: true`。

### 3.2 响应阶段 - 非流式 (Response Phase - Blocking)

1. **完整获取**：等待上游返回完整 JSON。
2. **响应体转换**（详见 [transformer.md §2](transformer.md)）：
   - `choices[0].message` → `candidates[0].content.parts[]`
   - `role="assistant"` → `role="model"`
   - `finish_reason` → `finishReason`（大写枚举）
   - `usage` → `usageMetadata`（字段重命名）。
3. **归一化错误**：若上游返回非 2xx 状态码，由 `ErrorHandler` 转换为 Gemini `error` 包装。

### 3.3 响应阶段 - 流式 (Response Phase - Streaming)

详见 [stream_handler.md](stream_handler.md)，此处仅概览：

1. **建立异步流**：用 `httpx.AsyncClient.stream()` 拿到 OpenAI SSE 行流。
2. **逐行解析**：`data: {...}\n\n` → JSON chunk；遇 `data: [DONE]` 收尾。
3. **重组 Gemini 流**：把每个 `delta.content` 增量包装为：
   ```json
   {"candidates":[{"content":{"role":"model","parts":[{"text":"..."}]}}]}
   ```
   以 `[\n{...},\n{...}\n]` 数组流形式（或分块 JSON）写入客户端。
4. **收尾注入**：在流末尾注入包含 `finishReason` 与 `usageMetadata` 的最后一个块。

### 3.4 辅助路径 - Token 计数 (Token Counting)

1. **拦截**：`POST /v1beta/models/{model}:countTokens`。
2. **策略**：
   - **优先转发**：将 Gemini 请求体尝试转发给上游（OpenAI 无标准计数端点，多数 LiteLLM 部署会拒绝）。
   - **启发式兜底**：本地按 `system + messages + tools` 的字符总数估算（`字符数 / 3 + 20`），返回：
     ```json
     {"totalTokens": <int>}
     ```

### 3.5 上游重试（仅非流式）

v0.1.0-r1 新增，详细规则见 [functional_requirements.md §2.8](../requirements/functional_requirements.md#28-上游失败重试机制)。

1. **触发位置**：在 §3.1 第 5 步（上游调用）与 §3.2 第 1 步（完整获取）之间。
2. **状态机**：
   ```text
   发起上游请求
       │
       ├─ 2xx → 进入 §3.2 转换
       ├─ 408 / 429 / 5xx / 网络错误
       │       │
       │       ├─ attempt < MAX → 退避 base * 2^(attempt-1) 秒 → 重试
       │       └─ attempt == MAX → 走 §3.2 错误归一化（按 transformer.md §3 映射 Gemini 错误）
       └─ 其他 4xx → 走 §3.2 错误归一化（不重试）
   ```
3. **重试内容**：使用同一份 OpenAI 请求体（不变），**不**重新转换、**不**重新计时 token。
4. **日志**：每次重试 WARN；最终失败 ERROR。
5. **流式路径不重试**——流式 `streamGenerateContent` 在 `httpx.AsyncClient.stream()` 上下文内发生错误时**直接**终止并下发 `error` 帧。

## 4. 核心组件职责

| 组件 | 路径 | 职责 |
|------|------|------|
| `AuthMiddleware` | `app/core/auth.py` | 入站 API Key 校验（多模式）；**不**做 IP / 限流 |
| `Settings` | `app/core/config.py` | Pydantic Settings：所有环境变量集中读取 |
| `GeminiRouter` | `app/api/gemini.py` | 注册 `/v1beta` 前缀下的所有路由，调用 Transformer 与 StreamProcessor |
| `RequestTransformer` | `app/services/transformer/to_openai.py` | Gemini 请求体 → OpenAI 请求体 |
| `ResponseTransformer` | `app/services/transformer/from_openai.py` | OpenAI 响应体 → Gemini 响应体 |
| `StreamProcessor` | `app/services/stream/processor.py` | OpenAI SSE 行流 → Gemini JSON 数组流的状态机 |
| `ErrorHandler` | `app/utils/error_handler.py` | `httpx` 异常 → Gemini `error` 包装；`finishReason` 映射 |
| `ModelDiscovery` | `app/services/discovery.py` | 拉取上游 `/v1/models`，转换为 Gemini `models[]` 格式 |

## 5. 配置与启动流程

```text
.env → pydantic-settings → Settings 单例
        │
        ├──> uvicorn 启动参数（host / port）
        ├──> AuthMiddleware 比对
        ├──> 上游 HTTPX 客户端注入
        └──> ModelDiscovery 启动时缓存
```

启动命令：
```bash
python -m app.main
# 等价于：
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 6. 关键设计取舍

### 6.1 为什么单入单出而非双入单出

- **v0.1.0 范围聚焦**：核心目标是把 `gemini-cli` 接通，**不需要**额外引入 OpenAI 客户端入站路径。
- **降低协议风险**：双入单出需要同时维护两套鉴权、两套错误对象、两套流式状态机，**短期投入产出比低**。
- **未来扩展性**：目录结构（`app/api/{gemini,openai}.py`）和组件命名（`GeminiRouter` / 未来 `OpenAIRouter`）已经预留双入扩展位。

### 6.2 为什么不在网关层做协议探测

- Gemini 的 `?key=` 查询参数与 OpenAI 的 `Authorization: Bearer` 头**冲突**——若由一个中间件做"看哪个先匹配"，将引入微妙的安全漏洞。
- **结论**：在 **FastAPI 路由层**用路径前缀（`/v1beta` vs `/v1`）天然区分协议；中间件层只做鉴权与日志，不做协议分支。

### 6.3 为什么流式转换优先"先转发"再"收尾注入"

- 用户对 `gemini-cli` 的"打字机"体验极其敏感——任何**先缓冲再下发**都会带来可感知的卡顿。
- **策略**：流式过程中**立刻**把 `delta.content` 转换为 Gemini 增量下发；只在流末尾一次性注入 `finishReason` + `usageMetadata`。
- **代价**：若客户端中途断开，已经下发的部分无法回收——可接受（HTTP 协议语义本就如此）。

### 6.4 取消 `MODEL_MAPPING`，改用 `UPSTREAM_MODEL` 单一强制

> ⚠️ 已过时：v0.1.0 初稿曾设计 `MODEL_MAPPING: dict[str, str]`（JSON 字符串）以支持 `gemini-2.5-pro → gpt-4o` 的客户端透明映射。v0.1.0-r1 起该设计**已被取消**，改为环境变量 `UPSTREAM_MODEL`（单一字符串）强制覆盖。

- **原因**：用户偏好"传过来的模型不管是什么，只用指定的一个模型"——多模型映射无意义。
- **新设计**：
  - 客户端传入的 `model` **完全忽略**（不校验、不映射、不警告）。
  - 出站 `chat/completions` 的 `model` 字段**始终**等于 `settings.upstream_model`。
- **实现简化**：`RequestTransformer` 不再读取 URL 路径 `{model}`，配置层无 JSON 解析，运维更简单。
- **可观测性**：每次请求记录 `inbound_model → outbound_model` 映射日志，便于排查。

### 6.5 引入上游重试机制

v0.1.0-r1 新增（详见 [functional_requirements.md §2.8](../requirements/functional_requirements.md#28-上游失败重试机制)）：

- **范围**：仅**非流式**请求（`generateContent`）。流式**不**重试——流式连接已建立后再失败，重试会让客户端拿到重复的 chunk，复杂度高。
- **重试次数**：3 次（含首次）= 重试 2 次。
- **可重试错误**：5xx + 429 + 网络超时/连接错误。
- **退避策略**：指数退避 1s/2s（无 jitter，保持简单）。
- **重试内容相同**：每次重试使用同一份 OpenAI 请求体，依赖上游 `chat/completions` 的幂等性。
- **失败归一化**：重试耗尽后，**仍**按 [transformer.md §3](../design/transformer.md#3-错误归一化响应方) 规则返回 Gemini `error` 包装，不暴露重试细节给客户端。

> 📌 之前 §6.4 标题"为什么 `MODEL_MAPPING` 是 JSON 字符串而非 YAML/TOML"在 v0.1.0-r1 已被 §6.4 "取消 `MODEL_MAPPING`..." 替代；本节为新追加。

## 7. 部署拓扑

```text
┌──────────────────┐    ┌─────────────────────┐    ┌──────────────────────┐
│   gemini-cli     │───▶│   gemini_proxy      │───▶│  OpenAI 兼容上游     │
│   (本机 / 容器)  │    │   (FastAPI 容器)    │    │  (LiteLLM / 厂商)   │
└──────────────────┘    └─────────────────────┘    └──────────────────────┘
                            :8000                       :4000
```

健康检查：`GET /health` 返回 `{"status":"healthy"}`；Docker healthcheck 直接 curl 该端点。

## Related

- [转换引擎详细设计](transformer.md) — 字段映射、边缘情况、错误归一化
- [流处理详细设计](stream_handler.md) — 流式状态机、SSE/JSON 数组流转换
- [功能需求文档](../requirements/functional_requirements.md) — 需求源头
- [设计文档总索引](index.md)
- [CLAUDE.md](../../CLAUDE.md) — 项目行为准则
