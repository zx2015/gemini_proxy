# 流处理详细设计 (Stream Handler)

> 适用版本：v0.1.0
> 最后更新：2026-07-02
>
> 修订记录（内容递增）：
> - 2026-07-02：v0.1.0 初稿
> - 2026-07-02：v0.1.0-r1 §1 修订——**纠正关键认知错误**：Gemini SDK 期望 SSE 格式而非"裸 JSON 数组流"；详见附录 A 与 `.learnings/experience/stream-sse-format-fix.md`
>
> 本文档是 `app/services/stream/processor.py` 的实现规范。

## 1. 协议流式差异

| 维度 | Gemini `streamGenerateContent` | OpenAI `chat/completions`（stream=true） |
|------|--------------------------------|-----------------------------------------|
| 传输编码 | `text/event-stream`（SSE） | `text/event-stream`（SSE） |
| 帧格式 | `data: [{...}]\n\n`（**数组**包单个对象） | `data: {...}\n\n`（**单对象**） |
| 结束信号 | 服务端关闭连接 | `data: [DONE]\n\n` |
| 增量粒度 | 字符级（通常按 token） | 字符级（`delta.content`） |
| Tool calls | 嵌在 `candidates[0].content.parts[].functionCall` 中一次性给出 | `delta.tool_calls` 是**分片累加**结构（`index`/`id`/`function.name`/`function.arguments`） |

> ⚠️ **关键陷阱 1（已修订）**：v0.1.0 初稿曾误认为 Gemini 流式是"裸 JSON 数组流"（`[{},{}]`）。**实际** `@google/genai` SDK 期望 SSE 格式，且每个 `data:` 字段后跟一个**JSON 数组**（即使只有 1 个元素）。详见 [附录 A](#附录-a-sse-格式的关键差异)。
>
> ⚠️ **关键陷阱 2**：OpenAI 流式中 `tool_calls` 不是一次性给的——`id`、`name`、`arguments` 都分多个 chunk 增量推送。Gemini 客户端期望一次性看到 `functionCall`。因此流转换器必须**缓存并拼装**所有 `tool_calls` 增量，在收到 `finish_reason` 后一次性注入。

## 2. 模块架构

```
app/services/stream/
├── __init__.py
└── processor.py        # StreamProcessor（主类）
```

调用方：
- `app/api/gemini.py` 中 `:streamGenerateContent` 路由构造 `StreamProcessor(model)`，传入 `httpx` 异步流对象。

```python
async def stream_generate_content(model: str, body: dict):
    processor = StreamProcessor(model=model)
    openai_body = request_transformer.transform(body, model=model, stream=True)
    async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
        async with client.stream(
            "POST", f"{settings.upstream_openai_url}/v1/chat/completions",
            json=openai_body,
            headers={"Authorization": f"Bearer {settings.upstream_api_key}"}
        ) as resp:
            async for chunk in processor.process(resp.aiter_lines()):
                yield chunk
```

## 3. StreamProcessor 状态机

### 3.1 内部状态

```python
class StreamProcessor:
    def __init__(self, model: str):
        self.model = model
        self.buffer = ""                 # 累积文本（必要时）
        self.tool_calls_acc: dict[int, dict] = {}  # OpenAI tool_calls 分片聚合
        self.last_finish_reason: str | None = None
        self.last_usage: dict | None = None
        self.array_mode = True           # 是否用数组流模式（参见 §4.2）
        self.is_first_chunk = True       # 用于数组流的 `[` 与逗号
        self.closed = False
```

### 3.2 事件驱动主循环

```text
[init] → emit "["
for each openai_chunk:
    parse JSON
    if delta.content:
        emit Gemini text-delta block
    if delta.tool_calls:
        accumulate tool_calls_acc
    if finish_reason:
        last_finish_reason = finish_reason
    if usage chunk:
        last_usage = usage
[end] → 
    if tool_calls_acc non-empty:
        emit Gemini functionCall block
    if last_finish_reason:
        emit Gemini finishReason block
    if last_usage:
        emit Gemini usageMetadata block
    emit "]"
```

### 3.3 方法签名

```python
async def process(self, openai_stream) -> AsyncGenerator[bytes, None]:
    """
    入参：openai_stream 为 httpx response 的 aiter_lines() 异步迭代器。
    出参：返回 bytes（JSON 序列化的 UTF-8），可直接作为 StreamingResponse 内容写出。
    """
```

## 4. 输出帧格式

### 4.1 SSE `data: [{...}]` 格式（默认且唯一）

**正确格式**（`@google/genai` SDK 期望）：

```
data: [{"candidates": [{"content": {"role": "model", "parts": [{"text": "..."}]}}]}]\n\n
data: [{"candidates": [{"content": {"role": "model", "parts": [{"text": "..."}]}}]}]\n\n
data: [{"candidates": [{"content": {"role": "model", "parts": []}, "finishReason": "STOP"}]}]\n\n
```

实现要点：
1. **每帧包成单元素数组**：`json.dumps([frame])`
2. **前缀 `data: ` + 空格**
3. **后缀 `\n\n`**（SSE 事件分隔符）
4. **不再发 `data: [DONE]`**——`@google/genai` 不需要，由服务端关闭连接触发 EOF

### 4.2 ⚠️ 已过时：v0.1.0 错误的"裸数组流"模式

v0.1.0 初稿曾设计输出裸 JSON 数组：

```
[\n
{...},\n
{...}\n
]\n
```

**这个格式是错的**——SDK 会报 `Incomplete JSON segment at the end`，因为：
- SDK 把 `data: [{...}]` 的 `]` 当作 segment 结束符
- 我们流末尾的 `]` 让 SDK 认为 segment 未闭合
- 此外 Content-Type 错误为 `application/json`，SDK 不会按 SSE 解析

> ⚠️ 已在 v0.1.0-r1 修复。修复经验沉淀至 `.learnings/experience/stream-sse-format-fix.md`。

---

## 附录 A: SSE 格式的关键差异

| 维度 | OpenAI（出站） | Gemini SDK 期望（入站） |
|------|----------------|--------------------------|
| Content-Type | `text/event-stream` | `text/event-stream` |
| 帧格式 | `data: {...}\n\n`（单对象） | `data: [{...}]\n\n`（**数组**包单对象） |
| 结束信号 | `data: [DONE]\n\n` | 服务端关闭连接（不发 DONE） |
| 解析器期望 | 解析 `data:` 后为单对象 | 解析 `data:` 后为数组后取 `[0]` |

**为什么 Gemini 用数组包对象**？

- Google 原始 SDK 设计是为了支持 `streamGenerateContent?alt=sse` 在某些版本下会**批量合并**多个 Gemini 响应进同一个 SSE 事件
- 即使只发 1 个响应，data: 后也必须是数组以保持协议一致性

## 5. 字段映射（流式版）

每个 OpenAI `delta` 转 Gemini 帧：

### 5.1 文本增量

OpenAI（输入）：
```json
{"choices":[{"delta":{"content":"Hello"}}]}
```

Gemini（输出）：
```json
{
  "candidates": [{
    "content": {"role": "model", "parts": [{"text": "Hello"}]},
    "index": 0
  }]
}
```

### 5.2 Tool calls 增量（多 chunk 拼装）

OpenAI 流的 `delta.tool_calls` 是**分片结构**，例：

```json
// chunk 1
{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_abc","function":{"name":"get_","arguments":""}}]}}]}

// chunk 2
{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"weather","arguments":"{\"loc"}}]}}]}

// chunk 3
{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ation\":\"SF\"}"}}]}}]}

// chunk N (末尾)
{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}
```

聚合逻辑：
1. 用 `delta.tool_calls[].index` 作为键。
2. 累加 `id`、`function.name`（字符串拼接）、`function.arguments`（字符串拼接）。
3. **不在流式过程中下发**，仅缓存。
4. 收到 `finish_reason` 为终止信号时，**一次性**下发完整的 `functionCall`。

输出：
```json
{
  "candidates": [{
    "content": {
      "role": "model",
      "parts": [
        {"functionCall": {"name": "get_weather", "args": {"location": "SF"}}}
      ]
    },
    "index": 0
  }]
}
```

### 5.3 `finishReason` 与 `usageMetadata` 收尾帧

```json
{
  "candidates": [{
    "content": {"role": "model", "parts": []},
    "finishReason": "STOP",
    "index": 0
  }],
  "usageMetadata": {
    "promptTokenCount": 12,
    "candidatesTokenCount": 87,
    "totalTokenCount": 99
  }
}
```

注意：`usageMetadata` 通常在 OpenAI 流的最末一个 chunk 才下发，且 chunk 形如 `{"usage": {...}, "choices": []}`。**必须**捕获这种"无 choices"的尾部 chunk。

## 6. 错误处理

### 6.1 上游连接错误

捕获 `httpx.RequestError` / `HTTPStatusError`：
- 立即停止迭代；
- 已下发的部分仍可能到达客户端（HTTP 协议语义无法回收）；
- **记录 ERROR 日志**，包含 request id 与 model。

### 6.2 上游 4xx / 5xx

若 `client.stream()` 上下文抛出 `HTTPStatusError`，**转换为 Gemini error 帧**作为最后一帧下发：
```json
{
  "error": {
    "code": <status>,
    "message": "<upstream reason>",
    "status": "<gemini status enum>"
  }
}
```
并在**关闭**前补 `]`（数组流模式）。

### 6.3 JSON 解析失败

跳过当前 chunk，记录 WARN，绝不中断流。

### 6.4 客户端断开

FastAPI 检测到 `asyncio.CancelledError`：
- 立刻 `break` 迭代；
- HTTPX 流自动因 `async with` 上下文结束而关闭；
- 日志记录 `Client disconnected`。

## 7. 性能要求

| 指标 | 目标 |
|------|------|
| **首字节延迟 (TTFB)** | ≤ 50ms（不含上游连接握手） |
| **帧间隔** | ≤ 10ms（与上游大致一致） |
| **缓冲大小** | 内存峰值 ≤ 32MB / 连接 |
| **并发连接数** | 单进程 ≥ 100 |

实现建议：
- 使用 `httpx.AsyncClient` 单例（lifespan 启动期创建）。
- 不在请求路径上做字符串搜索/正则（开销大）。

## 8. 单测覆盖矩阵（流式）

`tests/test_stream.py` 必须覆盖：

| 用例 | 期望 |
|------|------|
| 纯文本流（多 chunk） | 每 chunk 输出对应 Gemini text 帧，最后一帧含 `finishReason=STOP` |
| 工具调用流（多 chunk 拼装） | `tool_calls` 增量被聚合，**只**在末尾一次性输出 `functionCall` |
| 工具调用 + 文本混排 | 文本流先下发，工具调用在收尾帧输出 |
| 上游发送 `usage` 末帧 | 输出含 `usageMetadata` 的最终帧 |
| 上游中途断开 | 不下发 `]`，避免客户端解析错误 |
| 上游 5xx 错误 | 末尾下发 `error` 帧 + `]` |
| 客户端断开（CancelledError） | 流停止，无堆栈泄漏 |

## 9. 与 transform.py 的协作

| 转换阶段 | 调用方 |
|----------|--------|
| 入口（请求体） | `request_transformer.transform(body, stream=True)` → OpenAI `chat/completions` 请求 |
| 出口（响应体） | **流式不经过 `response_transformer`**——直接由 `StreamProcessor` 转换每帧 |

> ⚠️ **避免重复**：流式路径不应再走 `response_transformer.transform()`，否则会破坏打字机体验。`StreamProcessor` 内部**复用** `response_transformer` 的字段映射函数（如 `map_finish_reason`、`make_text_part`），但**不调用**其 `transform()` 顶层封装。

## Related

- [架构设计](architecture.md) — `StreamProcessor` 在整体中的位置
- [转换引擎详细设计](transformer.md) — 流式路径上复用的字段映射逻辑
- [功能需求文档](../requirements/functional_requirements.md) — 流式需求源头
- [CLAUDE.md](../../CLAUDE.md) — 打字机体验准则
