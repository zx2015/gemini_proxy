# 转换引擎详细设计 (Transformer)

> 适用版本：v0.1.0
> 最后更新：2026-07-02
>
> 修订记录：
> - 2026-07-02：v0.1.0 初稿
> - 2026-07-02：v0.1.0-r1 §1.2 字段映射表删除 `MODEL_MAPPING` 行；新增 §1.9 强制模型覆盖实现细节
>
> 本文档是 `app/services/transformer/` 下的实现规范。代码编写时必须以本文档为准；任何偏离都需要**先更新文档**再实现。

## 0. 模块布局

```
app/services/transformer/
├── __init__.py
├── to_openai.py        # Gemini 请求 → OpenAI 请求（RequestTransformer）
├── from_openai.py      # OpenAI 响应 → Gemini 响应（ResponseTransformer）
└── fields.py           # 字段映射表、枚举常量、边缘情况工具函数
```

每个模块对外**只暴露一个类实例**（单例）：

```python
# to_openai.py
request_transformer = RequestTransformer()

# from_openai.py
response_transformer = ResponseTransformer()
```

## 1. Gemini 请求 → OpenAI 请求

### 1.1 输入 / 输出契约

**输入**：FastAPI 解析后的 `dict[str, Any]`（即 Gemini `generateContent` 请求体）。
**输出**：可直接 JSON 序列化的 `dict[str, Any]`（OpenAI `chat/completions` 请求体）。

### 1.2 字段映射表

| Gemini 字段 | OpenAI 字段 | 转换规则 |
|-------------|-------------|----------|
| URL 路径 `{model}` | `model` | **强制覆盖**：忽略 URL 中的 `{model}`，**始终**使用 `settings.upstream_model`（详见 [§1.9](#19-强制上游模型覆盖)） |
| `contents` | `messages` | 见 [§1.3](#13-contents--messages) |
| `systemInstruction.parts[].text` | `messages[0]`（role=`system`） | 拼接到 `system` 字段；若 `contents` 第一条已是 `system` 角色，则合并 |
| `generationConfig.temperature` | `temperature` | 直接映射 |
| `generationConfig.maxOutputTokens` | `max_tokens` | 直接映射 |
| `generationConfig.topP` | `top_p` | 直接映射 |
| `generationConfig.topK` | — | **丢弃**（OpenAI 无对应字段），记录到 debug 日志 |
| `generationConfig.stopSequences` | `stop` | 数组 / 字符串直接透传 |
| `generationConfig.candidateCount` | `n` | 直接映射 |
| `generationConfig.responseMimeType` | `response_format.type` | `application/json` → `json`；`text/plain` → `text` |
| `safetySettings` | — | 丢弃（OpenAI 无对应安全分类器） |
| `tools[].functionDeclarations` | `tools[].function` | 见 [§1.4](#14-tools--functiondeclarations--toolsfunction) |
| `toolConfig.functionCallingConfig.mode` | `tool_choice` | 见 [§1.5](#15-toolconfig--tool_choice) |
| `cachedContent` | `extra_body.google.cached_content` | 通过 OpenAI `extra_body` 透传（实验性） |

### 1.3 `contents` → `messages`

`contents` 是 Gemini 的多轮对话数组，每项形如：
```json
{ "role": "user" | "model" | "function", "parts": [...] }
```

转换规则：

1. **遍历 `contents`** 生成 `messages`：
   - `role="user"` → `role="user"`，`parts` 转 `content`（见下）
   - `role="model"` → `role="assistant"`，`parts` 转 `content`
   - `role="function"` → `role="tool"`，见 [§1.6](#16-function-response--role--tool)
2. **`parts` → `content`**：
   - **纯文本**（只有 `parts[i].text`）：取 `parts[0].text` 作为字符串 content。
   - **多模态**（含 `inline_data`）：转换为 OpenAI 的多元素 content 数组：
     ```json
     [
       {"type": "text", "text": "..."},
       {"type": "image_url", "image_url": {"url": "data:<mime>;base64,<data>"}}
     ]
     ```
3. **空 parts**：直接跳过该项，不写空消息（避免触发 OpenAI "messages must be non-empty" 校验失败）。

### 1.4 `tools` — `functionDeclarations` → `tools.function`

Gemini：
```json
{
  "tools": [
    {
      "functionDeclarations": [
        {
          "name": "get_weather",
          "description": "...",
          "parametersJsonSchema": { ... }
        }
      ]
    }
  ]
}
```

OpenAI：
```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "...",
        "parameters": { ... }
      }
    }
  ]
}
```

转换要点：
1. **展平** `tools[].functionDeclarations[]`：一个 Gemini `tool` 内可能含多个声明，转换后变成多个 OpenAI `tools`。
2. **字段重命名**：`parametersJsonSchema` → `parameters`。
3. **多 tool 合并**：Gemini 允许多个 `tools[]`，每个内嵌 `functionDeclarations`。转换后所有 OpenAI `tools` 放进同一个数组。

### 1.5 `toolConfig` → `tool_choice`

Gemini `toolConfig.functionCallingConfig.mode` 取值：

| Gemini | OpenAI `tool_choice` |
|--------|----------------------|
| `AUTO`（默认） | `"auto"` |
| `ANY` | `"required"` |
| `NONE` | `"none"` |

`allowedFunctionNames` → `tool_choice` 不直接对应；Gemini 允许指定"只许调用这 N 个函数"，OpenAI **无内建等价物**。**当前版本**：尝试通过 `tool_choice={"type":"function","function":{"name":...}}` 透传单函数限制；多函数限制**丢弃**。

### 1.6 `function` Response → `role="tool"`

Gemini 函数调用回传形如：
```json
{
  "role": "function",
  "parts": [{
    "functionResponse": {
      "name": "get_weather",
      "response": { ... }
    }
  }]
}
```

转换：
```json
{
  "role": "tool",
  "tool_call_id": "<由 functionCall.id 提供>",
  "content": "{\"name\":\"get_weather\",\"content\":...}"
}
```

**强约束**：`functionResponse` 必须紧跟 `functionCall` 之后，且 `tool_call_id` 必须从前一条 `assistant.tool_calls` 中取到——因此**转换器必须维护一个临时的"上一条 assistant 的 tool_calls id 映射表"**。

### 1.7 完整伪代码

```python
def transform_to_openai(gemini_req: dict) -> dict:
    # model 强制覆盖：忽略入站任何位置传入的 model
    openai_req = {
        "model": settings.upstream_model,    # 唯一来源：环境变量
        "messages": [],
        "stream": gemini_req.get("_stream", False),
    }

    # 1. systemInstruction
    if sys := gemini_req.get("systemInstruction"):
        openai_req["messages"].append({
            "role": "system",
            "content": extract_text(sys.get("parts", [])),
        })

    # 2. contents
    last_tool_call_id = None
    for item in gemini_req.get("contents", []):
        msg = map_content_item(item, last_tool_call_id_ref=[...])
        if msg: openai_req["messages"].append(msg)

    # 3. generationConfig
    gen = gemini_req.get("generationConfig", {})
    if "temperature" in gen: openai_req["temperature"] = gen["temperature"]
    if "maxOutputTokens" in gen: openai_req["max_tokens"] = gen["maxOutputTokens"]
    if "topP" in gen: openai_req["top_p"] = gen["topP"]
    if "stopSequences" in gen: openai_req["stop"] = gen["stopSequences"]
    if "candidateCount" in gen: openai_req["n"] = gen["candidateCount"]
    if "responseMimeType" in gen:
        openai_req["response_format"] = {"type": map_response_type(gen["responseMimeType"])}

    # 4. tools
    if tools := gemini_req.get("tools"):
        openai_req["tools"] = flatten_function_declarations(tools)
        if tool_cfg := gemini_req.get("toolConfig"):
            openai_req["tool_choice"] = map_tool_choice(tool_cfg)

    return openai_req
```

### 1.8 边缘情况与降级策略

| 情况 | 处理 |
|------|------|
| Gemini 请求含 `topK` | 丢弃，记录到 debug 日志 |
| `contents` 为空 | 视为非法请求 → 422 |
| `parts` 全是 `functionResponse` | 与前一条 `functionCall` 配对，否则丢弃 |
| `inline_data` 体积过大（> 20MB） | 透传不截断；但记录 WARN 日志 |
| `systemInstruction` 缺失但 `contents[0].role="user"` | 不强制注入 system |
| 客户端传入任意 model 名（含不存在 / 拼写错误） | **忽略**，强制使用 `settings.upstream_model`（见 [§1.9](#19-强制上游模型覆盖)） |

### 1.9 强制上游模型覆盖

v0.1.0-r1 新增。

#### 1.9.1 规则

```text
入站 model (URL 路径或请求体) ──丢弃──┐
                                        ├─→ openai_req["model"] = settings.upstream_model
入站 model 不校验、不警告、不重定向──┘
```

- `RequestTransformer` **不**读取 `gemini_req` 任何位置的 `model`。
- `RequestTransformer` **不**维护 `MODEL_MAPPING` 字典。
- `settings.upstream_model` 是**唯一**出站 `model` 字段来源。

#### 1.9.2 日志约定

每次请求（无论非流 / 流）记录一条 INFO 日志：

```text
[inbound] model=<客户端传入值，字符串截断到 64 字符> → upstream_model=<settings.upstream_model>
```

- 仅记录 inbound 字符串**前 64 字符**（避免日志膨胀或泄漏 PII）。
- 若客户端未传 `model`（如 `gemini-cli` 走默认），记录 `model=<none>`。

#### 1.9.3 单元测试用例

`tests/test_transformer.py` 新增 / 修改以下用例：

| 用例 | 期望 |
|------|------|
| 客户端传入 `gemini-2.5-pro` | 转换后 `openai_req["model"] == settings.upstream_model`（**不**等于 `gemini-2.5-pro`） |
| 客户端传入 `claude-3-5-sonnet`（甚至非 Gemini 名） | 同上——仍然等于 `settings.upstream_model` |
| 客户端未传 `model` | 同上 |
| 客户端传入空字符串 `""` | 同上 |
| 客户端传入超长字符串（> 1KB） | 同上（不抛异常） |
| 错误：环境变量 `UPSTREAM_MODEL` 未设置 | 网关**启动失败**（Pydantic 校验失败），**不**在请求期才报错 |

## 2. OpenAI 响应 → Gemini 响应

## 2. OpenAI 响应 → Gemini 响应

### 2.1 输入 / 输出契约

**输入**：OpenAI `chat/completions` 响应 JSON `dict[str, Any]`。
**输出**：Gemini `generateContent` 响应 JSON `dict[str, Any]`。

### 2.2 字段映射表

| OpenAI 字段 | Gemini 字段 | 规则 |
|-------------|-------------|------|
| `id` | （丢弃） | Gemini 不需要统一 id |
| `model` | （丢弃，必要时写入 candidate `model`） | 当前丢弃 |
| `choices[0].message.role="assistant"` | `candidates[0].content.role="model"` | **强制转换** |
| `choices[0].message.content` | `candidates[0].content.parts[].text` | 字符串或数组均展平为多 text part |
| `choices[0].message.tool_calls` | `candidates[0].content.parts[].functionCall` | 见 [§2.3](#23-tool_calls--functioncall) |
| `choices[0].finish_reason` | `candidates[0].finishReason` | 见 [§2.4](#24-finish_reason--finishreason) |
| `usage.prompt_tokens` | `usageMetadata.promptTokenCount` | 字段重命名 |
| `usage.completion_tokens` | `usageMetadata.candidatesTokenCount` | 字段重命名 |
| `usage.total_tokens` | `usageMetadata.totalTokenCount` | 字段重命名 |

### 2.3 `tool_calls` → `functionCall`

OpenAI：
```json
{
  "id": "call_abc",
  "type": "function",
  "function": {
    "name": "get_weather",
    "arguments": "{\"location\":\"SF\"}"
  }
}
```

Gemini：
```json
{
  "functionCall": {
    "name": "get_weather",
    "args": {"location": "SF"}
  }
}
```

转换要点：
1. **`arguments` 是 JSON 字符串** —— 必须 `json.loads`；解析失败 → 记录 ERROR 并将原始字符串作为 `args`。
2. **`id` 丢弃**，但需要内部存表，以便后续 `functionResponse` 能回填 `tool_call_id`（仅在多轮对话中需要）。
3. **多个 tool_calls** → 多个 `parts[].functionCall`。

### 2.4 `finish_reason` → `finishReason`

| OpenAI `finish_reason` | Gemini `finishReason` | 备注 |
|------------------------|------------------------|------|
| `stop` | `STOP` | 标准完成 |
| `length` | `MAX_TOKENS` | 输出截断 |
| `tool_calls` | `STOP` | 工具调用也算正常结束（**不含** `functionCall` 标志） |
| `function_call` | `STOP` | 同上（旧版 OpenAI 字段） |
| `content_filter` | `SAFETY` | 安全过滤 |
| （其他 / 缺失） | `OTHER` | 兜底 |

**强约束**：当 `tool_calls` 非空时，`finishReason` 仍为 `STOP`——由 `parts[].functionCall` 的存在来表达"是工具调用"。

### 2.5 完整伪代码

```python
def transform_from_openai(openai_resp: dict) -> dict:
    choice = openai_resp["choices"][0]
    msg = choice["message"]

    parts = []
    if msg.get("content"):
        parts.append({"text": msg["content"]})
    for tc in msg.get("tool_calls", []):
        try:
            args = json.loads(tc["function"]["arguments"])
        except Exception:
            args = tc["function"]["arguments"]
        parts.append({"functionCall": {"name": tc["function"]["name"], "args": args}})

    gemini_resp = {
        "candidates": [{
            "content": {"role": "model", "parts": parts},
            "finishReason": map_finish_reason(choice.get("finish_reason")),
            "index": 0,
        }],
    }

    if usage := openai_resp.get("usage"):
        gemini_resp["usageMetadata"] = {
            "promptTokenCount": usage.get("prompt_tokens", 0),
            "candidatesTokenCount": usage.get("completion_tokens", 0),
            "totalTokenCount": usage.get("total_tokens", 0),
        }

    return gemini_resp
```

### 2.6 边缘情况与降级策略

| 情况 | 处理 |
|------|------|
| OpenAI 返回 `choices` 为空 | 返回 `{"candidates": [], "error": {...}}` |
| `finish_reason` 缺失 | 默认 `STOP`，仅当 `usage` 异常时降级 `OTHER` |
| `tool_calls[].function.arguments` 不是合法 JSON | 用原始字符串作为 `args`；记录 ERROR |
| 上游无 `usage` 字段 | 返回的 Gemini 响应中不包含 `usageMetadata` |
| 多 `choices`（`n > 1`） | 仅返回 `choices[0]`，其余丢弃（Gemini 无内建"多候选"语义），记录 WARN |

## 3. 错误归一化（响应方）

上游可能返回非 2xx（如 400 / 401 / 429 / 500），统一转换为 Gemini 错误对象：

```json
{
  "error": {
    "code": <http_status>,
    "message": "<reason or extracted detail>",
    "status": "<gemini_status_enum>"
  }
}
```

`status` 映射（参考 [GCP API errors](https://cloud.google.com/apis/design/errors)）：

| HTTP | Gemini `status` |
|------|-----------------|
| 400 | `INVALID_ARGUMENT` |
| 401 | `UNAUTHENTICATED` |
| 403 | `PERMISSION_DENIED` |
| 404 | `NOT_FOUND` |
| 429 | `RESOURCE_EXHAUSTED` |
| 500 / 502 / 503 / 504 | `INTERNAL` |
| 其他 | `UNKNOWN` |

## 4. 单测覆盖矩阵

转换器单元测试（`tests/test_transformer.py`）**必须**覆盖：

| 用例 | 期望 |
|------|------|
| 单轮纯文本（user） | 正确生成单条 user 消息 |
| 单轮纯文本（含 `systemInstruction`） | 生成 system + user |
| 多轮对话（含 model 回复） | model → assistant 转换 |
| 单模态图片请求 | inline_data → image_url data URL |
| 函数声明（functionDeclarations） | 展平为 OpenAI tools |
| 函数调用回传（functionResponse） | role="tool" + tool_call_id |
| `toolConfig.mode="ANY"` | tool_choice="required" |
| `generationConfig.responseMimeType="application/json"` | response_format.type="json" |
| OpenAI 响应含 `tool_calls` | parts 含 `functionCall` |
| `finish_reason="tool_calls"` | finishReason="STOP" + 含 functionCall |
| `usage` 字段缺失 | 不输出 usageMetadata |
| 上游 429 错误 | Gemini error 对象 `RESOURCE_EXHAUSTED` |

## Related

- [架构设计](architecture.md) — `Transformer` 在整体中的位置
- [流处理详细设计](stream_handler.md) — 流式场景下的转换差异
- [功能需求文档](../requirements/functional_requirements.md) — 字段映射表的需求源头
- [CLAUDE.md](../../CLAUDE.md) — 协议严谨性准则
