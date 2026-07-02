# TODO

## 进行中
（暂无）

## 待办
- [ ] 实现上游失败重试机制（指数退避，3 次尝试，仅非流式） — 优先级：高
- [ ] 实现 `UPSTREAM_MODEL` 强制覆盖（删除 `MODEL_MAPPING`） — 优先级：高
- [ ] 初始化 Python 虚拟环境与依赖清单（fastapi / uvicorn / httpx / pydantic-settings / tenacity） — 优先级：高
- [ ] 搭建项目基础结构与配置管理（`app/core/config.py`） — 优先级：高
- [ ] 实现多模式 API Key 鉴权（`x-goog-api-key` / `Authorization: Bearer` / `?key=`） — 优先级：高
- [ ] 实现 Gemini 非流式路由 `POST /v1beta/models/{model}:generateContent` — 优先级：高
- [ ] 实现 Gemini 流式路由 `POST /v1beta/models/{model}:streamGenerateContent` — 优先级：高
- [ ] 实现请求转换引擎：`generateContent` → `chat/completions`（`app/services/transformer/to_openai.py`） — 优先级：高
- [ ] 实现响应转换引擎：`chat/completions` → `generateContent`（`app/services/transformer/from_openai.py`） — 优先级：高
- [ ] 实现流式状态机：OpenAI SSE chunk → Gemini JSON 数组流（`app/services/stream/processor.py`） — 优先级：高
- [ ] 实现 `finishReason` 映射表（`STOP` / `MAX_TOKENS` / `SAFETY` / `RECITATION` / `OTHER` ↔ `stop` / `length` / `tool_calls` / `content_filter`） — 优先级：中
- [ ] 实现上游错误码 → Gemini 错误对象映射（`app/utils/error_handler.py`） — 优先级：中
- [ ] 编写单元测试（`tests/test_transformer.py`） — 优先级：中
- [ ] 编写 E2E 测试脚本（`tests/test_e2e.py`，mock 上游 OpenAI 兼容服务） — 优先级：中
- [ ] 编写 Dockerfile 与 docker-compose.yml — 优先级：低
- [ ] 建立 `.gitignore`（.venv/、.env、.learnings/ 不入版本控制） — 优先级：高
- [ ] 编写多模态（`inline_data` ↔ `image_url`）映射 — 优先级：中
- [ ] 编写 `functionCalling` 互转逻辑（`tools.functionDeclarations` ↔ `tools.function`） — 优先级：中
- [ ] 编写 `countTokens` 端点处理（转发 / 启发式估算） — 优先级：低
- [ ] 编写 `GET /v1beta/models` 列表端点（由上游 `/v1/models` 派生） — 优先级：中

## 已完成
- [x] 编写 CLAUDE.md（项目入口准则） — 2026-07-02
- [x] 编写功能需求文档 `docs/requirements/functional_requirements.md` — 2026-07-02
- [x] 编写架构设计文档 `docs/design/architecture.md` — 2026-07-02
- [x] 编写转换引擎详细设计 `docs/design/transformer.md` — 2026-07-02
- [x] 编写流处理详细设计 `docs/design/stream_handler.md` — 2026-07-02
- [x] 初始化 `.learnings/` 目录结构与 index.md — 2026-07-02
- [x] 需求变更 v0.1.0-r1：强制 `UPSTREAM_MODEL` 覆盖 + 上游重试机制 — 2026-07-02
- [x] 文档更新 v0.1.0-r1：functional_requirements.md §2.7/§2.8、architecture.md §3.5/§6.4/§6.5、transformer.md §1.2/§1.7/§1.9 — 2026-07-02
- [x] 实现核心代码（v0.1.0 完整功能） — 2026-07-02
  - app/core/{config,auth,logging}.py
  - app/utils/{retry,error_handler}.py
  - app/services/transformer/{fields,to_openai,from_openai}.py
  - app/services/stream/processor.py
  - app/services/discovery.py
  - app/api/gemini.py
  - app/main.py
  - 部署文件：Dockerfile / docker-compose.yml / requirements*.txt / .env.example / .gitignore
- [x] 编写测试（44 个用例全通过） — 2026-07-02
  - tests/test_transformer.py（24 个）
  - tests/test_e2e.py（12 个）
  - tests/test_stream.py（8 个）
- [x] 真实上游 LiteLLM 端到端验证（非流 + 流 + 鉴权） — 2026-07-02
- [x] 修复流式输出 BUG：数组流分隔符、收尾 finishReason 缺失、上游流式错误识别 — 2026-07-02
- [x] 需求变更 v0.1.0-r2：修复 MiniMax/M3 `<think>` 标签泄漏（非流与流式全清洗） — 2026-07-02
- [x] 修复流式响应格式（SSE data: {json}\n\n）通过 gemini-cli 全链验证 — 2026-07-02
- [x] 需求变更 v0.1.0-r3：修复 gemini-cli 工具回传 `role=user` 中 `functionResponse` 被静默丢弃 — 2026-07-02
