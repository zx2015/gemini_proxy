# 设计文档总索引 (Design Index)

> 最后更新：2026-07-02

本目录包含 `gemini_proxy` 项目的所有设计文档。**任何代码修改前必须先更新对应的设计文档**（参见 [CLAUDE.md §7](../../CLAUDE.md)）。

## 目录

### 整体架构
- [architecture.md](architecture.md) — 整体技术栈、双向拓扑、组件职责、数据生命周期、关键设计取舍

### 模块详细设计
- [transformer.md](transformer.md) — `app/services/transformer/` 模块规范：Gemini ↔ OpenAI 字段映射、错误归一化、边缘情况
- [stream_handler.md](stream_handler.md) — `app/services/stream/processor.py` 规范：流式状态机、数组流 vs 分块对象流、性能要求

### 待补充
（架构扩展为双入单出后，回填）：
- adapter.md — `app/adapters/` 上游 OpenAI 兼容服务适配器插件
- auth.md — 多模式鉴权细节
- deployment.md — 容器化 / K8s 部署
- observability.md — 日志 / 指标 / Trace

## 引用层次

```
docs/
├── requirements/
│   └── functional_requirements.md   ← 需求源头
└── design/
    ├── index.md                       ← 本文件
    ├── architecture.md                ← 整体设计
    ├── transformer.md                 ← 请求 / 响应转换引擎
    └── stream_handler.md              ← 流式处理
```

## Related

- [CLAUDE.md](../../CLAUDE.md) — 项目入口准则（强制文档优先）
- [功能需求文档](../requirements/functional_requirements.md) — 字段映射 / 接口约定 / 错误规范的源头
- [TODO.md](../../TODO.md) — 文档落地的待办进度
- [.learnings/](../../.learnings/index.md) — 本地知识沉淀
