<div align="center">

# Focus

**一个受 Codex 启发的、证据驱动的 Rust 编码代理，并自带可审计的主机测试框架。**

[English](README.md) | [中文](README.zh-CN.md)

![CI](https://img.shields.io/github/actions/workflow/status/XXXXXQ-0206/focus/ci.yml?branch=dev&label=CI)
![Rust](https://img.shields.io/badge/rust-1.97.1-blue)
![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)

</div>

Focus 是一个用 Rust 编写的编码代理。它把一个小而自有的代理内核，与一整套
运行时（会话、记忆、策略、工具、Provider 适配器、可审计主机）组合在一起。
它被设计为**证据驱动**：产品的各项能力都与持久化的会话事件、可重放的音频
记录（transcript）、确定性的消融（ablation）凭证，以及明确的发布门禁绑定。

主命令是 `focus`，同时保留 `focus-harness` 作为兼容别名，用于运行、检查与
验证产品。

---

## 目录

- [项目概览](#项目概览)
- [功能特性](#功能特性)
- [界面预览](#界面预览)
- [安装](#安装)
- [使用](#使用)
- [构建说明](#构建说明)
- [项目结构](#项目结构)
- [路线图](#路线图)
- [贡献](#贡献)
- [许可证](#许可证)
- [常见问题](#常见问题)
- [致谢](#致谢)
- [免责声明](#免责声明)

---

## 项目概览

编码代理是产品本体；主机测试框架则是它轻量的 CLI、诊断表面与验收主机。
`FocusRuntime` 与 `focus-kernel` 属于内部产品组件，而非彼此对立的“产品身份”。

Focus 自有的内核，其循环语义源自钉住的原始 MIT Pi 源码。Codex 启发了产品体验
与工程流程；Pi 则提供了最小代理循环的语义基线。

Focus 围绕**单条异步执行主干**构建：Provider 以规范化事件流式输出，工具与
Runtime 处理器暴露同一个异步调用契约，阻塞式的 MCP stdio 则留在私有 worker 中。
同步 API 只出现在明确的主机边界处，并且会拒绝嵌套的 Tokio 调用。

## 功能特性

- **交互式 TUI**——Codex 风格的全屏界面，包含紧凑的音频记录、增量模型文本、
  折叠的工具活动、底部输入框、审批对话框，以及斜杠命令
  （`/permissions`、`/help`、`/new`、`/clear`、`/status`、`/details`、
  `/review`、`/simplify`、`/update`、`/exit`）。
- **有界的有序工具**——默认最多并行运行 4 个独立工具调用，结果按模型调用顺序
  提交；写操作、shell 命令、工作流检查点、委派与 MCP 调用都是显式的互斥屏障。
- **同一条策略/工具/循环路径**——启用的内置工具、MCP 工具与委派都由同一个
  `PolicyEngine` 包装，并由同一条 MIT 语义的 `AgentLoop` 调度。
- **真实 MCP 传输**——持久化 stdio JSON-RPC 连接执行 `initialize`、
  `notifications/initialized`、分页的 `tools/list` 与可取消的 `tools/call`。
  发现的工具会进入规范的工具注册表。
- **两种 Provider 适配器**——兼容 OpenAI 的 HTTP 端点（Chat Completions 或
  Responses 线协议），或通过 stdio 交换一条规范化
  `ModelRequest`/`ModelResponse` JSON 文档的命令适配器。
- **一等公民子代理**——类型化的任务、结果、限额、单条异步管理器、隔离的子会话、
  协作式取消，以及有界的有序并行聚合，并通过规范的 `delegate` 工具暴露给模型。
- **有界的命令执行**——原生、Docker 与 Podman 后端共享超时、有界的
  stdout/stderr 捕获、协作式取消，以及进程/容器清理。容器后端会禁用网络，
  并接受内存、CPU 与 PID 限额。
- **网络策略**——网络默认禁用并采用白名单；支持精确主机、通配符与全局
  allow/deny 规则；URL 的查询串、片段与内嵌凭证在进入音频记录或模型上下文前
  都会被脱敏。
- **Codex 风格权限**——默认 YOLO 模式；可用 `--interactive` 或 `/permissions`
  切换为逐操作审批，或用 `--deny` 切换为只读模式。
- **无缝自更新**——`focus self-review`、`self-update stage` 与
  `self-update apply` 让你在不替换正在运行的可执行文件的情况下审查并暂存新版本，
  并带有原子化的 manifest 与 `previous` 回滚槽位。
- **证据与消融**——持久化会话事件、可重放的音频记录、确定性的消融凭证，以及
  可复现的发布合规门禁。

## 界面预览

Focus 是一个终端应用。它的主要界面是一套交互式 TUI，采用 cell-diff 终端后端
渲染，因此模型流式输出时只更新发生变化的单元格，而不是整屏重绘。典型布局如下：

```text
┌ Focus ─────────────────────────────────────────────────────────────┐
│ Session: worker-01    Mode: yolo    ● connected                    │
├────────────────────────────────────────────────────────────────────┤
│ You: inspect this repository and fix the failing test.             │
│ Focus:                                                            │
│   I inspected the workspace and ran the failing test. The issue is  │
│   a stale assertion in ...                                          │
│   ▸ tool: read crates/focus-kernel/src/lib.rs (0.6s)                │
│   ▸ tool: shell cargo test --workspace (4.1s) [completed]           │
├────────────────────────────────────────────────────────────────────┤
│ Thinking ▸ model streaming ▸ ▸ ▸                                   │
├────────────────────────────────────────────────────────────────────┤
│ > review the change_                                           │
└────────────────────────────────────────────────────────────────────┘
```

## 安装

### 前置要求

- 匹配 [`rust-toolchain.toml`](rust-toolchain.toml) 的 Rust 工具链（Rust 1.97.1）。
- `cargo`、`rustfmt` 与 `clippy`。

### 安装 CLI

从一次检出的源码中安装 CLI：

```bash
cargo install --path crates/focus-cli --locked
```

这样会把 `focus` 与兼容别名 `focus-harness` 一并安装到 Cargo 的用户二进制目录。
之后在任何项目目录下运行 `focus` 即可；当前目录会成为工作区，并打开交互式 TUI。

### 配置 Provider

请通过进程环境变量配置 Provider，而不是把凭证写进脚本：

```bash
export OPENAI_API_KEY="TOKEN"
export FOCUS_BASE_URL="https://ENDPOINT/v1"
export FOCUS_MODEL="MODEL"
focus
```

在 Windows 上，把 `export` 换成 `$env:OPENAI_API_KEY = "TOKEN"` 即可。

## 使用

### 交互式会话

```bash
focus                     # 在当前目录打开交互式 TUI
focus chat                # 交互式会话的显式形式
```

在 TUI 中：

- `/review`——检查仓库、修复具体问题并验证。
- `/simplify`——在不改变功能、行为与性能的前提下做有度量的低风险简化。
- `/permissions`——切换审批模式。
- `/details`——展开被折叠的工具活动。
- `/update`——把空闲会话交给已验证的自更新产物。
- `Esc`——取消当前活动轮次。

### 单次运行

```bash
focus run --workspace . --provider openai --model MODEL -- "implement the
requested change and run focused checks"
```

### 面向行/CI 的主机

当 stdin/stdout 为管道时，会自动保留面向行的主机。在终端里可用 `--plain`
强制该模式：

```bash
printf 'inspect this repository\n' | focus --plain --workspace .
```

### 结构化主机边界

GUI 与 IDE 主机可以消费版本化的 JSONL 事件流，而不是解析 PTY 文本：

```bash
echo '{"protocol":"focus-host-v1","operation":"attach","session_id":"SESSION_ID"}' \
  | focus host --stdio --workspace .
```

### 会话、目标与记忆

```bash
focus session list --workspace .
focus session replay --workspace . SESSION_ID
focus goal create --workspace . --title "Release" -- "Produce and verify the release artifact"
focus memory add --workspace . --scope project -- "The release uses the native backend"
```

## 构建说明

```bash
# 格式化
cargo fmt --all -- --check

# 静态分析（告警视为错误）
cargo clippy --workspace --all-targets --locked --offline -- -D warnings

# 测试
cargo test --workspace --locked --offline

# 发布构建
cargo build --workspace --release --locked

# 发布合规门禁
cargo run --locked --offline -p focus-release-compliance -- --mode production --workspace .
```

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) 中的 CI 工作流会在 Linux
与 Windows 上运行上述检查。

## 项目结构

| 路径 | 职责 |
| --- | --- |
| [`crates/focus-kernel`](crates/focus-kernel) | Focus 自有的 MIT-Pi 语义内核，负责规范化 provider/tool/event 调度、取消、有序并发工具与 JSONL 事件。 |
| [`crates/focus-runtime`](crates/focus-runtime) | 上下文、会话祖先、记忆、策略、内置工具、Provider 适配器、MCP、工作流门禁、沙箱后端与子代理生命周期。 |
| [`crates/focus-cli`](crates/focus-cli) | 可审计的主机 CLI，负责选择适配器、承载交互式审批，并暴露诊断与验收命令，而不重复代理行为。 |
| [`crates/focus-release-compliance`](crates/focus-release-compliance) | 发布前使用的、可复现的依赖/许可合规门禁。 |

关于所有权与执行流程，请参阅 [`ARCHITECTURE.md`](ARCHITECTURE.md)。

## 路线图

该项目正在积极开发中。我们计划进一步扩展的方向包括：

- 更广泛的 Provider 覆盖与工具包适配器。
- 更强的沙箱隔离与更多后端目标。
- 大规模下的性能与上下文组合调优。
- 更多的验收证据与基准测试框架。
- 更丰富的 MCP 与工作流集成。

## 贡献

欢迎贡献。请先阅读 [`CONTRIBUTING.md`](CONTRIBUTING.md)，并遵守
[行为准则](CODE_OF_CONDUCT.md)。简要流程：

1. Fork 本仓库。
2. 创建特性分支。
3. 使用 Conventional Commit 规范提交改动。
4. 向 `main` 分支发起 Pull Request。

## 许可证

Focus 采用如下双许可之一：

- Apache License, Version 2.0（[`LICENSE`](LICENSE)）；或
- MIT License（[`LICENSE-MIT`](LICENSE-MIT)）。

你可以依据任一许可的条款使用、修改与分发本软件。完整条款见各许可文件。
第三方依赖的许可与归属请参阅 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

## 常见问题

**Focus 是什么？** Focus 是一个编码代理，而不是模型。它包装模型 Provider，
并暴露工具、会话、记忆与可审计的事件流。

**我是否需要 Provider 的 API Key？** 需要。Focus 从环境变量读取凭证，不会把它们
写入仓库、音频记录或回放产物。

**支持哪些模型 Provider？** 任何兼容 OpenAI 的 HTTP 端点（Chat Completions 或
Responses 线协议），或能透过 stdio 使用规范化
`ModelRequest`/`ModelResponse` JSON 协议的命令适配器。

**Focus 可以离线运行吗？** 可以。`--network-domain` 采用白名单且默认禁用网络；
需要对外 Web 访问时显式启用 `web_fetch`，并可在 shell 级隔离时使用无网络容器后端。

**原生沙箱是安全边界吗？** 不是。它只约束工作区路径与进程控制，并非操作系统级
隔离边界。需要进程级网络隔离时，请使用容器后端。

**如何报告 Bug 或漏洞？** Bug 见 [`CONTRIBUTING.md`](CONTRIBUTING.md)，
漏洞报告见 [`SECURITY.md`](SECURITY.md)。

## 致谢

Focus 建立在若干开源项目之上：

- **OpenAI Codex**——用于启发 Focus 产品体验、架构与能力参考。
- **Pi（MIT）**——钉住的最小代理循环语义来源，`focus-kernel` 的循环语义由此派生。
- **PIDEX**——作为编码工作流实现材料的参考。
- **DeepSeek Harness**——作为插件组合、会话流、主机/UI 与沙箱设计参考。
- **Rust 生态**，以及 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) 中列出的各 crate。

以上上游源码仅用于架构研究与带出处审核的移植；除非在 `Cargo.toml` 中显式声明，
否则不会作为运行时依赖捆绑分发。

## 免责声明

> **使用条款与免责声明**
>
> 本软件按 **“原样”（AS IS）** 提供，不附带任何明示或默示的保证，包括但不限于
> 适销性、特定用途适用性、权利与非侵权的保证。无论因合同、侵权或其他原因产生的
> 任何索赔、损害或其他责任，作者或版权持有者均不承担责任。
>
> 本免责声明同样适用于本项目的各类能力：本项目可能自动执行具有真实世界影响的
> 操作，包括 shell 命令执行、文件修改、网络请求，以及向其他软件委派任务。
> **您对自己提供的配置以及允许本软件执行的操作负有全部责任**，并应确保您的使用
> 遵守所有适用的法律、法规与服务条款。
>
> **作者与贡献者对因使用或依赖本软件而导致的任何直接、间接、偶然、特殊或后果性
> 损害、数据丢失、利润损失或业务损失概不负责，即使已被告知此类损害的可能性。**
>
> **禁止非法用途。** 不得使用本软件从事任何违法、有害、欺诈或未经授权的活动，
> 包括但不限于：未经授权访问系统或网络、分发恶意软件、窃取凭证、发送垃圾信息，
> 或任何违反适用法律法规或第三方服务条款的行为。您有责任在使用本软件前取得
> 所需的任何许可或授权，并对未取得授权所产生的后果承担责任。
>
> 通过安装、构建或运行本软件，即表示您已阅读本免责声明并在自愿承担风险的前提下
> 接受其条款。如果您不同意，请勿使用本软件。

---

关于所有权与执行流程，请参阅 [`ARCHITECTURE.md`](ARCHITECTURE.md)；第三方归属
请参阅 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
