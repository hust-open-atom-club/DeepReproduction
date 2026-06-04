# DeepReproduction

DeepReproduction 是一个面向 CVE 漏洞复现的多阶段自动化框架。它把复现流程拆成 `knowledge -> build -> poc -> verify` 四个阶段，并通过 LangGraph 编排重试、人工 review 和最终收口。

当前仓库里：

- `knowledge`、`build`、`verify` 提供了独立脚本入口
- `poc` 通过主工作流进入，没有单独的 `scripts/run_poc.py`
- `build`、`poc`、`verify` 的真实执行依赖 Docker

## 目录概览

主要目录如下：

- `app/`
  - 核心实现
  - `orchestrator/` 是 LangGraph 主流程编排
  - `stages/` 是四个阶段的实现
  - `schemas/` 是 YAML/状态模型
  - `tools/` 是 git、docker、patch、抓取、日志解析等工具
- `scripts/`
  - 单阶段运行脚本
- `tests/`
  - pytest 测试
- `Dataset/`
  - knowledge 阶段输出的 CVE 数据集
- `workspaces/`
  - build / poc / verify 的工作区和动态产物
- `reports/`
  - 项目文档和分析报告

## 环境要求

建议环境：

- Python `>=3.10`
- `pdm`
- `git`
- `docker`

其中：

- `knowledge` 需要可访问外部网络和模型服务
- `build`、`poc`、`verify` 需要本机 Docker daemon 可用

## 安装依赖

在 `source/` 目录下执行：

```bash
cd source
pdm install
```

如果需要测试依赖：

```bash
cd source
pdm install -G test
```

## 配置 `.env`

先复制模板：

```bash
cd source
cp .env.example .env
```

当前主要配置项：

- 模型配置
  - `KNOWLEDGE_AGENT_MODEL`
  - `KNOWLEDGE_AGENT_API_KEY`
  - `KNOWLEDGE_AGENT_BASE_URL`
  - `BUILD_AGENT_MODEL`
  - `BUILD_AGENT_API_KEY`
  - `BUILD_AGENT_BASE_URL`
  - `POC_AGENT_MODEL`
  - `POC_AGENT_API_KEY`
  - `POC_AGENT_BASE_URL`
  - `VERIFY_AGENT_MODEL`
  - `VERIFY_AGENT_API_KEY`
  - `VERIFY_AGENT_BASE_URL`
- 运行时配置
  - `MAX_BUILD_RETRY`
  - `MAX_POC_RETRY`
  - `WORKSPACE_ROOT`
  - `KNOWLEDGE_MAX_REFERENCE_DEPTH`
  - `KNOWLEDGE_MAX_FETCH_COUNT`
  - `KNOWLEDGE_FETCH_TIMEOUT_SECONDS`
  - `KNOWLEDGE_ENABLE_LLM_CURATION`
  - `LLM_TIMEOUT_SECONDS`

说明：

- 只有配置了对应 agent 的 API key，该阶段才可调用模型。
- `knowledge` 允许关闭 LLM curation，走启发式整理。
- `build` 和 `poc` 即使模型不可用，也各自带有一定 fallback 逻辑，但效果会下降。

## 主流程说明

主流程为：

```text
knowledge -> build -> poc -> verify
```

阶段职责：

1. `knowledge`
   - 从 OSV、reference、网页、补丁中收集证据
   - 输出 `task.yaml`、`knowledge.yaml`、`patch.diff` 等
2. `build`
   - clone 仓库
   - 读取真实源码中的 build 证据
   - 生成 build plan
   - 在 Docker 中完成实际构建
3. `poc`
   - 基于 knowledge + build 结果生成复现载荷与运行脚本
   - 在 Docker 中执行
   - 产出 `poc_artifact.yaml` 和 `run_verify.yaml`
4. `verify`
   - 复用 build 镜像做 pre / post 两次运行
   - 对比修复前后行为
   - 输出最终 `verify_result.yaml`

## 运行方式

### 1. 单独运行 knowledge

命令：

```bash
cd source
pdm run python scripts/run_knowledge.py CVE-2022-28805 --dataset-root Dataset
```

成功后重点检查：

- `Dataset/CVE-2022-28805/vuln_yaml/task.yaml`
- `Dataset/CVE-2022-28805/vuln_yaml/knowledge.yaml`
- `Dataset/CVE-2022-28805/vuln_yaml/runtime_state.yaml`
- `Dataset/CVE-2022-28805/vuln_data/vuln_diffs/patch.diff`

常见清理方式：

```bash
cd source
rm -rf Dataset/CVE-2022-28805
```

### 2. 单独运行 build

前提：

- 对应 CVE 的 `knowledge.yaml` 已存在

命令：

```bash
cd source
pdm run python scripts/run_build.py CVE-2022-28805 --dataset-root Dataset --workspace-root workspaces
```

成功后重点检查：

- `workspaces/CVE-2022-28805/repo/`
- `workspaces/CVE-2022-28805/artifacts/build/build_context.yaml`
- `workspaces/CVE-2022-28805/artifacts/build/build_plan.yaml`
- `workspaces/CVE-2022-28805/artifacts/build/Dockerfile`
- `workspaces/CVE-2022-28805/artifacts/build/build.sh`
- `workspaces/CVE-2022-28805/artifacts/build/build.log`
- `workspaces/CVE-2022-28805/artifacts/build/build_artifact.yaml`

常见清理方式：

```bash
cd source
rm -rf workspaces/CVE-2022-28805
```

### 3. 单独运行 verify

前提：

- `knowledge` 已完成
- `build` 已完成
- `poc` 已完成
- `workspaces/<CVE>/artifacts/poc/poc_artifact.yaml` 已存在
- `workspaces/<CVE>/artifacts/build/build_artifact.yaml` 已存在

命令：

```bash
cd source
pdm run python scripts/run_verify.py CVE-2022-28805 --dataset-root Dataset --workspace-root workspaces
```

成功后重点检查：

- `workspaces/CVE-2022-28805/artifacts/verify/verify_context.yaml`
- `workspaces/CVE-2022-28805/artifacts/verify/verify_plan.yaml`
- `workspaces/CVE-2022-28805/artifacts/verify/pre_patch.log`
- `workspaces/CVE-2022-28805/artifacts/verify/post_patch.log`
- `workspaces/CVE-2022-28805/artifacts/verify/verify_result.yaml`

说明：

- `run_verify.py` 在 `verdict == success` 时返回 `0`
- 非 `success` 时返回 `1`

### 4. 运行完整 LangGraph 工作流

完整流程入口是：

```bash
cd source
pdm run python app/main.py --task <task-yaml-path> --dataset-root Dataset --workspace-root workspaces
```

最小任务 YAML 需要符合 `TaskModel`，例如：

```yaml
task_id: CVE-2022-28805
cve_id: CVE-2022-28805
cve_url: https://api.osv.dev/v1/vulns/CVE-2022-28805
repo_url:
vulnerable_ref:
fixed_ref:
language:
references: []
reference_details: []
```

说明：

- `knowledge` 会尝试从 OSV 引导和补全 `repo_url`、`vulnerable_ref`、`fixed_ref`、references。
- `thread_id` 不传时，默认使用 `task.task_id`。

如果要显式指定线程号：

```bash
cd source
pdm run python app/main.py --task <task-yaml-path> --dataset-root Dataset --workspace-root workspaces --thread-id demo-thread
```

### 5. 恢复被人工 review 中断的工作流

主流程中 `review` 节点会调用 LangGraph 的 interrupt，等待外部继续。

恢复方式：

```bash
cd source
pdm run python app/main.py --thread-id demo-thread --resume-json '{"action":"retry"}'
```

可用动作：

- `retry`
- `continue`
- `abort`

行为说明：

- `retry`：回到当前 review 对应阶段再执行一次
- `continue`：跳过当前 review，推进到后续阶段
- `abort`：直接收口

## 阶段产物

### knowledge 产物

输出目录：

```text
Dataset/<CVE>/
```

关键文件：

- `vuln_yaml/task.yaml`
- `vuln_yaml/knowledge.yaml`
- `vuln_yaml/knowledge_sources.yaml`
- `vuln_yaml/runtime_state.yaml`
- `vuln_data/vuln_diffs/patch.diff`
- `vuln_data/knowledge_sources/raw/`
- `vuln_data/knowledge_sources/cleaned/`
- `vuln_data/knowledge_sources/extracted/`
- `vuln_data/vuln_pocs/`

### build 产物

输出目录：

```text
workspaces/<CVE>/artifacts/build/
```

关键文件：

- `build_context.yaml`
- `build_plan.yaml`
- `Dockerfile`
- `build.sh`
- `build.log`
- `build_artifact.yaml`
- `build_verify.yaml`
- `llm/`

### poc 产物

输出目录：

```text
workspaces/<CVE>/artifacts/poc/
```

关键文件：

- `poc_context.yaml`
- `poc_plan.yaml`
- `Dockerfile`
- `run.sh`
- `poc.log`
- `crash_report.txt`
- `poc_artifact.yaml`
- `run_verify.yaml`
- `payloads/`
- `inputs/`
- `llm/`

### verify 产物

输出目录：

```text
workspaces/<CVE>/artifacts/verify/
```

关键文件：

- `verify_context.yaml`
- `verify_plan.yaml`
- `Dockerfile`
- `verify_run.sh`
- `patch.diff`
- `pre_patch.log`
- `post_patch.log`
- `verify_result.yaml`

## 测试方式

项目测试使用 `pytest`。

### 1. 运行全部测试

```bash
cd source
pdm run pytest
```

### 2. 运行安静模式

```bash
cd source
pdm run pytest -q
```

### 3. 只跑某个测试文件

例如：

```bash
cd source
pdm run pytest tests/test_build_stage.py
```

### 4. 只跑某类测试

例如：

```bash
cd source
pdm run pytest tests/test_knowledge_refs.py
pdm run pytest tests/test_poc_stage.py
pdm run pytest tests/test_verify_stage.py
```

当前测试文件包括：

- `tests/test_build_stage.py`
- `tests/test_build_verify.py`
- `tests/test_docker_tools.py`
- `tests/test_graph_workflow.py`
- `tests/test_knowledge_refs.py`
- `tests/test_patch_tools.py`
- `tests/test_poc_stage.py`
- `tests/test_router.py`
- `tests/test_run_verify.py`
- `tests/test_schemas.py`
- `tests/test_verifier.py`
- `tests/test_verify_stage.py`

这些测试主要覆盖：

- schema
- route / graph 行为
- knowledge reference 处理
- build fallback / build retry / build plan
- poc plan / poc retry / `eligible_for_verify`
- verify 判定逻辑
- docker tool 的命令拼装

## 常见检查点

### knowledge 成功检查

- 命令成功退出
- `runtime_state.yaml` 中 `final_status: success`
- `knowledge.yaml` 的 `summary` 非空
- `task.yaml` 中 `repo_url`、`vulnerable_ref`、`fixed_ref` 尽可能被补全
- `patch.diff` 非空

### build 成功检查

- `build_artifact.yaml` 中 `build_success: true`
- `build.log` 中镜像构建和容器运行成功
- `build_plan.yaml` 中存在有效 `build_commands`

### poc 成功检查

- `poc_artifact.yaml` 已生成
- `poc.log` 中可定位 target 行为
- `run_verify.yaml` 中 `eligible_for_verify` 与日志表现一致

### verify 成功检查

- `verify_result.yaml` 已生成
- `verdict` 为 `success`、`failed` 或 `inconclusive`
- pre / post 日志与最终判定相互一致

## 注意事项

- `build`、`poc`、`verify` 都会写入 `workspaces/`，调试时建议一条 CVE 一个独立工作区。
- `verify` 会在容器执行脚本里进行仓库 reset 和 patch apply，默认假设 workspace 内仓库是可重置的临时副本。
- 路由层当前对 `build` 和 `poc` 的自动重试上限是 2 次；即使 `.env` 里有同名配置，也应以当前实现为准，除非后续代码改动已打通。
- 独立 `verify` 脚本不会替你生成 `poc_artifact.yaml`，必须先通过主流程或内部调用完成 `poc` 阶段。

## 推荐调试顺序

建议按下面顺序排查：

1. 先跑 `knowledge`
2. 确认 `knowledge.yaml` 与 `patch.diff`
3. 再跑 `build`
4. 看 `build.log` 和 `build_artifact.yaml`
5. 再进入 `poc`
6. 最后看 `verify_result.yaml`

如果一个 CVE 的复现质量异常，优先检查：

- `knowledge` 是否抓到了有效 patch 和 reproducer 线索
- `build_plan.yaml` 是否选对了 ref 与 build system
- `run_verify.yaml` 是否把噪声执行误判成 eligible
