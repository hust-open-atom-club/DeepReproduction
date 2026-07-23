<!-- 文件说明：PoC 阶段详细架构设计。用于指导 source/app/stages/poc.py 及其配套 schema、模板、测试的实现。 -->

# PoC 阶段架构设计

## 1. 设计目标

`poc` 阶段的目标不是“让模型随便生成一个样例”，而是基于已有事实生成一个**可执行、可保存、可重试、可被 verify 消费**的最小复现产物。

它必须满足以下约束：

1. 只消费现有输入，不重新做外部网页搜索
2. 不重新 clone 仓库，不重新决定构建方案
3. 真实触发动作必须在 Docker 容器中完成
4. 输出必须结构化，能够直接被 `verify` 阶段复用
5. 失败时要保留完整上下文，支持重规划和重试

## 2. 在主流程中的位置

主流程保持不变：

```text
knowledge -> build -> poc -> verify
```

其中 `poc` 的输入来自：

- `KnowledgeModel`
- `BuildArtifact`
- `workspaces/<CVE>/repo`
- `Dataset/<CVE>/vuln_data/vuln_diffs/patch.diff`

输出到：

- `workspaces/<CVE>/artifacts/poc/`
- `PoCArtifact`

因此它本质上是一个“从知识线索和已确认构建事实中，收敛出最小触发方案并执行”的阶段。

## 3. 职责边界

### 3.1 应该做的事

1. 收集与 PoC 构造相关的本地证据
2. 生成结构化触发计划
3. 产出 PoC 主文件、附属输入和运行脚本
4. 基于 `build` 产物在容器中执行 PoC
5. 收集 stdout/stderr/退出码/崩溃模式
6. 将结果落盘为 `poc_artifact.yaml`

### 3.2 不应该做的事

1. 不重新搜索 CVE 页面、issue、博客
2. 不重新判断 repo 从哪里下载
3. 不重新决定 build system
4. 不做最终漏洞是否修复的裁决
5. 不直接修改 `artifacts/build/` 下的已有产物

## 4. 总体分层

建议 PoC 阶段完全对齐 `build` 阶段的结构，分成 5 层：

```text
PocStage
├── 1) build_plan()           -> 生成静态目录与执行元数据
├── 2) collect_poc_context()  -> 收集本地证据，形成 PocContext
├── 3) plan_poc()             -> 生成结构化 PocPlan
├── 4) execute_poc_plan()     -> 渲染文件并在 Docker 中执行
└── 5) persist_artifacts()    -> 落盘日志、yaml、PoC 文件
```

这套分层的好处是：

- 和 `BuildStage` 一致，代码风格统一
- 失败点清晰，便于重试
- 每一层都可以单测
- 后续接 LLM 规划器时，不会把文件系统和容器逻辑混在一起

## 5. 目录与产物设计

建议 `poc` 阶段拥有以下目录：

```text
workspaces/<CVE>/artifacts/poc/
├── poc_context.yaml
├── poc_plan.yaml
├── Dockerfile
├── run.sh
├── poc.log
├── crash_report.txt
├── poc_artifact.yaml
├── payloads/
│   ├── poc.<ext>
│   └── ...
└── inputs/
    ├── input-1
    └── ...
```

说明：

- `poc_context.yaml`：本地证据汇总，供规划和排障使用
- `poc_plan.yaml`：规划阶段的结构化决策结果
- `Dockerfile`：PoC 执行镜像定义，可复用 build 镜像或做薄封装
- `run.sh`：容器内实际执行入口
- `poc.log`：容器执行日志
- `crash_report.txt`：从 stderr/log 中提炼出的关键失败信息
- `payloads/`：PoC 主文件及辅助脚本
- `inputs/`：外部输入数据，如二进制样本、文本样本、配置文件

## 6. 核心数据流

建议的数据流如下：

```text
KnowledgeModel + BuildArtifact + repo + patch.diff
        │
        ▼
collect_poc_context()
        │
        ▼
PocContext
        │
        ▼
plan_poc()
        │
        ▼
PocPlan
        │
        ▼
execute_poc_plan()
        │
        ├── payloads/*
        ├── inputs/*
        ├── run.sh
        ├── Dockerfile
        └── poc.log / crash_report.txt
        ▼
PoCArtifact
```

## 7. 推荐的数据模型

### 7.1 新增 `PocStagePaths`

建议在 `source/app/stages/poc.py` 中增加：

- `workspace_root`
- `repo_dir`
- `artifacts_dir`
- `build_dir`
- `poc_dir`
- `payloads_dir`
- `inputs_dir`
- `poc_context_yaml`
- `poc_plan_yaml`
- `dockerfile`
- `run_script`
- `poc_log`
- `crash_report`
- `poc_artifact_yaml`

这样可以和 `BuildStagePaths` 保持一致。

### 7.2 新增 `PocContext`

`PocContext` 建议表达“PoC 规划所依赖的全部本地事实”，至少包含：

- `cve_id`
- `repo_url`
- `resolved_ref`
- `repo_local_path`
- `target_binary`
- `build_system`
- `build_success`
- `patch_diff_excerpt`
- `patch_affected_files`
- `knowledge_summary`
- `reproduction_hints`
- `expected_error_patterns`
- `expected_stack_keywords`
- `candidate_entrypoints`
- `candidate_trigger_files`
- `candidate_cli_flags`
- `previous_failure_kind`
- `previous_execution_log`
- `planner_attempt`

其中：

- `candidate_entrypoints` 来自 `BuildArtifact.binary_or_entrypoint`、`expected_binary_path`、仓库扫描
- `candidate_trigger_files` 来自 patch 涉及文件、知识中的 reproduction hints
- `candidate_cli_flags` 来自 README、测试文件、已有 PoC 线索

### 7.3 新增 `PocPlan`

`PocPlan` 建议作为规划器唯一输出，至少包含：

- `trigger_mode`
- `target_binary`
- `target_args`
- `environment_variables`
- `payload_filename`
- `payload_content`
- `auxiliary_files`
- `run_command`
- `expected_exit_code`
- `expected_stdout_patterns`
- `expected_stderr_patterns`
- `expected_crash_type`
- `source_of_truth`
- `confidence`
- `rationale`
- `dockerfile_override`
- `run_script_override`

推荐 `trigger_mode` 枚举值：

- `cli-file`
- `cli-stdin`
- `cli-argv`
- `script-driver`
- `library-harness`

### 7.4 扩展 `PoCArtifact`

当前 `PoCArtifact` 字段偏少，建议扩展为：

- `trigger_mode`
- `trigger_command`
- `target_binary`
- `input_file_paths`
- `auxiliary_file_paths`
- `expected_stdout_patterns`
- `expected_stderr_patterns`
- `expected_exit_code`
- `observed_exit_code`
- `observed_stdout`
- `observed_stderr`
- `observed_crash_type`
- `matched_error_patterns`
- `matched_stack_keywords`
- `reproducer_verified`

对现有字段的映射建议：

- 保留 `root_cause_analysis`
- 保留 `payload_generation_strategy`
- 保留 `poc_filename`
- 保留 `poc_content`
- 保留 `run_script_content`
- 保留 `execution_success`
- 保留 `execution_logs`

这样 `verify` 阶段就不必重新解析原始日志。

## 8. 证据收集层设计

`collect_poc_context()` 建议只做“收集事实”，不做裁决。

### 8.1 输入来源

1. `KnowledgeModel`
2. `BuildArtifact`
3. `Dataset/<CVE>/vuln_data/vuln_diffs/patch.diff`
4. `workspaces/<CVE>/repo` 中的源码、README、tests、examples
5. `Dataset/<CVE>/vuln_data/vuln_pocs/` 中已有参考 PoC（如果存在）

### 8.2 重点提取的信息

1. patch 修改的是哪类文件
2. patch 修改了哪些函数、参数检查或边界检查
3. repo 中有哪些与漏洞点相关的测试、示例、命令行入口
4. build 阶段确认的二进制入口是什么
5. 预期错误模式是崩溃、断言失败、非零退出还是错误日志

### 8.3 建议实现方式

优先使用轻量规则收集：

- patch 解析：沿用正则提取 affected files / hunk 片段
- 入口扫描：在 repo 中搜索 `main(`、测试命令、README 示例命令
- 线索搜索：对 `affected_files` 的文件名、函数名做 `rg`
- PoC 参考：扫描 `vuln_data/vuln_pocs/`

这里的目标不是完整语义分析，而是给规划器提供足够多但受控的证据。

## 9. 规划层设计

`plan_poc()` 建议遵循“规则优先，模型补全”的策略。

### 9.1 规则规划优先级

1. 数据集已有参考 PoC，优先转成标准化 `PocPlan`
2. `reproduction_hints` 已明确命令格式，优先直接采用
3. patch 和 README 已能推断文件型输入，生成最小样本
4. 以上都不够时，再让 LLM 根据 `PocContext` 输出结构化计划

### 9.2 规划约束

规划器必须输出“最小可执行方案”，避免大而泛的脚本。建议强制：

1. 只使用一个主触发命令
2. 附属文件数量最小化
3. 默认使用相对路径
4. 执行脚本必须可重跑
5. 不能依赖容器外的宿主机路径

### 9.3 重规划机制

如果第一次执行失败，应允许按失败类型重规划：

- `missing_binary`
- `bad_arguments`
- `payload_invalid`
- `non_triggering`
- `container_runtime`

重规划输入：

- 上一次 `PocPlan`
- 上一次 `poc.log`
- 已观测退出码和 stderr

最多建议 `3` 次尝试，与 `build` 保持相同风格。

## 10. 执行层设计

`execute_poc_plan()` 负责把计划变成可执行产物。

### 10.1 Docker 策略

建议优先复用 `build` 阶段镜像：

1. 若 `BuildArtifact.docker_image_tag` 存在，则基于该镜像生成薄 `Dockerfile`
2. `poc` 镜像只补充运行时依赖，不重复完整构建
3. 容器启动后挂载整个 `workspaces/<CVE>/`，统一以 workspace 为根执行

这样可以避免 `poc` 阶段重复编译。

### 10.2 运行脚本职责

`run.sh` 必须完成：

1. 进入 repo 或 workspace 指定目录
2. 检查目标二进制是否存在
3. 准备输入文件权限
4. 执行 PoC 命令
5. 收集 stdout/stderr/exit code
6. 将关键结果输出到统一日志

### 10.3 日志规范

建议统一输出以下键值，便于后续解析：

```text
target_binary=...
trigger_command=...
execution_exit_code=...
stdout_begin
...
stdout_end
stderr_begin
...
stderr_end
```

这样后续 `verify` 阶段无需依赖脆弱的纯文本猜测。

## 11. 与 verify 的接口契约

`verify` 阶段真正需要的是“前后版本复跑同一份触发方案”，所以 `poc` 产物必须完整保留以下能力：

1. 可重复执行的 PoC 文件集合
2. 明确的入口二进制
3. 明确的触发命令
4. 明确的预期错误模式
5. 一次实际运行后的观察结果

因此对 `verify` 的最小接口建议是：

- `trigger_command`
- `target_binary`
- `input_file_paths`
- `expected_stderr_patterns`
- `expected_stdout_patterns`
- `matched_error_patterns`
- `matched_stack_keywords`
- `observed_exit_code`
- `observed_crash_type`

## 12. 建议的代码组织

建议新增或补齐以下文件：

```text
source/app/stages/poc.py
source/app/templates/poc.Dockerfile.j2
source/app/templates/poc_run.sh.j2
source/tests/test_poc_stage.py
```

其中 `source/app/stages/poc.py` 建议包含：

- `PocStagePaths`
- `PocContext`
- `PocPlan`
- `PocStage`
- `poc_node`

`PocStage` 建议至少实现以下方法：

- `build_plan()`
- `collect_poc_context()`
- `plan_poc()`
- `_heuristic_poc_plan()`
- `_llm_poc_plan()`
- `_normalize_poc_plan()`
- `_render_dockerfile()`
- `_render_run_script()`
- `_write_payload_files()`
- `_execute_poc_plan()`
- `_extract_execution_observation()`
- `replan_after_failure()`
- `run()`

## 13. 节点行为建议

`poc_node()` 建议保持当前 LangGraph 语义，但补齐两个约束：

1. `execution_success=True` 不代表漏洞已确认，只代表 PoC 已按预期执行到目标路径
2. 如果文件写出成功但未命中目标错误模式，应该视为 `poc` 阶段完成但 `reproducer_verified=False`

建议：

- `execution_success` 表示脚本完成执行
- `reproducer_verified` 表示已至少一次观测到与漏洞相关的行为

这样 `route_after_poc()` 后续可以更精确地决定是否进入 `verify`。

## 14. 测试设计

建议优先补以下测试，而不是一开始做端到端集成：

1. `collect_poc_context()` 能从 `patch.diff` 和 `build_artifact` 收集关键字段
2. `_heuristic_poc_plan()` 在已有参考 PoC 时优先采用参考文件
3. `_normalize_poc_plan()` 会补齐默认脚本和默认模式
4. `_execute_poc_plan()` 能正确写入 `payloads/`、`inputs/`、`run.sh`
5. `poc_node()` 在异常时会累计 `retry_count["poc"]`
6. 执行日志解析能正确提取 `exit_code/stdout/stderr`

## 15. 推荐实现顺序

建议按下面顺序推进：

1. 扩展 `PoCArtifact` schema
2. 在 `poc.py` 中实现 `PocStagePaths`、`PocContext`、`PocPlan`
3. 先写纯规则版 `collect_poc_context()` 和 `_heuristic_poc_plan()`
4. 接入模板渲染和 Docker 执行
5. 完成 `run()` 主链路
6. 补 `test_poc_stage.py`
7. 最后再考虑引入 LLM 规划和重规划

## 16. 结论

对当前项目来说，`poc` 阶段最合适的实现不是“单个 Agent 直接生成一个 PoC 文本”，而是沿用 `build` 阶段已经验证过的模式：

```text
本地证据收集 -> 结构化计划 -> 文件渲染 -> Docker 执行 -> 结构化产物落盘
```

这样做的价值是：

1. 与现有 `build` 架构一致，开发成本最低
2. `poc` 与 `verify` 的边界更清晰
3. 后续排障、重试、测试都更容易
4. 生成的 PoC 不只是“文本”，而是完整的可重复执行资产

如果后续开始实现代码，建议直接以本设计为蓝本补全 `source/app/stages/poc.py`，而不是另起新的 Agent 目录或额外抽象层。
