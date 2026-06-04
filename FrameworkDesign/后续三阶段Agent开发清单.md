<!-- 文件说明：build / poc / verify 三阶段 Agent 的开发清单。用于明确阶段边界、输入输出、实现顺序和落地注意事项。 -->

# 后续三阶段 Agent 开发清单

## 设计前提

当前项目主流程已经固定为：

`knowledge -> build -> poc -> verify`

其中：

- `knowledge` 作为静态前处理阶段，负责收集外部漏洞知识并生成 `knowledge.yaml`
- `build`、`poc`、`verify` 作为动态执行阶段，负责在 Docker 容器中完成真实构建、触发与验证

因此，后续三阶段的设计原则应为：

1. 不要求 `knowledge` 准确确认构建方式
2. `knowledge` 中的 build 信息只视为候选线索，不视为最终事实
3. 后续三阶段共享同一个 workspace，不重复做外部知识搜索
4. 真正的源码下载、版本切换、构建确认应由 `build` 阶段主导完成
5. `build`、`poc`、`verify` 的真实执行均通过 Docker 完成，宿主机只负责准备 workspace 和调度容器

## 总体开发顺序

建议按以下顺序推进，而不是先搭完整 LangGraph 再往里填逻辑：

1. 明确三阶段输入输出契约
2. 明确每个阶段的工作目录和产物落盘方式
3. 先分别实现 `build.py`、`poc.py`、`verify.py` 的最小可运行骨架
4. 为三个 schema 增补必要字段
5. 为每个阶段补单测或最小集成测试
6. 最后接入和细化 LangGraph 路由与重试

## 阶段职责边界

### 1. Build Agent

职责：

- 消费 `knowledge.yaml`
- 准备本地 workspace
- clone 仓库并 checkout 到指定版本
- 扫描仓库中的真实构建文件
- 读取 README、INSTALL、CI 配置等构建线索
- 生成 `Dockerfile`、`build.sh`
- 通过 Docker 执行构建并产出日志
- 输出“确认后的构建事实”

不要做的事：

- 不重复搜索外部 advisory / issue / CVE 页面
- 不负责生成 PoC
- 不负责最终漏洞是否修复的判定

输入：

- `KnowledgeModel`
- `workspace`

最低应确认的信息：

- 仓库是否可拉取
- `vulnerable_ref` / `fixed_ref` 是否可 checkout
- 构建系统是什么
- 依赖安装方式是什么
- 核心构建命令是什么
- 预期二进制或执行入口在哪里

### 2. PoC Agent

职责：

- 基于 `KnowledgeModel` 和 `BuildArtifact` 生成最小触发样例
- 从 patch、源码、函数路径、错误模式中提炼触发条件
- 生成 PoC 主文件、辅助输入文件和运行脚本
- 在已准备好的 workspace 中执行 PoC
- 通过 Docker 容器执行 PoC
- 收集崩溃日志或错误输出

不要做的事：

- 不重新 clone 仓库
- 不重新搜索外部网页资料
- 不负责最终成功/失败裁决

输入：

- `KnowledgeModel`
- `BuildArtifact`
- `workspace`

最低应确认的信息：

- 触发入口是命令行、输入文件、参数还是网络输入
- 需要哪些输入文件
- 应使用哪个可执行文件或脚本触发
- 预期命中的错误模式或栈关键词是什么

### 3. Verify Agent

职责：

- 基于 `KnowledgeModel`、`BuildArtifact`、`PoCArtifact` 做前后版本对比验证
- 在 vulnerable 版本上验证漏洞可触发
- 在 fixed 版本上验证漏洞不再触发或行为改变
- 通过 Docker 容器完成前后版本运行和日志比对
- 归纳验证结论和证据

不要做的事：

- 不重复生成 PoC
- 不重复决定构建方案
- 不重新搜索外部资料

输入：

- `KnowledgeModel`
- `BuildArtifact`
- `PoCArtifact`
- `workspace`

最低应确认的信息：

- vulnerable 版本是否稳定触发
- fixed 版本是否不再触发
- 是否命中预期错误模式
- 是否命中预期栈关键词
- 是否存在误报或非预期失败

## Workspace 设计建议

后续三阶段必须复用统一 workspace，避免每个 Agent 各自下载与构建。

建议在 `workspaces/<CVE>/` 下统一组织：

```text
workspaces/<CVE>/
├── repo/                  # clone 下来的目标仓库
├── artifacts/
│   ├── build/
│   │   ├── Dockerfile
│   │   ├── build.sh
│   │   ├── build.log
│   │   └── build_artifact.yaml
│   ├── poc/
│   │   ├── poc.*
│   │   ├── run.sh
│   │   ├── inputs/
│   │   ├── poc.log
│   │   └── poc_artifact.yaml
│   └── verify/
│       ├── verify.log
│       └── verify_result.yaml
└── temp/
```

建议约束：

- `build` 阶段拥有 `repo/` 和 `artifacts/build/` 的写权限
- `poc` 阶段不重建 repo，只读 `repo/`，写 `artifacts/poc/`
- `verify` 阶段只消费前两者产物，写 `artifacts/verify/`
- `build`、`poc`、`verify` 的脚本都默认以 `workspaces/<CVE>/` 作为 Docker build context 或挂载根目录
- 宿主机不直接运行 `build.sh`、`run.sh` 或 verify 对比命令

## Schema 开发清单

### 1. `BuildArtifact`

当前已有字段基础可用，但建议补充以下字段：

- `repo_local_path`
- `resolved_ref`
- `build_system`
- `detected_build_files`
- `dependency_sources`
- `source_of_truth`
- `binary_or_entrypoint`
- `docker_image_tag`

字段语义建议：

- `install_packages` 表达系统依赖
- `build_commands` 表达确认后的构建命令
- `detected_build_files` 表达从仓库扫描确认的构建文件
- `source_of_truth` 标记该结论来自 `repo_scan`、`readme`、`knowledge_hint` 或 `manual_fallback`

### 2. `PoCArtifact`

建议补充以下字段：

- `trigger_command`
- `input_file_paths`
- `target_binary`
- `expected_stdout_patterns`
- `expected_stderr_patterns`
- `observed_crash_type`
- `reproducer_verified`

字段语义建议：

- `poc_content` 表达主输入或主脚本
- `input_files` 表达需要落盘的额外文件
- `execution_logs` 表达本阶段单次执行日志
- `reproducer_verified` 表达该 PoC 是否至少成功执行到目标入口

### 3. `VerifyResult`

建议补充以下字段：

- `pre_patch_exit_code`
- `post_patch_exit_code`
- `pre_patch_logs`
- `post_patch_logs`
- `pre_patch_observation`
- `post_patch_observation`
- `confidence`

字段语义建议：

- `verdict` 只保留少量枚举，例如 `success`、`failed`、`inconclusive`
- `reason` 只写最终结论摘要
- 详细证据通过新增字段承载，而不是都塞到 `reason`

## Build Agent 实现清单

### 第一步：实现构建计划生成

目标：

- 从 `KnowledgeModel` 中读取 `repo_url`、`vulnerable_ref`、`fixed_ref`
- 读取候选 `build_hints`、`build_commands`、`build_files`
- 生成内部 plan

plan 最少包含：

- 仓库地址
- 目标 ref
- 候选构建系统
- 候选依赖安装方案
- 候选构建命令
- 预期产物路径

### 第二步：实现仓库准备

目标：

- clone 仓库到 `workspace/repo`
- checkout 到目标版本
- 补充分支、tag、commit 不存在时的错误处理

要求：

- `build` 是唯一负责下载源码的阶段
- 后续 `poc` / `verify` 不重复 clone

### 第三步：实现构建信息确认

目标：

- 扫描真实仓库中的构建文件
- 读取 README / INSTALL / CI 配置
- 确认 build system 和 install strategy

优先检查：

- `Makefile`
- `configure` / `configure.ac`
- `CMakeLists.txt`
- `meson.build`
- `build.ninja`
- `Cargo.toml`
- `go.mod`
- `package.json`
- `pom.xml`
- `build.gradle`
- `.github/workflows/*`
- `.gitlab-ci.yml`

### 第四步：实现构建产物生成

目标：

- 生成 `Dockerfile`
- 生成 `build.sh`
- 写入 `artifacts/build/`

要求：

- `Dockerfile` 和 `build.sh` 必须可单独落盘复用
- 模板生成与命令执行分离

### 第五步：实现构建执行与日志收集

目标：

- 执行构建脚本
- 保存 stdout / stderr
- 输出 `BuildArtifact`

要求：

- 失败时也要有完整日志产物
- 不要只返回布尔值

## PoC Agent 实现清单

### 第一步：实现输入上下文整理

目标：

- 汇总 `KnowledgeModel` 中的漏洞摘要、受影响文件、错误模式、栈关键词
- 汇总 `BuildArtifact` 中的执行入口、二进制路径、构建命令
- 形成 PoC 计划

### 第二步：实现 PoC 策略生成

目标：

- 判断触发方式属于哪一类

建议至少覆盖：

- 文件输入型
- 命令行参数型
- 配置文件型
- API / 请求型

### 第三步：生成 PoC 文件和运行脚本

目标：

- 生成主 PoC 文件
- 生成附属输入文件
- 生成 `run.sh`
- 所有产物写入 `artifacts/poc/`

### 第四步：执行并收集结果

目标：

- 使用 `BuildArtifact` 里的入口执行 PoC
- 保存执行日志
- 判断 PoC 是否至少跑通目标路径

要求：

- 区分“脚本执行成功”和“漏洞成功触发”
- 这一步只负责执行，不做最终结论

## Verify Agent 实现清单

### 第一步：实现验证计划整理

目标：

- 基于已有构建产物和 PoC 产物，确定前后版本验证步骤
- 明确先验证 vulnerable，再验证 fixed

### 第二步：实现前版本验证

目标：

- checkout vulnerable 版本
- 复用或重新执行必要构建
- 运行 PoC
- 收集日志和退出码

### 第三步：实现后版本验证

目标：

- checkout fixed 版本
- 复用或重新执行必要构建
- 再次运行同一个 PoC
- 收集日志和退出码

### 第四步：实现结论归纳

目标：

- 比较前后行为差异
- 判断是否命中错误模式和栈关键词
- 输出 `VerifyResult`

结论规则建议：

- vulnerable 触发且 fixed 不触发：`success`
- vulnerable 未触发：`failed`
- 两边都触发：`failed`
- 日志不足或构建不稳定：`inconclusive`

## LangGraph 接入清单

在三个阶段各自能够独立运行后，再做编排层集成。

### 第一阶段：保持当前主图不变

继续使用：

- `knowledge`
- `build`
- `poc`
- `verify`

不建议在这个阶段先把 `build / poc / verify` 再拆成更细子图。

### 第二阶段：补全路由规则

需要明确：

- `build` 失败后哪些错误允许重试
- `poc` 失败后哪些错误允许重试
- `verify` 的 `inconclusive` 是否直接结束还是回退重跑

### 第三阶段：补全状态字段

建议在 `AppState` 中增加或确认：

- `workspace`
- `build`
- `poc`
- `verify`
- `last_error`
- `retry_count`
- `stage_history`
- `final_status`

必要时可增加：

- `artifacts_root`
- `current_ref`
- `execution_mode`

## 测试清单

### 单元测试

- `build plan` 生成逻辑
- 仓库构建文件识别逻辑
- PoC 策略选择逻辑
- Verify 结论归纳逻辑

### 集成测试

- 从现有 `knowledge.yaml` 构造 `BuildArtifact`
- 从 `BuildArtifact` 构造 `PoCArtifact`
- 从三者构造 `VerifyResult`

### 样本建议

至少准备三类 CVE：

- 有明显构建系统的简单项目
- 只有单一 C/C++ 可执行程序的项目
- 文档稀缺、需要从仓库实际扫描构建文件的项目

## 实施优先级

建议按以下优先级开发：

1. `BuildArtifact` 字段完善
2. `BuildStage.run()` 最小实现
3. `PoCArtifact` 字段完善
4. `PoCStage.run()` 最小实现
5. `VerifyResult` 字段完善
6. `VerifyStage.run()` 最小实现
7. 三阶段各自的落盘产物
8. LangGraph 路由与重试细化

## 最终原则

后续三阶段应遵守以下边界：

1. `knowledge` 提供事实基础和候选线索，但不负责确认真实构建方式
2. `build` 是唯一负责准备源码 workspace 和确认构建方案的阶段
3. `poc` 和 `verify` 复用 `build` 产物，不重复下载和外部搜索
4. 各阶段优先共享本地产物，而不是各自维护独立事实源
5. LangGraph 只负责编排，不负责替代阶段设计本身
