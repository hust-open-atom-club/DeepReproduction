"""文件说明：常见漏洞类型的 PoC 模板。

这些模板为 LLM 提供具体的参考骨架，减少从零生成时的错误。
每个模板包含触发模式描述、运行命令模式、payload 结构和编译注意事项。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PocTemplate:
    """一个 PoC 模板，包含参考骨架和使用说明。"""

    name: str
    description: str
    trigger_modes: list[str]
    vulnerability_keywords: list[str]
    run_command_pattern: str
    payload_structure: str
    compilation_notes: str
    example_payload_excerpt: str = ""
    example_run_command: str = ""


TEMPLATES: list[PocTemplate] = [
    PocTemplate(
        name="library_harness",
        description=(
            "编译一个 C 程序，链接目标共享库，调用库 API 触发漏洞。"
            "适用于需要通过库函数调用才能触发的漏洞（如 unicorn、openssl、libxml2 等）。"
        ),
        trigger_modes=["library-harness"],
        vulnerability_keywords=[
            "out-of-bounds write", "out-of-bounds read",
            "heap-buffer-overflow", "use-after-free",
            "out-of-bounds", "memory corruption",
        ],
        run_command_pattern=(
            "cd {project_dir} && "
            "gcc -fsanitize=address -o /tmp/poc "
            "/workspace/artifacts/poc/payloads/{payload_filename} "
            "{include_flags} {library_flags} && "
            "{runtime_env} /tmp/poc"
        ),
        payload_structure=(
            "C 源码文件 (.c)，包含 main() 函数，直接调用目标库 API。\n"
            "关键要素：\n"
            "  1. #include 目标库的头文件\n"
            "  2. 构造触发漏洞的输入数据（字节序列、结构体、字符串等）\n"
            "  3. 调用目标函数并传入构造好的输入\n"
            "  4. 不需要 printf 调试输出，ASan 会自动报告崩溃"
        ),
        compilation_notes=(
            "- 必须加 -fsanitize=address（如果 build 用了 ASan）\n"
            "- -I 指向 include 目录，-L 指向 .so 所在目录（用 PROBED SHARED LIBRARIES 的结果）\n"
            "- 运行时需要 LD_LIBRARY_PATH 指向 .so 目录\n"
            "- 注意 bool 类型需要用 C99 的 <stdbool.h> 或改用 int"
        ),
        example_payload_excerpt=(
            "#include <unicorn/unicorn.h>\n"
            "int main() {\n"
            "    uc_engine *uc;\n"
            "    uc_open(UC_ARCH_ARM, UC_MODE_THUMB, &uc);\n"
            "    // ... 构造触发条件 ...\n"
            "    uc_emu_start(uc, start, end, 0, 1);  // stepping mode\n"
            "    uc_close(uc);\n"
            "    return 0;\n"
            "}"
        ),
        example_run_command=(
            "cd /src/unicorn && gcc -fsanitize=address -o /tmp/poc "
            "/workspace/artifacts/poc/payloads/poc.c "
            "-I include -L build -lunicorn && LD_LIBRARY_PATH=build /tmp/poc"
        ),
    ),
    PocTemplate(
        name="cli_file_input",
        description=(
            "目标二进制读取一个构造的输入文件并崩溃。"
            "适用于文件解析类漏洞（如 mdbtools、libpng、libtiff、sqlite 等）。"
        ),
        trigger_modes=["cli-file"],
        vulnerability_keywords=[
            "buffer overflow", "stack-buffer-overflow",
            "heap-buffer-overflow", "heap buffer",
            "out-of-bounds read", "crash",
        ],
        run_command_pattern=(
            "'{target_binary}' {cli_flags} "
            "/workspace/artifacts/poc/payloads/{payload_filename}"
        ),
        payload_structure=(
            "二进制或文本输入文件，由目标程序直接读取。\n"
            "构造方法：\n"
            "  1. 分析 patch 中受影响的解析函数\n"
            "  2. 构造触发边界检查缺失或整数溢出的输入\n"
            "  3. 对于二进制格式，需要用 Python 脚本生成（放在 auxiliary_files 中）\n"
            "  4. payload_filename 直接是输入文件"
        ),
        compilation_notes=(
            "- 通常不需要编译，直接用目标二进制处理输入文件\n"
            "- 如果需要生成二进制 payload，用 Python 的 struct.pack 生成辅助脚本\n"
            "- 注意目标二进制的命令行参数格式（用 PROBED BINARY USAGE 的结果）"
        ),
        example_run_command=(
            "'/src/mdbtools/src/util/mdb_dump' -S "
            "/workspace/artifacts/poc/payloads/payload.mdb"
        ),
    ),
    PocTemplate(
        name="script_driver",
        description=(
            "用解释器（lua、python、perl 等）运行一个构造的脚本文件。"
            "适用于脚本语言解释器自身的漏洞（如 Lua、Python、Perl 等）。"
        ),
        trigger_modes=["script-driver"],
        vulnerability_keywords=[
            "heap buffer", "heap-buffer-overflow",
            "out-of-bounds", "use-after-free",
            "crash", "assert",
        ],
        run_command_pattern=(
            "{interpreter} /workspace/artifacts/poc/payloads/{payload_filename}"
        ),
        payload_structure=(
            "脚本文件（.lua、.py、.pl 等），直接调用触发漏洞的内置函数。\n"
            "关键要素：\n"
            "  1. 使用目标语言的内置函数/API\n"
            "  2. 构造触发边界检查缺失的输入（大量重复、嵌套、特殊字符等）\n"
            "  3. 不需要额外的编译步骤\n"
            "  4. 脚本应自包含，不依赖外部文件"
        ),
        compilation_notes=(
            "- 不需要编译，解释器直接执行脚本\n"
            "- 注意解释器的路径（通常在 /usr/local/bin/ 或 build 目录）\n"
            "- Lua 示例：table.foreach 构造恶意表触发 heap buffer over-read"
        ),
        example_payload_excerpt=(
            "-- CVE-2022-28805 Lua table.foreach heap buffer over-read\n"
            "local t = {}\n"
            "for i = 1, 100 do\n"
            "    t[i] = i\n"
            "end\n"
            "table.foreach(t, function(k, v)\n"
            "    -- 触发漏洞的回调\n"
            "end)"
        ),
        example_run_command=(
            "lua /workspace/artifacts/poc/payloads/poc.lua"
        ),
    ),
]


def select_template(
    trigger_mode: str,
    vulnerability_type: str,
    inferred_input_modes: list[str],
    chosen_strategy: str = "",
    target_binary: str = "",
) -> PocTemplate | None:
    """根据触发模式和漏洞类型选择最匹配的模板。

    优先匹配 trigger_mode，其次根据 inferred_input_modes 推断。
    """

    trigger_lower = (trigger_mode or "").strip().lower()
    vtype_lower = (vulnerability_type or "").strip().lower()
    input_modes = {m.strip().lower() for m in inferred_input_modes}

    # Exact trigger_mode match
    for tmpl in TEMPLATES:
        if trigger_lower in tmpl.trigger_modes:
            return tmpl

    # library-harness when the target is a .so file or strategy involves a recipe
    # that calls library APIs (e.g. unicorn's uc_open/uc_emu_start)
    if target_binary.endswith(".so") or target_binary.endswith(".so.1"):
        for tmpl in TEMPLATES:
            if tmpl.name == "library_harness":
                return tmpl

    # Heuristic: library-harness if vulnerability type suggests memory
    # corruption in a library (not a standalone binary)
    if any(kw in vtype_lower for kw in ("out-of-bounds", "heap", "use-after-free")):
        # If the input mode is file-based, it could be either library_harness
        # or cli_file_input.  Default to cli_file_input (more common).
        # Library harness is selected when the chosen_strategy says so.
        pass

    # script-driver if input modes include script-related keywords
    if any(m in input_modes for m in ("script", "lua", "python", "perl")):
        for tmpl in TEMPLATES:
            if tmpl.name == "script_driver":
                return tmpl

    # cli-file for file-based input (most common pattern)
    if "file" in input_modes:
        for tmpl in TEMPLATES:
            if tmpl.name == "cli_file_input":
                return tmpl

    # stdin-based input
    if "stdin" in input_modes:
        for tmpl in TEMPLATES:
            if tmpl.name == "cli_file_input":
                return tmpl

    return None


def format_template_for_prompt(template: PocTemplate) -> str:
    """将模板格式化为 prompt 可读的文本。"""

    sections = [
        f"=== PoC Template: {template.name} ===",
        f"适用场景: {template.description}",
        "",
        f"运行命令模式:",
        f"  {template.run_command_pattern}",
        "",
        f"Payload 结构:",
        template.payload_structure,
        "",
        f"编译注意事项:",
        template.compilation_notes,
    ]
    if template.example_payload_excerpt:
        sections.extend([
            "",
            "示例 Payload 片段:",
            template.example_payload_excerpt,
        ])
    if template.example_run_command:
        sections.extend([
            "",
            "示例运行命令:",
            f"  {template.example_run_command}",
        ])
    return "\n".join(sections)
