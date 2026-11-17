"""技能：data/ai/skills/<name>/SKILL.md，提示词中只列名称与说明，
模型用 read_skill 读取全文；附带的脚本只能由超级用户在沙箱中运行"""

import asyncio
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import ClassVar
import zipfile

import yaml

from zhenxun.configs.config import Config
from zhenxun.configs.path_config import DATA_PATH
from zhenxun.services.ai.run.context import RunContext
from zhenxun.services.ai.tools.core.decorators import tool
from zhenxun.services.ai.tools.core.toolkit import BaseToolkit
from zhenxun.services.ai.tools.models import ResolvedToolPayload, ToolResult
from zhenxun.services.log import logger

from ._data_source import MODULE

SKILL_PATH = DATA_PATH / "ai" / "skills"
SKILL_FILE = "SKILL.md"
SKILL_MAX_BYTES = 5 * 1024 * 1024
"""上传的技能包上限"""
SKILL_TEXT_MAX = 64 * 1024
"""SKILL.md 正文上限"""
SANDBOX_DIR = "skills"
"""技能文件在沙箱工作目录下的位置"""
NAME = re.compile(r"^[A-Za-z0-9][\w.-]{0,63}$")
_FRONT = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.S)


@dataclass
class Skill:
    """一个技能"""

    name: str
    description: str
    path: Path
    enabled: bool


def _parse(text: str) -> tuple[dict, str]:
    """拆出 SKILL.md 的 frontmatter 与正文

    参数:
        text: SKILL.md 内容

    返回:
        tuple[dict, str]: frontmatter, 正文
    """
    match = _FRONT.match(text.lstrip("﻿"))
    if not match:
        return {}, text
    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    return (meta if isinstance(meta, dict) else {}), text[match.end() :]


class Skills:
    """技能目录的读取、启用与安装"""

    _cache: ClassVar[tuple[float, list[Skill]] | None] = None

    @classmethod
    def disabled(cls) -> set[str]:
        """停用的技能名

        返回:
            set[str]: 技能名
        """
        return set(Config.get_config(MODULE, "DISABLED_SKILLS") or [])

    @classmethod
    def all(cls) -> list[Skill]:
        """全部技能，按名称排序；目录未修改时复用上次的扫描结果

        返回:
            list[Skill]: 技能
        """
        SKILL_PATH.mkdir(parents=True, exist_ok=True)
        stamp = max(
            [SKILL_PATH.stat().st_mtime]
            + [p.stat().st_mtime for p in SKILL_PATH.glob(f"*/{SKILL_FILE}")]
        )
        if cls._cache is None or cls._cache[0] != stamp:
            skills = []
            for file in sorted(SKILL_PATH.glob(f"*/{SKILL_FILE}")):
                try:
                    meta, _ = _parse(file.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError) as e:
                    logger.warning(f"读取技能 {file.parent.name} 失败: {e}", MODULE)
                    continue
                skills.append(
                    Skill(
                        name=file.parent.name,
                        description=" ".join(str(meta.get("description", "")).split()),
                        path=file.parent,
                        enabled=True,
                    )
                )
            cls._cache = (stamp, skills)
        disabled = cls.disabled()
        for skill in cls._cache[1]:
            skill.enabled = skill.name not in disabled
        return list(cls._cache[1])

    @classmethod
    def available(cls, allowed: list[str] | None) -> list[Skill]:
        """人设可用的已启用技能

        参数:
            allowed: 人设的技能白名单，为空时不限制

        返回:
            list[Skill]: 技能
        """
        return [
            s for s in cls.all() if s.enabled and (allowed is None or s.name in allowed)
        ]

    @staticmethod
    def files(skill: Skill) -> list[str]:
        """技能附带的文件（SKILL.md 以外）

        参数:
            skill: 技能

        返回:
            list[str]: 相对路径
        """
        return sorted(
            p.relative_to(skill.path).as_posix()
            for p in skill.path.rglob("*")
            if p.is_file() and p.name != SKILL_FILE
        )

    @classmethod
    def set_enabled(cls, name: str, enabled: bool) -> None:
        """启用或停用技能

        参数:
            name: 技能名
            enabled: 是否启用
        """
        disabled = cls.disabled() - {name} if enabled else cls.disabled() | {name}
        Config.set_config(MODULE, "DISABLED_SKILLS", sorted(disabled))
        Config.save(save_simple_data=True)

    @classmethod
    def install(cls, filename: str, data: bytes) -> Skill:
        """安装技能：SKILL.md 单文件或 zip 包（SKILL.md 在根目录或唯一的顶层目录中），
        技能名取 frontmatter 的 name，同名技能被替换

        参数:
            filename: 文件名
            data: 文件内容

        返回:
            Skill: 安装的技能

        异常:
            ValueError: 文件无效
        """
        if len(data) > SKILL_MAX_BYTES:
            raise ValueError("技能包超过 5MB。")
        files: dict[str, bytes] = {}
        if filename.lower().endswith(".zip"):
            try:
                archive = zipfile.ZipFile(BytesIO(data))
            except zipfile.BadZipFile as e:
                raise ValueError("不是有效的 zip 文件。") from e
            infos = [i for i in archive.infolist() if not i.is_dir()]
            if sum(i.file_size for i in infos) > SKILL_MAX_BYTES * 4:
                raise ValueError("技能包解压后过大。")
            roots = {PurePosixPath(i.filename).parts[0] for i in infos}
            prefix = ""
            if SKILL_FILE not in {i.filename for i in infos} and len(roots) == 1:
                prefix = f"{roots.pop()}/"
            for info in infos:
                path = PurePosixPath(info.filename[len(prefix) :])
                if not info.filename.startswith(prefix) or path.is_absolute():
                    continue
                if ".." in path.parts or not path.parts:
                    raise ValueError(f"技能包中的路径无效: {info.filename}")
                files[path.as_posix()] = archive.read(info)
        else:
            files[SKILL_FILE] = data
        if SKILL_FILE not in files:
            raise ValueError("技能包中没有 SKILL.md。")
        try:
            text = files[SKILL_FILE].decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError("SKILL.md 不是 UTF-8 文本。") from e
        meta, body = _parse(text)
        name = str(meta.get("name") or "").strip()
        if not NAME.match(name):
            raise ValueError("SKILL.md 的 name 只能由字母、数字、._- 组成。")
        if not str(meta.get("description") or "").strip() or not body.strip():
            raise ValueError("SKILL.md 需要 description 与正文。")
        if len(body.encode()) > SKILL_TEXT_MAX:
            raise ValueError("SKILL.md 正文超过 64KB。")
        target = SKILL_PATH / name
        if target.exists():
            shutil.rmtree(target)
        for rel, content in files.items():
            path = target / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        cls._cache = None
        return next(s for s in cls.all() if s.name == name)

    @classmethod
    def remove(cls, name: str) -> bool:
        """删除技能

        参数:
            name: 技能名

        返回:
            bool: 是否删除
        """
        target = SKILL_PATH / name
        if not NAME.match(name) or not (target / SKILL_FILE).exists():
            return False
        shutil.rmtree(target)
        cls._cache = None
        cls.set_enabled(name, True)
        return True


async def _copy_to_sandbox(skill: Skill, context: RunContext) -> str | None:
    """超级用户开启 Agent 且沙箱可用时，把技能文件复制到沙箱

    参数:
        skill: 技能
        context: 运行上下文

    返回:
        str | None: 沙箱中的目录，不能复制时为空
    """
    from zhenxun.services.ai.sandbox.manager import SandboxManager

    from ._agent import AgentCapability

    if not await AgentCapability().active(context):
        return None
    if not (await SandboxManager.ready())[0]:
        return None
    sandbox = await SandboxManager.get(context.session_id or "default")
    folder = f"{SANDBOX_DIR}/{skill.name}"
    for rel in [SKILL_FILE, *Skills.files(skill)]:
        data = await asyncio.to_thread((skill.path / rel).read_bytes)
        await sandbox.write_file(f"{folder}/{rel}", data)
    return folder


class SkillToolkit(BaseToolkit):
    """按人设白名单列出已启用的技能，提供 read_skill"""

    def __init__(self, allowed: list[str] | None = None):
        super().__init__()
        self.allowed = allowed

    async def resolve(self, context: RunContext | None = None) -> ResolvedToolPayload:
        skills = Skills.available(self.allowed)
        if not skills:
            return ResolvedToolPayload()
        payload = await super().resolve(context)
        lines = [f"- {s.name}：{s.description}" for s in skills]
        instructions = (
            "## 技能\n"
            "下面是可用的技能。任务与某个技能相关时，先用 `read_skill` "
            "读取它的完整说明，再按说明去做；无关时不要读取。\n" + "\n".join(lines)
        )
        payload.instruction_builders.append(
            lambda names: [instructions] if "read_skill" in names else []
        )
        return payload

    @tool(name="read_skill", description="读取一个技能的完整说明。")
    async def read_skill(self, name: str, context: RunContext) -> ToolResult:
        """读取技能全文，附带文件时说明能否在沙箱中运行

        参数:
            name: 技能名
        """
        skill = next(
            (s for s in Skills.available(self.allowed) if s.name == name.strip()), None
        )
        if skill is None:
            return ToolResult(output=f"没有技能 {name}。").as_error(False)
        _, body = _parse(
            await asyncio.to_thread((skill.path / SKILL_FILE).read_text, "utf-8")
        )
        files = Skills.files(skill)
        if not files:
            return ToolResult(output=body.strip())
        try:
            folder = await _copy_to_sandbox(skill, context)
        except Exception as e:
            logger.warning(f"复制技能 {skill.name} 到沙箱失败: {e}", MODULE)
            folder = None
        note = (
            f"技能文件已放在沙箱的 {folder}/ 中，可以在沙箱里运行。"
            if folder
            else "这个技能附带的脚本只有超级用户在沙箱中才能运行，"
            "现在只能按说明中不需要脚本的部分去做。"
        )
        listing = "\n".join(f"- {f}" for f in files)
        return ToolResult(output=f"{body.strip()}\n\n附带文件：\n{listing}\n{note}")
