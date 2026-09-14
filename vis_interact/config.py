"""
vis_interact 配置系统

从项目根目录的 config.toml 加载配置，支持环境变量覆盖。
所有模块通过 `from vis_interact.config import settings` 访问配置。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from functools import lru_cache
from typing import Optional

# Python 3.11+ 内置 tomllib，更早版本用 tomli，都不行则用内置简易解析器
try:
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib  # type: ignore[no-redef]
    _HAS_TOML_LIB = True
except ImportError:
    _HAS_TOML_LIB = False
    tomllib = None  # type: ignore[assignment]


# ── 子配置数据类 ────────────────────────────────────────────────────────────

@dataclass
class ApiConfig:
    base_url: str = ""
    api_key: str = ""


@dataclass
class ApisConfig:
    openai: ApiConfig = field(default_factory=ApiConfig)
    qwen: ApiConfig = field(default_factory=ApiConfig)
    openrouter: ApiConfig = field(default_factory=ApiConfig)
    gemini: ApiConfig = field(default_factory=ApiConfig)


@dataclass
class ModelsConfig:
    default_model: str = ""
    user_text_model: str = ""
    user_vis_model: str = ""
    eval_model: str = ""


@dataclass
class PathsConfig:
    database_path: Path = field(default_factory=lambda: Path("VisInteractBench/databases"))
    dataset_path: Path = field(default_factory=lambda: Path("VisInteractBench/test.json"))


@dataclass
class Settings:
    apis: ApisConfig = field(default_factory=ApisConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


# ── 加载逻辑 ────────────────────────────────────────────────────────────────

def _find_config_toml() -> Optional[Path]:
    """在项目根目录（vis_interact/ 的上一级）找 config.toml。"""
    here = Path(__file__).resolve().parent  # vis_interact/
    candidate = here.parent / "config.toml"
    if candidate.exists():
        return candidate
    # 也支持 VISINTERACT_CONFIG 环境变量指定路径
    env_path = os.environ.get("VISINTERACT_CONFIG")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
    return None


def _simple_toml_parse(text: str) -> dict:
    """Minimal TOML parser for flat sections with string/path values.
    Only handles our specific config.toml format; not a full TOML parser.
    """
    import re
    result: dict = {}
    current_section: list = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        # Section header like [api.openai] or [models]
        section_m = re.match(r"^\[([^\]]+)\]$", line)
        if section_m:
            current_section = section_m.group(1).split(".")
            # Ensure nested dicts exist
            d = result
            for part in current_section:
                d = d.setdefault(part, {})
            continue
        # Key = value
        kv_m = re.match(r'^(\w+)\s*=\s*(.+)$', line)
        if kv_m and current_section:
            key = kv_m.group(1)
            raw_val = kv_m.group(2).strip()
            # Strip surrounding quotes (and any trailing inline comment)
            str_m = re.match(r'^"([^"]*)"', raw_val) or re.match(r"^'([^']*)'", raw_val)
            if str_m:
                value: object = str_m.group(1)
            elif raw_val.lower() == "true":
                value = True
            elif raw_val.lower() == "false":
                value = False
            else:
                try:
                    value = int(raw_val)
                except ValueError:
                    try:
                        value = float(raw_val)
                    except ValueError:
                        value = raw_val  # leave as string
            d = result
            for part in current_section:
                d = d[part]
            d[key] = value

    return result


def _load_toml(path: Path) -> dict:
    if _HAS_TOML_LIB:
        with open(path, "rb") as f:
            return tomllib.load(f)  # type: ignore[union-attr]
    # Fallback: use built-in minimal parser
    with open(path, encoding="utf-8") as f:
        return _simple_toml_parse(f.read())


def _apply_env_overrides(s: Settings) -> None:
    """将环境变量中的值覆盖到 Settings（优先级高于 config.toml）。"""
    overrides = {
        "OPENAI_BASE_URL": ("apis", "openai", "base_url"),
        "OPENAI_API_KEY": ("apis", "openai", "api_key"),
        "QWEN_BASE_URL": ("apis", "qwen", "base_url"),
        "QWEN_API_KEY": ("apis", "qwen", "api_key"),
        "OPENROUTER_API_KEY": ("apis", "openrouter", "api_key"),
        "OPENROUTER_BASE_URL": ("apis", "openrouter", "base_url"),
        "GEMINI_API_KEY": ("apis", "gemini", "api_key"),
        "GEMINI_BASE_URL": ("apis", "gemini", "base_url"),
        "DEFAULT_MODEL": ("models", "default_model"),
        "USER_TEXT_MODEL": ("models", "user_text_model"),
        "USER_VIS_MODEL": ("models", "user_vis_model"),
        "EVAL_MODEL": ("models", "eval_model"),
        "DATABASE_PATH": ("paths", "database_path"),
        "DATASET_PATH": ("paths", "dataset_path"),
    }
    for env_key, attr_path in overrides.items():
        val = os.environ.get(env_key)
        if val is None:
            continue
        obj = s
        for part in attr_path[:-1]:
            obj = getattr(obj, part)
        last = attr_path[-1]
        # Path fields
        if isinstance(getattr(obj, last), Path):
            setattr(obj, last, Path(val))
        else:
            setattr(obj, last, val)


def _build_settings(data: dict) -> Settings:
    """从 TOML dict 构建 Settings 对象。"""
    s = Settings()

    api_section = data.get("api", {})
    for provider in ("openai", "qwen", "openrouter", "gemini"):
        cfg = api_section.get(provider, {})
        api_obj = getattr(s.apis, provider)
        if "base_url" in cfg:
            api_obj.base_url = cfg["base_url"]
        if "api_key" in cfg:
            api_obj.api_key = cfg["api_key"]

    models_section = data.get("models", {})
    for field_name in ("default_model", "user_text_model", "user_vis_model", "eval_model"):
        if field_name in models_section:
            setattr(s.models, field_name, models_section[field_name])

    paths_section = data.get("paths", {})
    if "database_path" in paths_section:
        s.paths.database_path = Path(paths_section["database_path"])
    if "dataset_path" in paths_section:
        s.paths.dataset_path = Path(paths_section["dataset_path"])

    return s


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """加载并缓存全局配置（首次调用时读取 config.toml）。"""
    config_path = _find_config_toml()
    if config_path is not None:
        data = _load_toml(config_path)
    else:
        data = {}

    s = _build_settings(data)
    _apply_env_overrides(s)
    return s


def reset_settings_cache() -> None:
    """清除配置缓存（用于测试或热重载）。"""
    get_settings.cache_clear()


# ── 模块级单例 ───────────────────────────────────────────────────────────────

settings: Settings = get_settings()
