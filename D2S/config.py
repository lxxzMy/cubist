# -*- coding: utf-8 -*-
"""settings.yaml 读写与自愈（损坏隔离 → .default 恢复）。

从 utils.py 拆出（2026-11-16）。导入副作用与原 utils 相同：
settings 在导入期读取一次（绝对路径锚定本文件旁）。
"""
import os
import shutil
import time
import yaml

def _quarantine_corrupt_yaml(path):
    """Move an unreadable YAML aside so the next start can fall back to the
    shipped default instead of crashing on the same corrupt bytes forever."""
    try:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        os.replace(path, f"{path}.corrupt-{stamp}")
        print(f"[utils] moved corrupt {path} to {path}.corrupt-{stamp}")
    except OSError as e:
        print(f"[utils] could not move corrupt {path} aside: {e}")
    return {}


def read_yaml(path):
    """Load a YAML file. Never raises: a missing file maps to {}, a corrupt
    file is moved aside (see _quarantine_corrupt_yaml) and also maps to {}."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}
    except UnicodeDecodeError:
        # Fallback to try other common encodings if UTF-8 fails
        try:
            with open(path, "r", encoding="gbk") as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            print(f"Failed to load settings.yaml with GBK encoding: {e}")
            return _quarantine_corrupt_yaml(path)
        except Exception as e:
            print(f"Failed to load settings.yaml with GBK encoding: {e}")
            return _quarantine_corrupt_yaml(path)
    except yaml.YAMLError as e:
        print(f"[utils] {path} is not valid YAML: {e}")
        return _quarantine_corrupt_yaml(path)
    except OSError as e:
        print(f"[utils] cannot read {path}: {e}")
        return {}


def write_yaml(path, updates):
    """Merge ``updates`` into the YAML at *path* and write it back.

    Reads the existing file (or starts with an empty dict if missing/corrupt),
    overlays the keys from ``updates``, and writes it back as UTF-8.  Unknown
    keys already present in the file are preserved — only the keys explicitly
    in ``updates`` are touched.  This is what runtime callers (xrviewer's
    brand-switch / environment-switch hooks) use to persist user choices
    without clobbering GUI-only fields.

    Returns True on success, False on failure (errors are printed but never
    raised so a save failure can never crash the VR session).
    """
    try:
        cfg = {}
        if os.path.exists(path):
            try:
                cfg = read_yaml(path)
            except Exception:
                cfg = {}
        cfg.update(updates or {})
        # Atomic write: a crash mid-write must never leave a truncated YAML
        # behind (the module-level read at import time would die on it).
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        os.replace(tmp_path, path)
        return True
    except Exception as e:
        print(f"[utils] Failed to write {path}: {e}")
        return False
# load customized settings
# 必须用绝对路径：相对路径依赖当前工作目录，从别处启动时会把 settings.yaml
# 读到/app 之外（实测会凭空在运行目录造一个默认 yaml，用户设置全部丢失）。
_HERE_DIR = os.path.dirname(os.path.abspath(__file__))
_SETTINGS_PATH = os.path.join(_HERE_DIR, "settings.yaml")
settings = read_yaml(_SETTINGS_PATH)
if not settings:
    # Missing or corrupt beyond repair (read_yaml already moved a corrupt
    # file aside): restore the shipped default so the ~36 module-level key
    # accesses below don't KeyError at import time.
    _default_yaml = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "settings.yaml.default"
    )
    if os.path.exists(_default_yaml):
        print("[utils] settings.yaml missing/corrupt - restoring shipped default")
        try:
            import shutil
            shutil.copyfile(_default_yaml, _SETTINGS_PATH)
            settings = read_yaml(_SETTINGS_PATH)
        except Exception as e:
            print(f"[utils] failed to restore default settings: {e}")
