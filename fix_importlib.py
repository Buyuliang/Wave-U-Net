"""
修复 Python 3.6 中 importlib.resources 的兼容性问题
需要在导入任何使用 stempeg 的模块之前导入此模块
"""
import sys

if sys.version_info < (3, 9):
    try:
        import importlib_resources
        # Monkey patch: 让 importlib.resources 在 Python 3.6 中可用
        import importlib
        if not hasattr(importlib, 'resources'):
            importlib.resources = importlib_resources
        # 同时修复 sys.modules，让 'importlib.resources' 模块可用
        sys.modules['importlib.resources'] = importlib_resources
    except ImportError:
        print("警告: 未安装 importlib_resources，某些功能可能无法使用")
        pass
