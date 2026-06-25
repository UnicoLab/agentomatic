"""ML Model Plugins API for Agentomatic."""

from __future__ import annotations

from .ml import BaseMLPlugin
from .registry import PluginRegistry
from .router import create_plugin_router

__all__ = ["BaseMLPlugin", "PluginRegistry", "create_plugin_router"]
