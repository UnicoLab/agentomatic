"""ML Model Plugins API for Agentomatic."""

from .ml import BaseMLPlugin
from .registry import PluginRegistry
from .router import create_plugin_router

__all__ = ["BaseMLPlugin", "PluginRegistry", "create_plugin_router"]
