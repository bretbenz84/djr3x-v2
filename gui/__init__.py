"""Optional PySide6 dashboard for DJ-R3X.

Importing this package must stay cheap and Qt-free so headless controller runs
do not gain a GUI dependency.
"""

from .state_bridge import GUIDashboardBridge, gui_bridge, get_bridge

__all__ = ["GUIDashboardBridge", "gui_bridge", "get_bridge"]
