# app/ui/core/theme_manager.py
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import qdarktheme
from PySide6 import QtCore, QtGui, QtWidgets

from app.ui.core.asset_manager import AssetManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ThemeSpec:
    """Immutable metadata specification for a UI theme."""

    id: str
    display_name: str
    qss_filename: str
    base_mode: str = "dark"  # "dark" or "light" for qdarktheme base layer
    primary_color: str = "#4090a3"  # Default accent color passed to qdarktheme


# Built-in curated registry matching app/ui/styles/*.qss
BUILTIN_THEMES: Dict[str, ThemeSpec] = {
    "true_dark": ThemeSpec(
        id="true_dark",
        display_name="True Dark (Default)",
        qss_filename="true_dark_styles.qss",
        base_mode="dark",
        primary_color="#4090a3",
    ),
    "dark": ThemeSpec(
        id="dark",
        display_name="Classic Dark",
        qss_filename="dark_styles.qss",
        base_mode="dark",
        primary_color="#4090a3",
    ),
    "oled_black": ThemeSpec(
        id="oled_black",
        display_name="OLED Pitch Black",
        qss_filename="oled_black.qss",
        base_mode="dark",
        primary_color="#4090a3",
    ),
    "windows11_dark": ThemeSpec(
        id="windows11_dark",
        display_name="Windows 11 Dark",
        qss_filename="windows11_dark.qss",
        base_mode="dark",
        primary_color="#60cdff",
    ),
    "glass_studio": ThemeSpec(
        id="glass_studio",
        display_name="Glass Studio",
        qss_filename="glass_studio.qss",
        base_mode="dark",
        primary_color="#58a6ff",
    ),
    "dark_blue": ThemeSpec(
        id="dark_blue",
        display_name="Deep Ocean Blue",
        qss_filename="dark_blue.qss",
        base_mode="dark",
        primary_color="#3b82f6",
    ),
    "dracula": ThemeSpec(
        id="dracula",
        display_name="Dracula",
        qss_filename="dracula.qss",
        base_mode="dark",
        primary_color="#bd93f9",
    ),
    "nord": ThemeSpec(
        id="nord",
        display_name="Nordic Frost",
        qss_filename="nord.qss",
        base_mode="dark",
        primary_color="#88c0d0",
    ),
    "gruvbox": ThemeSpec(
        id="gruvbox",
        display_name="Gruvbox Dark",
        qss_filename="gruvbox.qss",
        base_mode="dark",
        primary_color="#fe8019",
    ),
    "monokai": ThemeSpec(
        id="monokai",
        display_name="Monokai Pro",
        qss_filename="monokai.qss",
        base_mode="dark",
        primary_color="#a6e22e",
    ),
    "solarized_dark": ThemeSpec(
        id="solarized_dark",
        display_name="Solarized Dark",
        qss_filename="solarized_dark.qss",
        base_mode="dark",
        primary_color="#268bd2",
    ),
    "light": ThemeSpec(
        id="light",
        display_name="Clean Light",
        qss_filename="light_styles.qss",
        base_mode="light",
        primary_color="#0078d4",
    ),
    "solarized_light": ThemeSpec(
        id="solarized_light",
        display_name="Solarized Light",
        qss_filename="solarized_light.qss",
        base_mode="light",
        primary_color="#268bd2",
    ),
}

# Files that are legacy aliases so they don't appear twice in auto-discovery
_IGNORED_DUPLICATE_FILES = {"true_dark.qss"}

# Regex to strip broken placeholder image URLs from legacy QSS files
# e.g., `image: url(PLACEHOLDER_FOR_DOWN_ARROW_ICON.png);`
_PLACEHOLDER_URL_RE = re.compile(
    r"image\s*:\s*url\([^)]*PLACEHOLDER_FOR_[^)]*\)\s*;",
    re.IGNORECASE,
)


class ThemeManager(QtCore.QObject):
    """Centralized QSS & pyqtdarktheme stylesheet loader and hot-reloader.

    Combines pyqtdarktheme's vector base layer with VisoMaster's custom QSS
    overrides while patching legacy QRC paths via AssetManager.
    """

    theme_changed = QtCore.Signal(str)  # Emits active theme_id when changed

    _instance: Optional[ThemeManager] = None

    def __init__(
        self,
        styles_dir: Optional[Path] = None,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.styles_dir: Path = (
            styles_dir.resolve()
            if styles_dir is not None
            else Path(__file__).resolve().parent.parent / "styles"
        )
        self._themes: Dict[str, ThemeSpec] = {}
        self._current_theme_id: str = "true_dark"
        self._discover_themes()

    @classmethod
    def instance(cls) -> ThemeManager:
        """Returns the singleton ThemeManager instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Discovery & Inspection
    # ------------------------------------------------------------------
    def _discover_themes(self) -> None:
        """Populates available themes from built-in specs and disk scan."""
        self._themes.clear()

        # 1. Register built-in themes whose .qss file exists on disk
        mapped_files = set(_IGNORED_DUPLICATE_FILES)
        for theme_id, spec in BUILTIN_THEMES.items():
            qss_path = self.styles_dir / spec.qss_filename
            if qss_path.is_file():
                self._themes[theme_id] = spec
                mapped_files.add(spec.qss_filename)

        # 2. Auto-discover any extra user-added .qss files in app/ui/styles/
        if self.styles_dir.is_dir():
            for qss_file in sorted(self.styles_dir.glob("*.qss")):
                if qss_file.name in mapped_files:
                    continue
                custom_id = qss_file.stem.lower()
                display_name = qss_file.stem.replace("_", " ").title()
                is_light = "light" in custom_id
                self._themes[custom_id] = ThemeSpec(
                    id=custom_id,
                    display_name=display_name,
                    qss_filename=qss_file.name,
                    base_mode="light" if is_light else "dark",
                    primary_color="#0078d4" if is_light else "#4090a3",
                )

        if not self._themes:
            logger.warning("No QSS theme files found in %s", self.styles_dir)

    def available_themes(self) -> List[ThemeSpec]:
        """Returns the ordered list of available ThemeSpecs."""
        return list(self._themes.values())

    @property
    def current_theme_id(self) -> str:
        """Returns the ID of the currently active theme."""
        return self._current_theme_id

    def get_theme_spec(self, theme_id: str) -> Optional[ThemeSpec]:
        """Returns the ThemeSpec for a given theme_id, or None if not found."""
        return self._themes.get(theme_id)

    # ------------------------------------------------------------------
    # Stylesheet Compilation & Application
    # ------------------------------------------------------------------
    @staticmethod
    def sanitize_qss(raw_qss: str) -> str:
        """Patches legacy QRC paths and removes broken placeholder URLs."""
        patched = AssetManager.patch_qss(raw_qss)
        patched = _PLACEHOLDER_URL_RE.sub("/* placeholder icon removed */", patched)
        return patched

    def build_stylesheet(self, theme_id: str = "true_dark") -> str:
        """Builds the composite stylesheet (qdarktheme base + custom QSS)."""
        AssetManager.initialize()

        spec = self._themes.get(theme_id) or self._themes.get("true_dark")
        if spec is None:
            # Fallback if styles folder is missing entirely
            logger.warning("Fallback to pure qdarktheme ('dark'); no QSS found.")
            return qdarktheme.load_stylesheet(
                theme="dark", custom_colors={"primary": "#4090a3"}
            )

        # 1. Load vector base stylesheet from pyqtdarktheme
        try:
            base_qss = qdarktheme.load_stylesheet(
                theme=spec.base_mode,
                custom_colors={"primary": spec.primary_color},
            )
        except Exception as exc:
            logger.error("Failed to load qdarktheme base layer: %s", exc)
            base_qss = ""

        # 2. Load and sanitize the custom VisoMaster QSS overlay
        qss_path = self.styles_dir / spec.qss_filename
        custom_qss = ""
        if qss_path.is_file():
            try:
                raw_content = qss_path.read_text(encoding="utf-8")
                custom_qss = self.sanitize_qss(raw_content)
            except Exception as exc:
                logger.error("Failed to read QSS file %s: %s", qss_path, exc)

        return f"{base_qss}\n\n/* --- VisoMaster Theme: {spec.display_name} --- */\n{custom_qss}"

    def apply_theme(
        self,
        theme_id: str = "true_dark",
        app: Optional[QtWidgets.QApplication] = None,
    ) -> bool:
        """Applies the requested theme to the running QApplication."""
        target_app = app or QtWidgets.QApplication.instance()
        if not isinstance(target_app, QtWidgets.QApplication):
            logger.error("Cannot apply theme: no active QApplication instance.")
            return False

        if theme_id not in self._themes and "true_dark" in self._themes:
            logger.warning(
                "Unknown theme '%s', falling back to 'true_dark'.", theme_id
            )
            theme_id = "true_dark"

        stylesheet = self.build_stylesheet(theme_id)
        target_app.setStyleSheet(stylesheet)

        self._current_theme_id = theme_id
        logger.info("Applied UI theme: %s", theme_id)
        self.theme_changed.emit(theme_id)
        return True

    def reload_current_theme(self) -> bool:
        """Re-reads the active .qss file from disk and reapplies it (live reload)."""
        return self.apply_theme(self._current_theme_id)

    # ------------------------------------------------------------------
    # Menu Integration Helper
    # ------------------------------------------------------------------
    def populate_theme_menu(
        self,
        parent_menu: QtWidgets.QMenu,
        parent_window: Optional[QtWidgets.QWidget] = None,
    ) -> QtWidgets.QMenu:
        """Creates a checkable 'Theme' submenu with live switching and Ctrl+R reload."""
        theme_menu = QtWidgets.QMenu("Theme", parent_menu)
        theme_menu.setObjectName("menuTheme")

        action_group = QtGui.QActionGroup(parent_window or theme_menu)
        action_group.setExclusive(True)

        actions_by_id: Dict[str, QtGui.QAction] = {}

        for spec in self.available_themes():
            action = QtGui.QAction(spec.display_name, parent_window or theme_menu)
            action.setCheckable(True)
            action.setData(spec.id)
            action.setChecked(spec.id == self._current_theme_id)
            action.triggered.connect(
                lambda checked, tid=spec.id: self.apply_theme(tid)
            )
            action_group.addAction(action)
            theme_menu.addAction(action)
            actions_by_id[spec.id] = action

        # Keep menu checkmarks synced if theme is changed programmatically
        def _sync_checked(active_id: str) -> None:
            if active_id in actions_by_id:
                actions_by_id[active_id].setChecked(True)

        self.theme_changed.connect(_sync_checked)

        theme_menu.addSeparator()
        reload_action = QtGui.QAction(
            "Reload Current Theme (Hot-Reload QSS)", parent_window or theme_menu
        )
        reload_action.setShortcut(QtGui.QKeySequence("Ctrl+Shift+R"))
        reload_action.triggered.connect(self.reload_current_theme)
        theme_menu.addAction(reload_action)

        parent_menu.addMenu(theme_menu)
        return theme_menu