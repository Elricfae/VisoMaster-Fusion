# app/ui/core/asset_manager.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

from PySide6.QtCore import QDir, QSize
from PySide6.QtGui import QIcon, QPixmap

logger = logging.getLogger(__name__)


class AssetManager:
    """Pure-Python replacement for media.qrc and media_rc.py.

    Registers the 'media:' search path with Qt's file engine so both Python code
    and QSS stylesheets can load PNGs and SVGs directly from disk without compilation.
    """

    _initialized: bool = False
    _media_dir: Path = Path(__file__).resolve().parent / "media"
    _icon_cache: Dict[str, QIcon] = {}
    _pixmap_cache: Dict[str, QPixmap] = {}

    @classmethod
    def initialize(cls, custom_media_dir: Optional[Path] = None) -> None:
        """Registers 'media:' and 'glyphs:' search paths with Qt at runtime."""
        if custom_media_dir is not None:
            cls._media_dir = custom_media_dir.resolve()

        if not cls._media_dir.exists():
            logger.warning("UI Media directory not found at: %s", cls._media_dir)
            return

        QDir.addSearchPath("media", str(cls._media_dir))
        glyphs_dir: Path = cls._media_dir / "glyphs"
        if glyphs_dir.exists():
            QDir.addSearchPath("glyphs", str(glyphs_dir))

        cls._initialized = True
        logger.info("AssetManager initialized -> %s", cls._media_dir)

    @classmethod
    def normalize_legacy_path(cls, resource_path: str) -> str:
        """Converts legacy ':/media/foo.png' paths into 'media:foo.png' or disk paths."""
        if not cls._initialized:
            cls.initialize()

        clean_path: str = resource_path.strip()
        if clean_path.startswith(":/media/"):
            rel_part = clean_path[len(":/media/") :]
            return str((cls._media_dir / rel_part).as_posix())
        if clean_path.startswith("media:"):
            rel_part = clean_path[len("media:") :]
            return str((cls._media_dir / rel_part).as_posix())
        return str((cls._media_dir / clean_path).as_posix())

    @classmethod
    def patch_qss(cls, qss_content: str) -> str:
        """Replaces legacy ':/media/' QRC references in QSS strings with 'media:'."""
        if not cls._initialized:
            cls.initialize()
        return qss_content.replace(":/media/", "media:")

    @classmethod
    def get_icon(cls, filename: str) -> QIcon:
        """Returns a cached QIcon from 'app/ui/core/media/' (supports 'glyphs/...' too)."""
        if not cls._initialized:
            cls.initialize()

        if filename in cls._icon_cache:
            return cls._icon_cache[filename]

        resolved: str = cls.normalize_legacy_path(filename)
        if not Path(resolved).is_file():
            logger.error("Missing icon asset: %s", resolved)
            return QIcon()

        icon = QIcon(resolved)
        cls._icon_cache[filename] = icon
        return icon

    @classmethod
    def get_stateful_icon(
        cls,
        off_file: str,
        on_file: Optional[str] = None,
        disabled_off_file: Optional[str] = None,
        disabled_on_file: Optional[str] = None,
    ) -> QIcon:
        """Builds a multi-state QIcon (Normal/Disabled, On/Off) without QRC."""
        if not cls._initialized:
            cls.initialize()

        cache_key = f"state::{off_file}::{on_file}::{disabled_off_file}::{disabled_on_file}"
        if cache_key in cls._icon_cache:
            return cls._icon_cache[cache_key]

        icon = QIcon()
        icon.addFile(
            cls.normalize_legacy_path(off_file),
            QSize(),
            QIcon.Mode.Normal,
            QIcon.State.Off,
        )
        if on_file:
            icon.addFile(
                cls.normalize_legacy_path(on_file),
                QSize(),
                QIcon.Mode.Normal,
                QIcon.State.On,
            )
        if disabled_off_file:
            icon.addFile(
                cls.normalize_legacy_path(disabled_off_file),
                QSize(),
                QIcon.Mode.Disabled,
                QIcon.State.Off,
            )
        if disabled_on_file:
            icon.addFile(
                cls.normalize_legacy_path(disabled_on_file),
                QSize(),
                QIcon.Mode.Disabled,
                QIcon.State.On,
            )

        cls._icon_cache[cache_key] = icon
        return icon

    @classmethod
    def get_pixmap(cls, filename: str) -> QPixmap:
        """Returns a cached QPixmap from the media directory."""
        if not cls._initialized:
            cls.initialize()

        if filename in cls._pixmap_cache:
            return cls._pixmap_cache[filename]

        resolved: str = cls.normalize_legacy_path(filename)
        if not Path(resolved).is_file():
            logger.error("Missing pixmap asset: %s", resolved)
            return QPixmap()

        pixmap = QPixmap(resolved)
        cls._pixmap_cache[filename] = pixmap
        return pixmap