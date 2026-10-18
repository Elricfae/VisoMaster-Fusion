# app/ui/widgets/card_manager.py
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from app.ui.core.theme_manager import ThemeManager
from app.ui.widgets.sortable_widgets import SortableDataRole, SortableListWidget


class CardLayoutMode(str, Enum):
    """Layout aspect ratio mode for thumbnail cards."""

    WIDESCREEN_16_9 = "16:9"  # For Target Media (videos, photos)
    SQUARE_1_1 = "1:1"        # For Face Panels (input faces, embeddings)


class CardSizePreset(str, Enum):
    """Supported thumbnail card size presets across media panels."""

    SMALL = "Small"
    MEDIUM = "Medium"
    LARGE = "Large"


@dataclass(frozen=True)
class CardDimensions:
    """Immutable geometry specification for a media card size preset."""

    label: str
    icon_width: int
    icon_height: int
    padding_x: int
    padding_top: int
    text_height: int
    spacing: int = 6

    @property
    def icon_qsize(self) -> QtCore.QSize:
        return QtCore.QSize(self.icon_width, self.icon_height)

    @property
    def grid_qsize(self) -> QtCore.QSize:
        width = self.icon_width + (self.padding_x * 2)
        height = self.icon_height + (self.padding_top) + self.text_height + 6
        return QtCore.QSize(width, height)


# 16:9 Widescreen Presets (Zero vertical gaps for videos/images)
WIDESCREEN_16_9_PRESETS: Dict[CardSizePreset, CardDimensions] = {
    CardSizePreset.SMALL: CardDimensions(
        label="Small",
        icon_width=120,
        icon_height=68,
        padding_x=8,
        padding_top=6,
        text_height=34,
        spacing=4,
    ),
    CardSizePreset.MEDIUM: CardDimensions(
        label="Medium",
        icon_width=160,
        icon_height=90,
        padding_x=8,
        padding_top=6,
        text_height=36,
        spacing=6,
    ),
    CardSizePreset.LARGE: CardDimensions(
        label="Large",
        icon_width=224,
        icon_height=126,
        padding_x=10,
        padding_top=8,
        text_height=38,
        spacing=8,
    ),
}

# 1:1 Square Presets (For Face & Embedding cards)
SQUARE_1_1_PRESETS: Dict[CardSizePreset, CardDimensions] = {
    CardSizePreset.SMALL: CardDimensions(
        label="Small",
        icon_width=64,
        icon_height=64,
        padding_x=8,
        padding_top=6,
        text_height=34,
        spacing=4,
    ),
    CardSizePreset.MEDIUM: CardDimensions(
        label="Medium",
        icon_width=96,
        icon_height=96,
        padding_x=8,
        padding_top=6,
        text_height=36,
        spacing=6,
    ),
    CardSizePreset.LARGE: CardDimensions(
        label="Large",
        icon_width=140,
        icon_height=140,
        padding_x=10,
        padding_top=8,
        text_height=38,
        spacing=8,
    ),
}

# Custom data role storing the raw high-res QPixmap
RAW_THUMBNAIL_PIXMAP_ROLE: int = int(Qt.ItemDataRole.UserRole) + 50


@dataclass(frozen=True)
class CardThemeTokens:
    """Resolved QSS color & border tokens extracted from the active ThemeManager stylesheet."""

    has_native_item_qss: bool = True
    bg_normal: QtGui.QColor = field(default_factory=lambda: QtGui.QColor("#1a1a1a"))
    bg_hover: QtGui.QColor = field(default_factory=lambda: QtGui.QColor("#1f1f1f"))
    bg_selected: QtGui.QColor = field(default_factory=lambda: QtGui.QColor("#274d57"))
    border_normal: QtGui.QColor = field(default_factory=lambda: QtGui.QColor("#363636"))
    border_hover: QtGui.QColor = field(default_factory=lambda: QtGui.QColor("#4090a3"))
    border_selected: QtGui.QColor = field(default_factory=lambda: QtGui.QColor("#4090a3"))
    text_normal: QtGui.QColor = field(default_factory=lambda: QtGui.QColor("#f0f0f0"))
    text_selected: QtGui.QColor = field(default_factory=lambda: QtGui.QColor("#ffffff"))
    border_radius: float = 5.0
    border_width_normal: float = 1.0
    border_width_selected: float = 2.0

    @classmethod
    def from_active_theme(cls, list_object_name: str = "") -> CardThemeTokens:
        """Extract card styling tokens from the active QApplication / ThemeManager stylesheet."""
        app = QtWidgets.QApplication.instance()
        qss = app.styleSheet() if isinstance(app, QtWidgets.QApplication) else ""

        spec = ThemeManager.instance().get_theme_spec(
            ThemeManager.instance().current_theme_id
        )
        is_light = spec.base_mode == "light" if spec else False
        primary = spec.primary_color if spec else "#4090a3"

        bg_normal = "#f5f5f5" if is_light else "#1a1a1a"
        bg_hover = "#eaeaea" if is_light else "#1f1f1f"
        bg_selected = "#d0e8f8" if is_light else "#274d57"
        border_normal = "#d0d0d0" if is_light else "#363636"
        border_hover = primary
        border_selected = primary
        text_normal = "#1a1a1a" if is_light else "#f0f0f0"
        text_selected = "#000000" if is_light else "#ffffff"
        radius = 5.0
        bw_normal = 1.0
        bw_selected = 2.0

        has_native_item_qss = False
        if qss:
            if list_object_name and f"#{list_object_name}::item" in qss:
                has_native_item_qss = True
            elif "QListWidget::item" in qss or "QListView::item" in qss:
                has_native_item_qss = True

            normal_block = cls._extract_qss_block(
                qss,
                (
                    f"#{list_object_name}::item" if list_object_name else "",
                    "#targetVideosList::item",
                    "TargetMediaCardButton",
                ),
                pseudo="",
            )
            if normal_block:
                bg_normal = cls._parse_prop(normal_block, "background-color") or bg_normal
                text_normal = cls._parse_prop(normal_block, "color") or text_normal
                b_w, b_col = cls._parse_border(normal_block)
                if b_col:
                    border_normal = b_col
                if b_w is not None:
                    bw_normal = b_w
                rad = cls._parse_px(normal_block, "border-radius")
                if rad is not None:
                    radius = rad

            hover_block = cls._extract_qss_block(
                qss,
                (
                    f"#{list_object_name}::item" if list_object_name else "",
                    "#targetVideosList::item",
                    "TargetMediaCardButton",
                ),
                pseudo=":hover",
            )
            if hover_block:
                bg_hover = cls._parse_prop(hover_block, "background-color") or bg_hover
                _, b_col = cls._parse_border(hover_block)
                if b_col:
                    border_hover = b_col

            sel_block = cls._extract_qss_block(
                qss,
                (
                    f"#{list_object_name}::item" if list_object_name else "",
                    "#targetVideosList::item",
                ),
                pseudo=":selected",
            ) or cls._extract_qss_block(
                qss,
                ("TargetMediaCardButton",),
                pseudo=":checked",
            )
            if sel_block:
                bg_selected = cls._parse_prop(sel_block, "background-color") or bg_selected
                text_selected = cls._parse_prop(sel_block, "color") or text_selected
                b_w, b_col = cls._parse_border(sel_block)
                if b_col:
                    border_selected = b_col
                if b_w is not None:
                    bw_selected = b_w

        return cls(
            has_native_item_qss=has_native_item_qss,
            bg_normal=QtGui.QColor(bg_normal),
            bg_hover=QtGui.QColor(bg_hover),
            bg_selected=QtGui.QColor(bg_selected),
            border_normal=QtGui.QColor(border_normal),
            border_hover=QtGui.QColor(border_hover),
            border_selected=QtGui.QColor(border_selected),
            text_normal=QtGui.QColor(text_normal),
            text_selected=QtGui.QColor(text_selected),
            border_radius=radius,
            border_width_normal=bw_normal,
            border_width_selected=bw_selected,
        )

    @staticmethod
    def _extract_qss_block(
        qss: str, selectors: tuple[str, ...], pseudo: str = ""
    ) -> str:
        matched_body = ""
        for raw_sel in selectors:
            if not raw_sel:
                continue
            target = f"{raw_sel}{pseudo}"
            pattern = re.compile(
                r"(?:^|[\s,}])"
                + re.escape(target)
                + r"(?=\s*[,{])[^{]*\{([^}]*)\}",
                re.IGNORECASE | re.MULTILINE,
            )
            for match in pattern.finditer(qss):
                matched_body = match.group(1)
        return matched_body

    @staticmethod
    def _parse_prop(block: str, prop_name: str) -> Optional[str]:
        m = re.search(
            rf"(?:^|;|\s){re.escape(prop_name)}\s*:\s*([^;}}]+)",
            block,
            re.IGNORECASE,
        )
        return m.group(1).strip() if m else None

    @classmethod
    def _parse_px(cls, block: str, prop_name: str) -> Optional[float]:
        val = cls._parse_prop(block, prop_name)
        if not val:
            return None
        m = re.search(r"([0-9.]+)\s*px", val, re.IGNORECASE)
        return float(m.group(1)) if m else None

    @classmethod
    def _parse_border(cls, block: str) -> tuple[Optional[float], Optional[str]]:
        val = cls._parse_prop(block, "border")
        if not val:
            return None, None
        w_match = re.search(r"([0-9.]+)\s*px", val, re.IGNORECASE)
        c_match = re.search(r"(#[0-9a-fA-F]{3,8}|rgba?\([^)]+\))", val)
        width = float(w_match.group(1)) if w_match else None
        color = c_match.group(1) if c_match else None
        return width, color


class MediaCardDelegate(QtWidgets.QStyledItemDelegate):
    """Theme-aware item delegate rendering both independent collapsible folder headers
    (with live loading progress bar) and uniform media cards.
    """

    FOLDER_HEADER_HEIGHT: int = 30

    def __init__(
        self,
        dimensions: CardDimensions,
        tokens: CardThemeTokens,
        show_text: bool = True,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._dims = dimensions
        self._tokens = tokens
        self._show_text = show_text

    def set_dimensions(self, dimensions: CardDimensions) -> None:
        self._dims = dimensions

    def set_theme_tokens(self, tokens: CardThemeTokens) -> None:
        self._tokens = tokens

    def set_show_text(self, show_text: bool) -> None:
        self._show_text = show_text

    def sizeHint(
        self,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> QtCore.QSize:
        if bool(index.data(SortableDataRole.IsFolderHeaderRole)):
            parent_widget = self.parent()
            if isinstance(parent_widget, QtWidgets.QAbstractScrollArea):
                vw = parent_widget.viewport().width()
            elif option.widget is not None:
                vw = option.widget.width()
            else:
                vw = 300
            width = max(120, vw - (self._dims.spacing * 2) - 4)
            return QtCore.QSize(width, self.FOLDER_HEADER_HEIGHT)

        if not self._show_text:
            width = self._dims.icon_width + (self._dims.padding_x * 2)
            height = self._dims.icon_height + (self._dims.padding_top * 2)
            return QtCore.QSize(width, height)
        return self._dims.grid_qsize

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> None:
        if bool(index.data(SortableDataRole.IsFolderHeaderRole)):
            self._paint_folder_header(painter, option, index)
            return

        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)

        opt = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)

        opt.text = ""
        opt.icon = QtGui.QIcon()
        opt.features &= ~QtWidgets.QStyleOptionViewItem.ViewItemFeature.HasDisplay
        opt.features &= ~QtWidgets.QStyleOptionViewItem.ViewItemFeature.HasDecoration
        opt.state &= ~QtWidgets.QStyle.StateFlag.State_HasFocus

        is_selected = bool(opt.state & QtWidgets.QStyle.StateFlag.State_Selected)
        is_hovered = bool(opt.state & QtWidgets.QStyle.StateFlag.State_MouseOver)

        widget = opt.widget
        style = widget.style() if widget else QtWidgets.QApplication.style()

        # 1. Draw Card Background & Border via QSS (or fallback)
        if self._tokens.has_native_item_qss and style is not None:
            style.drawControl(
                QtWidgets.QStyle.ControlElement.CE_ItemViewItem,
                opt,
                painter,
                widget,
            )
        else:
            self._paint_fallback_card_box(painter, opt.rect, is_selected, is_hovered)

        rect = opt.rect.adjusted(2, 2, -2, -2)

        # 2. Draw Thumbnail inside Fixed Box
        icon_w = self._dims.icon_width
        icon_h = self._dims.icon_height
        icon_x = rect.x() + (rect.width() - icon_w) // 2
        icon_y = (
            rect.y() + self._dims.padding_top
            if self._show_text
            else rect.y() + (rect.height() - icon_h) // 2
        )
        icon_rect = QtCore.QRect(icon_x, icon_y, icon_w, icon_h)

        pixmap = index.data(RAW_THUMBNAIL_PIXMAP_ROLE)
        if not isinstance(pixmap, QtGui.QPixmap) or pixmap.isNull():
            icon = index.data(Qt.ItemDataRole.DecorationRole)
            if isinstance(icon, QtGui.QIcon) and not icon.isNull():
                pixmap = icon.pixmap(icon_w, icon_h)

        if isinstance(pixmap, QtGui.QPixmap) and not pixmap.isNull():
            scaled = pixmap.scaled(
                icon_rect.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            px = icon_rect.x() + (icon_rect.width() - scaled.width()) // 2
            py = icon_rect.y() + (icon_rect.height() - scaled.height()) // 2
            painter.drawPixmap(px, py, scaled)

        # 3. Draw 2-Line Wrapped Text using Theme Colors
        if self._show_text:
            raw_text = str(index.data(Qt.ItemDataRole.DisplayRole) or "")
            if raw_text:
                text_rect = QtCore.QRect(
                    rect.x() + 4,
                    icon_rect.bottom() + 4,
                    rect.width() - 8,
                    self._dims.text_height - 2,
                )
                text_color = (
                    self._tokens.text_selected
                    if is_selected
                    else self._tokens.text_normal
                )
                painter.setPen(text_color)

                font = opt.font
                if font.pointSize() > 0:
                    font.setPointSize(max(8, font.pointSize() - 1))
                painter.setFont(font)

                self._draw_two_line_wrapped_text(painter, text_rect, raw_text, font)

        painter.restore()

    def _paint_folder_header(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> None:
        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

        is_hovered = bool(option.state & QtWidgets.QStyle.StateFlag.State_MouseOver)
        expanded = bool(index.data(SortableDataRole.FolderExpandedRole))
        vis_count = int(index.data(SortableDataRole.FolderVisibleCountRole) or 0)
        tot_count = int(index.data(SortableDataRole.FolderTotalCountRole) or 0)
        raw_progress = index.data(SortableDataRole.FolderProgressRole)
        progress = -1.0 if raw_progress is None else float(raw_progress)
        raw_label = str(index.data(Qt.ItemDataRole.DisplayRole) or "")

        bar_rect = option.rect.adjusted(2, 2, -2, -2)

        t = self._tokens
        bg_color = t.bg_hover if is_hovered else t.bg_normal
        border_color = t.border_hover if is_hovered else t.border_normal

        path = QtGui.QPainterPath()
        path.addRoundedRect(QtCore.QRectF(bar_rect), 4.0, 4.0)
        painter.fillPath(path, bg_color)

        # Draw Colored Progress Bar Background when folder is actively loading (0.0 <= progress < 1.0)
        if 0.0 <= progress < 1.0:
            clamped = max(0.0, min(1.0, progress))
            fill_w = int(round(bar_rect.width() * clamped))
            if fill_w > 0:
                painter.save()
                painter.setClipPath(path)
                fill_rect = QtCore.QRect(
                    bar_rect.left(),
                    bar_rect.top(),
                    fill_w,
                    bar_rect.height(),
                )
                prog_color = QtGui.QColor(t.border_selected)
                prog_color.setAlpha(95 if not is_hovered else 120)
                painter.fillRect(fill_rect, prog_color)

                # Bottom accent progress indicator line
                accent_color = QtGui.QColor(t.border_selected)
                accent_color.setAlpha(210)
                painter.fillRect(
                    QtCore.QRect(
                        bar_rect.left(),
                        bar_rect.bottom() - 2,
                        fill_w,
                        2,
                    ),
                    accent_color,
                )
                painter.restore()

        painter.setPen(QtGui.QPen(border_color, 1.0))
        painter.drawPath(path)

        # Draw Expand / Collapse Chevron
        chev_cx = bar_rect.left() + 12
        chev_cy = bar_rect.center().y()
        chev_path = QtGui.QPainterPath()
        if expanded:
            chev_path.moveTo(chev_cx - 4, chev_cy - 2)
            chev_path.lineTo(chev_cx + 4, chev_cy - 2)
            chev_path.lineTo(chev_cx, chev_cy + 3)
        else:
            chev_path.moveTo(chev_cx - 2, chev_cy - 4)
            chev_path.lineTo(chev_cx + 3, chev_cy)
            chev_path.lineTo(chev_cx - 2, chev_cy + 4)
        chev_path.closeSubpath()
        painter.fillPath(chev_path, t.text_normal)

        # Draw Count Badge ("shown / total") on the right
        count_str = f"{vis_count} / {tot_count}"
        badge_font = QtGui.QFont(option.font)
        if badge_font.pointSize() > 0:
            badge_font.setPointSize(max(8, badge_font.pointSize() - 1))
        badge_font.setBold(True)
        badge_fm = QtGui.QFontMetrics(badge_font)
        badge_w = badge_fm.horizontalAdvance(count_str) + 14
        badge_h = min(18, bar_rect.height() - 6)
        badge_rect = QtCore.QRect(
            bar_rect.right() - badge_w - 6,
            bar_rect.center().y() - badge_h // 2,
            badge_w,
            badge_h,
        )

        badge_path = QtGui.QPainterPath()
        badge_path.addRoundedRect(QtCore.QRectF(badge_rect), 3.0, 3.0)
        badge_bg = QtGui.QColor(t.border_selected if 0.0 <= progress < 1.0 else t.border_normal)
        badge_bg.setAlpha(140 if 0.0 <= progress < 1.0 else 110)
        painter.fillPath(badge_path, badge_bg)

        painter.setFont(badge_font)
        painter.setPen(t.text_selected if 0.0 <= progress < 1.0 else t.text_normal)
        painter.drawText(badge_rect, int(Qt.AlignmentFlag.AlignCenter), count_str)

        # Draw Full Folder Path on the left (middle-elided so root & leaf folder names stay visible)
        title_font = QtGui.QFont(option.font)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(t.text_selected if is_hovered else t.text_normal)

        text_left = chev_cx + 10
        text_right = badge_rect.left() - 8
        if text_right > text_left:
            text_rect = QtCore.QRect(
                text_left,
                bar_rect.top(),
                text_right - text_left,
                bar_rect.height(),
            )
            title_fm = QtGui.QFontMetrics(title_font)
            elided_label = title_fm.elidedText(
                raw_label, Qt.TextElideMode.ElideMiddle, text_rect.width()
            )
            painter.drawText(
                text_rect,
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                elided_label,
            )

        painter.restore()

    def _paint_fallback_card_box(
        self,
        painter: QtGui.QPainter,
        item_rect: QtCore.QRect,
        is_selected: bool,
        is_hovered: bool,
    ) -> None:
        rect = item_rect.adjusted(1, 1, -1, -1)
        t = self._tokens

        if is_selected:
            bg_color = t.bg_selected
            border_color = t.border_selected
            border_width = t.border_width_selected
        elif is_hovered:
            bg_color = t.bg_hover
            border_color = t.border_hover
            border_width = t.border_width_normal
        else:
            bg_color = t.bg_normal
            border_color = t.border_normal
            border_width = t.border_width_normal

        path = QtGui.QPainterPath()
        path.addRoundedRect(QtCore.QRectF(rect), t.border_radius, t.border_radius)
        painter.fillPath(path, bg_color)

        pen = QtGui.QPen(border_color, border_width)
        painter.setPen(pen)
        painter.drawPath(path)

    @staticmethod
    def _draw_two_line_wrapped_text(
        painter: QtGui.QPainter,
        rect: QtCore.QRect,
        text: str,
        font: QtGui.QFont,
    ) -> None:
        fm = QtGui.QFontMetrics(font)
        max_width = max(16, rect.width())
        line_height = fm.lineSpacing()

        clean_text = text.replace("\u200b", "")

        if fm.horizontalAdvance(clean_text) <= max_width:
            painter.drawText(
                rect,
                int(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter),
                clean_text,
            )
            return

        split_idx = 1
        for i in range(1, len(clean_text) + 1):
            if fm.horizontalAdvance(clean_text[:i]) > max_width:
                split_idx = max(1, i - 1)
                break
        else:
            split_idx = len(clean_text)

        line1 = clean_text[:split_idx]
        remainder = clean_text[split_idx:]
        line2 = fm.elidedText(remainder, Qt.TextElideMode.ElideMiddle, max_width)

        line1_rect = QtCore.QRect(rect.x(), rect.y(), rect.width(), line_height)
        line2_rect = QtCore.QRect(
            rect.x(), rect.y() + line_height, rect.width(), line_height
        )

        painter.drawText(
            line1_rect,
            int(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter),
            line1,
        )
        painter.drawText(
            line2_rect,
            int(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter),
            line2,
        )


class CardManager(QtCore.QObject):
    """Reusable controller managing grid layout, card size presets, QSS theme syncing,
    and thumbnail formatting for any QListWidget / SortableListWidget.
    """

    sizePresetChanged = QtCore.Signal(CardSizePreset)

    # Master canvas resolutions for crisp high-DPI thumbnails
    WIDESCREEN_CANVAS: tuple[int, int] = (320, 180)  # 16:9
    SQUARE_CANVAS: tuple[int, int] = (180, 180)      # 1:1

    def __init__(
        self,
        list_widget: QtWidgets.QListWidget,
        size_combobox: Optional[QtWidgets.QComboBox] = None,
        *,
        layout_mode: CardLayoutMode = CardLayoutMode.WIDESCREEN_16_9,
        default_preset: CardSizePreset = CardSizePreset.MEDIUM,
        show_text: bool = True,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._list_widget = list_widget
        self._size_combobox = size_combobox
        self._layout_mode = layout_mode
        self._current_preset = default_preset
        self._show_text = show_text

        self._presets = (
            WIDESCREEN_16_9_PRESETS
            if layout_mode == CardLayoutMode.WIDESCREEN_16_9
            else SQUARE_1_1_PRESETS
        )

        initial_tokens = CardThemeTokens.from_active_theme(
            self._list_widget.objectName()
        )
        self._delegate = MediaCardDelegate(
            dimensions=self._presets[default_preset],
            tokens=initial_tokens,
            show_text=show_text,
            parent=self._list_widget,
        )

        self._configure_list_widget()
        if self._size_combobox is not None:
            self._configure_combobox(self._size_combobox, default_preset)

        ThemeManager.instance().theme_changed.connect(self._on_theme_changed)
        self.apply_size_preset(default_preset)

    def _configure_list_widget(self) -> None:
        lw = self._list_widget
        grouped = isinstance(lw, SortableListWidget) and lw.groupByFolder()
        lw.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        lw.setFlow(QtWidgets.QListView.Flow.LeftToRight)
        lw.setWrapping(True)
        lw.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        lw.setMovement(QtWidgets.QListView.Movement.Static)
        lw.setUniformItemSizes(not grouped)
        lw.setWordWrap(True)
        lw.setTextElideMode(Qt.TextElideMode.ElideRight)
        lw.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        lw.setVerticalScrollMode(
            QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        lw.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        lw.setLayoutMode(QtWidgets.QListView.LayoutMode.SinglePass)
        lw.viewport().setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        lw.setItemDelegate(self._delegate)

    def _configure_combobox(
        self,
        combobox: QtWidgets.QComboBox,
        default_preset: CardSizePreset,
    ) -> None:
        combobox.blockSignals(True)
        combobox.clear()
        for preset in (
            CardSizePreset.SMALL,
            CardSizePreset.MEDIUM,
            CardSizePreset.LARGE,
        ):
            combobox.addItem(preset.value, preset)
        idx = combobox.findText(default_preset.value)
        if idx >= 0:
            combobox.setCurrentIndex(idx)
        combobox.blockSignals(False)
        combobox.currentTextChanged.connect(self.set_size_by_label)

    @QtCore.Slot(str)
    def _on_theme_changed(self, _theme_id: str) -> None:
        tokens = CardThemeTokens.from_active_theme(self._list_widget.objectName())
        self._delegate.set_theme_tokens(tokens)
        self._list_widget.style().unpolish(self._list_widget)
        self._list_widget.style().polish(self._list_widget)
        self._list_widget.viewport().update()

    @property
    def current_preset(self) -> CardSizePreset:
        return self._current_preset

    @property
    def layout_mode(self) -> CardLayoutMode:
        return self._layout_mode

    @QtCore.Slot(str)
    def set_size_by_label(self, label: str) -> None:
        try:
            preset = CardSizePreset(label)
        except ValueError:
            preset = CardSizePreset.MEDIUM
        self.apply_size_preset(preset)

    def apply_size_preset(self, preset: CardSizePreset) -> None:
        self._current_preset = preset
        dims = self._presets[preset]
        self._delegate.set_dimensions(dims)

        self._delegate.set_theme_tokens(
            CardThemeTokens.from_active_theme(self._list_widget.objectName())
        )

        lw = self._list_widget
        grouped = isinstance(lw, SortableListWidget) and lw.groupByFolder()
        lw.setSpacing(dims.spacing)
        lw.setIconSize(dims.icon_qsize)

        if grouped:
            # Allow full-width folder headers and media card cells to coexist via delegate.sizeHint()
            lw.setUniformItemSizes(False)
            lw.setGridSize(QtCore.QSize())
        else:
            lw.setUniformItemSizes(True)
            if self._show_text:
                lw.setGridSize(dims.grid_qsize)
            else:
                width = dims.icon_width + (dims.padding_x * 2)
                height = dims.icon_height + (dims.padding_top * 2)
                lw.setGridSize(QtCore.QSize(width, height))

        lw.scheduleDelayedItemsLayout()
        lw.viewport().update()
        self.sizePresetChanged.emit(preset)

    @classmethod
    def create_card_pixmap(
        cls,
        raw_pixmap: QtGui.QPixmap,
        layout_mode: CardLayoutMode = CardLayoutMode.WIDESCREEN_16_9,
        crop_to_fill: bool = False,
    ) -> QtGui.QPixmap:
        """Adapts any raw pixmap into a uniform 16:9 or 1:1 master card pixmap."""
        if raw_pixmap.isNull():
            return QtGui.QPixmap()

        canvas_w, canvas_h = (
            cls.WIDESCREEN_CANVAS
            if layout_mode == CardLayoutMode.WIDESCREEN_16_9
            else cls.SQUARE_CANVAS
        )

        if crop_to_fill:
            scale = max(canvas_w / raw_pixmap.width(), canvas_h / raw_pixmap.height())
            scaled_w = max(1, int(round(raw_pixmap.width() * scale)))
            scaled_h = max(1, int(round(raw_pixmap.height() * scale)))

            scaled = raw_pixmap.scaled(
                scaled_w,
                scaled_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

            crop_x = max(0, (scaled.width() - canvas_w) // 2)
            crop_y = max(0, (scaled.height() - canvas_h) // 2)
            return scaled.copy(crop_x, crop_y, canvas_w, canvas_h)
        else:
            canvas = QtGui.QPixmap(canvas_w, canvas_h)
            canvas.fill(Qt.GlobalColor.transparent)

            scaled = raw_pixmap.scaled(
                canvas_w,
                canvas_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

            painter = QtGui.QPainter(canvas)
            painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)
            x = (canvas_w - scaled.width()) // 2
            y = (canvas_h - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
            painter.end()
            return canvas

    @classmethod
    def prepare_list_item(
        cls,
        item: QtWidgets.QListWidgetItem,
        raw_pixmap: QtGui.QPixmap,
        display_name: Optional[str] = None,
        file_path: Optional[str | Path] = None,
        layout_mode: CardLayoutMode = CardLayoutMode.WIDESCREEN_16_9,
        crop_to_fill: bool = False,
    ) -> None:
        """Attach the adapted card pixmap, clean filename, and tooltip to an item."""
        card_pixmap = cls.create_card_pixmap(
            raw_pixmap=raw_pixmap,
            layout_mode=layout_mode,
            crop_to_fill=crop_to_fill,
        )
        item.setData(RAW_THUMBNAIL_PIXMAP_ROLE, card_pixmap)
        item.setIcon(QtGui.QIcon(card_pixmap))

        if display_name is None and file_path is not None:
            display_name = Path(file_path).name

        if display_name is not None:
            item.setText(display_name)
            if not item.toolTip():
                item.setToolTip(str(file_path) if file_path else display_name)