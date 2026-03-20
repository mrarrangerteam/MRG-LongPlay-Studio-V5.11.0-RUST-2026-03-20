"""
Effects library — 10+ video/audio effects with FFmpeg filter generation.

Story 3.4 — Epic 3: CapCut Features.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from gui.utils.compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QDoubleSpinBox, QCheckBox, QGroupBox,
    QSlider, QComboBox, QScrollArea, QFrame,
    Qt, QSize, QSizePolicy,
    QColor, QFont,
    pyqtSignal,
)

from gui.models.keyframes import KeyframeTrack


# ---------------------------------------------------------------------------
# Effect types
# ---------------------------------------------------------------------------

class EffectType(Enum):
    """Available effect types."""
    BRIGHTNESS = auto()
    CONTRAST = auto()
    SATURATION = auto()
    HUE = auto()
    BLUR = auto()
    SHARPEN = auto()
    VIGNETTE = auto()
    FILM_GRAIN = auto()
    CHROMATIC_ABERRATION = auto()
    COLOR_TEMPERATURE = auto()


# Default parameters per effect type
_EFFECT_DEFAULTS: Dict[EffectType, Dict[str, Any]] = {
    EffectType.BRIGHTNESS: {"brightness": 0.0},           # -1.0 to 1.0
    EffectType.CONTRAST: {"contrast": 1.0},                # 0.0 to 3.0
    EffectType.SATURATION: {"saturation": 1.0},            # 0.0 to 3.0
    EffectType.HUE: {"hue_degrees": 0.0},                  # -180 to 180
    EffectType.BLUR: {"radius": 5.0},                      # 0 to 50
    EffectType.SHARPEN: {"amount": 1.0, "size": 5},        # luma amount + matrix size
    EffectType.VIGNETTE: {"angle": 0.5, "aspect": 1.0},   # PI/5 default
    EffectType.FILM_GRAIN: {"strength": 50, "seed": -1},   # noise strength
    EffectType.CHROMATIC_ABERRATION: {"offset": 4},         # pixel offset
    EffectType.COLOR_TEMPERATURE: {"temperature": 6500},    # Kelvin
}

_EFFECT_LABELS: Dict[EffectType, str] = {
    EffectType.BRIGHTNESS: "Brightness",
    EffectType.CONTRAST: "Contrast",
    EffectType.SATURATION: "Saturation",
    EffectType.HUE: "Hue Shift",
    EffectType.BLUR: "Blur",
    EffectType.SHARPEN: "Sharpen",
    EffectType.VIGNETTE: "Vignette",
    EffectType.FILM_GRAIN: "Film Grain",
    EffectType.CHROMATIC_ABERRATION: "Chromatic Aberration",
    EffectType.COLOR_TEMPERATURE: "Color Temperature",
}


# ---------------------------------------------------------------------------
# Effect model
# ---------------------------------------------------------------------------

@dataclass
class Effect:
    """An effect applied to a clip or track.

    Attributes:
        id:              Unique identifier.
        type:            The effect type.
        parameters:      Current parameter values.
        enabled:         Whether the effect is active.
        keyframe_tracks: Animated parameter tracks (keyed by param name).
    """
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    type: EffectType = EffectType.BRIGHTNESS
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    keyframe_tracks: Dict[str, KeyframeTrack] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Fill in defaults for missing parameters
        defaults = _EFFECT_DEFAULTS.get(self.type, {})
        for k, v in defaults.items():
            self.parameters.setdefault(k, v)

    @property
    def label(self) -> str:
        return _EFFECT_LABELS.get(self.type, self.type.name)

    def get_param_at(self, param: str, time: float) -> Any:
        """Get parameter value at *time*, checking keyframes first."""
        if param in self.keyframe_tracks:
            val = self.keyframe_tracks[param].get_value_at(time)
            if val is not None:
                return val
        return self.parameters.get(param)

    # -- FFmpeg filter generation -------------------------------------------

    def to_ffmpeg_filter(self, time: float = 0.0) -> str:
        """Generate the FFmpeg filter string for this effect."""
        if not self.enabled:
            return ""

        t = self.type

        if t == EffectType.BRIGHTNESS:
            b = self.get_param_at("brightness", time)
            return f"eq=brightness={b:.3f}"

        if t == EffectType.CONTRAST:
            c = self.get_param_at("contrast", time)
            return f"eq=contrast={c:.3f}"

        if t == EffectType.SATURATION:
            s = self.get_param_at("saturation", time)
            return f"eq=saturation={s:.3f}"

        if t == EffectType.HUE:
            h = self.get_param_at("hue_degrees", time)
            return f"hue=h={h:.1f}"

        if t == EffectType.BLUR:
            r = int(self.get_param_at("radius", time))
            # boxblur requires odd luma_radius
            r = max(1, r if r % 2 == 1 else r + 1)
            return f"boxblur={r}:{r}"

        if t == EffectType.SHARPEN:
            amt = self.get_param_at("amount", time)
            sz = int(self.get_param_at("size", time))
            return f"unsharp={sz}:{sz}:{amt:.2f}"

        if t == EffectType.VIGNETTE:
            angle = self.get_param_at("angle", time)
            return f"vignette=angle={angle:.3f}"

        if t == EffectType.FILM_GRAIN:
            strength = int(self.get_param_at("strength", time))
            seed = int(self.get_param_at("seed", time))
            return f"noise=alls={strength}:allf=t+u" if seed < 0 else f"noise=alls={strength}:allf=t+u:seed={seed}"

        if t == EffectType.CHROMATIC_ABERRATION:
            offset = int(self.get_param_at("offset", time))
            # Simulate via channel offset using rgbashift
            return f"rgbashift=rh={offset}:bh=-{offset}"

        if t == EffectType.COLOR_TEMPERATURE:
            temp = int(self.get_param_at("temperature", time))
            # Approximate warm/cool shift via colorbalance
            if temp > 6500:
                shift = min((temp - 6500) / 6500.0, 1.0)
                return f"colorbalance=rs={shift:.2f}:gs=0:bs=-{shift:.2f}"
            elif temp < 6500:
                shift = min((6500 - temp) / 6500.0, 1.0)
                return f"colorbalance=rs=-{shift:.2f}:gs=0:bs={shift:.2f}"
            return ""

        return ""


# ---------------------------------------------------------------------------
# EffectsPanel — widget for managing clip effects
# ---------------------------------------------------------------------------

class EffectsPanel(QWidget):
    """Panel listing applied effects with parameter controls."""

    effect_changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._effects: List[Effect] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Header
        header_row = QHBoxLayout()
        header = QLabel("Effects")
        header.setStyleSheet("color: #FFFFFF; font-size: 14px; font-weight: bold;")
        header_row.addWidget(header)
        header_row.addStretch()

        self._add_combo = QComboBox()
        self._add_combo.addItem("+ Add Effect...")
        for et in EffectType:
            self._add_combo.addItem(_EFFECT_LABELS.get(et, et.name), et)
        self._add_combo.currentIndexChanged.connect(self._on_add_effect)
        header_row.addWidget(self._add_combo)
        layout.addLayout(header_row)

        # Scroll area for effect list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._effects_container = QWidget()
        self._effects_layout = QVBoxLayout(self._effects_container)
        self._effects_layout.setContentsMargins(0, 0, 0, 0)
        self._effects_layout.addStretch()
        scroll.setWidget(self._effects_container)
        layout.addWidget(scroll)

    # -- public API ---------------------------------------------------------

    def set_effects(self, effects: List[Effect]) -> None:
        self._effects = effects
        self._rebuild_ui()

    def get_effects(self) -> List[Effect]:
        return list(self._effects)

    # -- internal -----------------------------------------------------------

    def _rebuild_ui(self) -> None:
        # Clear existing widgets
        while self._effects_layout.count() > 1:
            item = self._effects_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        for effect in self._effects:
            card = self._create_effect_card(effect)
            self._effects_layout.insertWidget(self._effects_layout.count() - 1, card)

    def _create_effect_card(self, effect: Effect) -> QGroupBox:
        card = QGroupBox(effect.label)
        card.setCheckable(True)
        card.setChecked(effect.enabled)
        card.toggled.connect(lambda checked, eff=effect: self._toggle_effect(eff, checked))
        card_layout = QGridLayout(card)

        row = 0
        for param_name, value in effect.parameters.items():
            card_layout.addWidget(QLabel(param_name.replace("_", " ").title()), row, 0)
            spin = QDoubleSpinBox()
            spin.setRange(-1000.0, 10000.0)
            spin.setSingleStep(0.1)
            try:
                spin.setValue(float(value))
            except (TypeError, ValueError):
                spin.setValue(0.0)
            spin.valueChanged.connect(
                lambda val, eff=effect, pn=param_name: self._on_param_changed(eff, pn, val)
            )
            card_layout.addWidget(spin, row, 1)
            row += 1

        # Remove button
        rm_btn = QPushButton("Remove")
        rm_btn.clicked.connect(lambda _checked=False, eff=effect: self._remove_effect(eff))
        card_layout.addWidget(rm_btn, row, 0, 1, 2)

        return card

    def _toggle_effect(self, effect: Effect, enabled: bool) -> None:
        effect.enabled = enabled
        self.effect_changed.emit()

    def _on_param_changed(self, effect: Effect, param: str, value: float) -> None:
        effect.parameters[param] = value
        self.effect_changed.emit()

    def _remove_effect(self, effect: Effect) -> None:
        if effect in self._effects:
            self._effects.remove(effect)
            self._rebuild_ui()
            self.effect_changed.emit()

    def _on_add_effect(self, index: int) -> None:
        if index <= 0:
            return
        et = self._add_combo.itemData(index)
        if et is None:
            self._add_combo.setCurrentIndex(0)
            return
        new_effect = Effect(type=et)
        self._effects.append(new_effect)
        self._rebuild_ui()
        self._add_combo.setCurrentIndex(0)
        self.effect_changed.emit()
