"""
Rust/C++ backend wrapper for MasterChain.
Provides the same API as chain.py but delegates to Rust (via PyO3) or C++ (via pybind11).
If neither is available, the original Python chain.py is used as fallback.

V5.8: Each Proxy keeps a Python fallback module instance so that ui_panel.py can
access .bands, .single_band, .width, .ceiling, etc. directly. The Proxy forwards
known methods to Rust, and falls through to the Python object for everything else.
"""
import os
import logging

logger = logging.getLogger(__name__)

# Detect backend
_BACKEND = "python"  # default fallback
_RUST_BACKEND_AVAILABLE = False

try:
    import longplay
    _BACKEND = "rust"
    _RUST_BACKEND_AVAILABLE = True
    logger.info("Using Rust backend (longplay)")
except ImportError as e:
    try:
        import longplay_cpp
        _BACKEND = "cpp"
        _RUST_BACKEND_AVAILABLE = True
        logger.info("Using C++ backend (longplay_cpp)")
    except ImportError as e2:
        _BACKEND = "python"
        _RUST_BACKEND_AVAILABLE = False
        logger.warning(f"Rust backend unavailable: {e}; C++ backend unavailable: {e2}; using Python fallback")


class _EQProxy:
    """Wraps Rust/C++ EQ methods, delegates attribute access to Python Equalizer."""
    def __init__(self, chain, backend):
        self._c = chain
        self._b = backend
        # Keep a Python-side Equalizer for attribute access (.bands, .preset_mode, etc.)
        from .equalizer import Equalizer
        self._py = Equalizer()

    def __getattr__(self, name):
        # Forward unknown attributes to the Python Equalizer
        return getattr(self._py, name)

    def __setattr__(self, name, value):
        if name.startswith('_') or name in ('_c', '_b', '_py'):
            super().__setattr__(name, value)
        else:
            setattr(self._py, name, value)

    def load_tone_preset(self, name):
        self._py.load_tone_preset(name)
        if self._b == "rust":
            try:
                self._c.eq_load_tone_preset(name)
            except Exception as e:
                logger.warning(f"Rust fallback: load_tone_preset failed: {e}")
        elif self._b == "cpp":
            try:
                self._c.eq_apply_tone_preset(name)
            except Exception as e:
                logger.warning(f"Rust fallback: load_tone_preset (C++) failed: {e}")

    def set_band(self, index, **kwargs):
        self._py.set_band(index, **kwargs)
        if self._b == "rust":
            try:
                band = self._py.bands[index]
                self._c.eq_set_band(index, band.freq, band.gain, band.width)
                self._c.eq_set_band_enabled(index, band.enabled)
            except Exception as e:
                logger.warning(f"Rust fallback: eq_set_band failed: {e}")

    def reset(self):
        self._py.__init__()
        if self._b in ("rust", "cpp"):
            try:
                self._c.eq_reset()
            except Exception as e:
                logger.warning(f"Rust fallback: eq_reset failed: {e}")


class _DynamicsProxy:
    """Wraps Rust/C++ Dynamics methods, delegates to Python Dynamics."""
    def __init__(self, chain, backend):
        self._c = chain
        self._b = backend
        from .dynamics import Dynamics
        self._py = Dynamics()

    def __getattr__(self, name):
        return getattr(self._py, name)

    def __setattr__(self, name, value):
        if name.startswith('_') or name in ('_c', '_b', '_py'):
            super().__setattr__(name, value)
        else:
            setattr(self._py, name, value)

    def set_threshold(self, value):
        self._py.set_threshold(value)
        if self._b == "rust":
            try:
                self._c.dynamics_set_threshold(float(value))
            except Exception as e:
                logger.warning(f"Rust fallback: dynamics_set_threshold failed: {e}")

    def set_ratio(self, value):
        self._py.set_ratio(value)
        if self._b == "rust":
            try:
                self._c.dynamics_set_ratio(float(value))
            except Exception as e:
                logger.warning(f"Rust fallback: dynamics_set_ratio failed: {e}")

    def set_attack(self, value):
        self._py.set_attack(value)
        if self._b == "rust":
            try:
                self._c.dynamics_set_attack(float(value))
            except Exception as e:
                logger.warning(f"Rust fallback: dynamics_set_attack failed: {e}")

    def set_release(self, value):
        self._py.set_release(value)
        if self._b == "rust":
            try:
                self._c.dynamics_set_release(float(value))
            except Exception as e:
                logger.warning(f"Rust fallback: dynamics_set_release failed: {e}")

    def set_makeup_gain(self, value):
        self._py.set_makeup_gain(value)
        if self._b == "rust":
            try:
                self._c.dynamics_set_makeup_gain(float(value))
            except Exception as e:
                logger.warning(f"Rust fallback: dynamics_set_makeup_gain failed: {e}")

    def load_preset(self, name):
        self._py.load_preset(name)
        if self._b in ("rust", "cpp"):
            try:
                self._c.dynamics_apply_preset(name)
            except Exception as e:
                logger.warning(f"Rust fallback: dynamics_apply_preset failed: {e}")

    def reset(self):
        self._py.__init__()
        if self._b in ("rust", "cpp"):
            try:
                self._c.dynamics_reset()
            except Exception as e:
                logger.warning(f"Rust fallback: dynamics_reset failed: {e}")


class _ImagerProxy:
    """Wraps Rust/C++ Imager methods, delegates to Python Imager."""
    def __init__(self, chain, backend):
        self._c = chain
        self._b = backend
        from .imager import Imager
        self._py = Imager()

    def __getattr__(self, name):
        return getattr(self._py, name)

    def __setattr__(self, name, value):
        if name.startswith('_') or name in ('_c', '_b', '_py'):
            super().__setattr__(name, value)
        else:
            setattr(self._py, name, value)
            # Sync width to Rust
            if name == 'width' and self._b in ("rust", "cpp"):
                try:
                    self._c.imager_set_width(value)
                except Exception as e:
                    logger.warning(f"Rust fallback: imager_set_width failed: {e}")

    def set_width(self, w):
        self._py.set_width(w)
        if self._b in ("rust", "cpp"):
            try:
                self._c.imager_set_width(w)
            except Exception as e:
                logger.warning(f"Rust fallback: imager_set_width failed: {e}")

    def load_preset(self, name):
        self._py.load_preset(name)
        if self._b in ("rust", "cpp"):
            try:
                self._c.imager_apply_preset(name)
            except Exception as e:
                logger.warning(f"Rust fallback: imager_apply_preset failed: {e}")

    def reset(self):
        self._py.__init__()
        if self._b in ("rust", "cpp"):
            try:
                self._c.imager_reset()
            except Exception as e:
                logger.warning(f"Rust fallback: imager_reset failed: {e}")


class _MaximizerProxy:
    """Wraps Rust/C++ Maximizer methods, delegates to Python Maximizer."""
    def __init__(self, chain, backend):
        self._c = chain
        self._b = backend
        from .maximizer import Maximizer
        self._py = Maximizer()

    def __getattr__(self, name):
        return getattr(self._py, name)

    def __setattr__(self, name, value):
        if name.startswith('_') or name in ('_c', '_b', '_py'):
            super().__setattr__(name, value)
        else:
            setattr(self._py, name, value)

    def set_gain(self, g):
        self._py.set_gain(g)
        if self._b in ("rust", "cpp"):
            try:
                self._c.maximizer_set_gain(g)
            except Exception as e:
                logger.warning(f"Rust fallback: maximizer_set_gain failed: {e}")

    def set_ceiling(self, c):
        self._py.set_ceiling(c)
        if self._b in ("rust", "cpp"):
            try:
                self._c.maximizer_set_ceiling(c)
            except Exception as e:
                logger.warning(f"Rust fallback: maximizer_set_ceiling failed: {e}")

    def set_character(self, v):
        self._py.set_character(v)
        if self._b == "rust":
            try:
                self._c.maximizer_set_character(int(v))
            except Exception as e:
                logger.warning(f"Rust fallback: maximizer_set_character failed: {e}")

    def set_irc_mode(self, mode, sub_mode=None):
        self._py.set_irc_mode(mode, sub_mode)
        if self._b == "rust":
            try:
                self._c.maximizer_set_irc_mode(mode, sub_mode or "Balanced")
            except Exception as e:
                logger.warning(f"Rust fallback: maximizer_set_irc_mode failed: {e}")
        elif self._b == "cpp":
            mode_map = {"IRC 1": 1, "IRC 2": 2, "IRC 3": 3, "IRC 4": 4, "IRC 5": 5, "IRC LL": 0}
            mode_int = mode_map.get(mode, 3) if isinstance(mode, str) else int(mode)
            try:
                self._c.maximizer_set_irc_mode(mode_int)
                self._c.maximizer_set_sub_mode(sub_mode or "Balanced")
            except Exception as e:
                logger.warning(f"Rust fallback: maximizer_set_irc_mode (C++) failed: {e}")

    def set_irc_sub_mode(self, sub_mode):
        self._py.set_irc_sub_mode(sub_mode)
        if self._b == "rust":
            try:
                # Re-apply IRC mode with new sub-mode
                mode = self._py.irc_mode
                self._c.maximizer_set_irc_mode(mode, sub_mode)
            except Exception as e:
                logger.warning(f"Rust fallback: maximizer_set_irc_mode (sub) failed: {e}")

    def set_upward_compress(self, v):
        self._py.set_upward_compress(v)
        if self._b == "rust":
            try:
                self._c.maximizer_set_upward_compress(float(v))
            except Exception as e:
                logger.warning(f"Rust fallback: maximizer_set_upward_compress failed: {e}")

    def set_soft_clip(self, enabled, pct):
        self._py.set_soft_clip(enabled, pct)
        if self._b == "rust":
            try:
                # Rust expects single float amount (0.0-1.0)
                self._c.maximizer_set_soft_clip(float(pct) / 100.0 if enabled else 0.0)
            except Exception as e:
                logger.warning(f"Rust fallback: maximizer_set_soft_clip failed: {e}")

    def set_transient_emphasis(self, pct, band):
        self._py.set_transient_emphasis(pct, band)
        if self._b == "rust":
            try:
                # Rust expects single float amount (0.0-1.0)
                self._c.maximizer_set_transient_emphasis(float(pct) / 100.0)
            except Exception as e:
                logger.warning(f"Rust fallback: maximizer_set_transient_emphasis failed: {e}")

    def set_stereo_independence(self, transient, sustain):
        self._py.set_stereo_independence(transient, sustain)

    def learn_input_gain(self, audio_path):
        return self._py.learn_input_gain(audio_path)

    def get_learned_lufs(self):
        return self._py.get_learned_lufs()

    def measure_levels(self, audio_path, start=0, duration=10):
        return self._py.measure_levels(audio_path, start, duration)

    def set_tone(self, t):
        self._py.tone = t
        if self._b in ("rust", "cpp"):
            try:
                self._c.maximizer_set_tone(t)
            except Exception as e:
                logger.warning(f"Rust fallback: maximizer_set_tone failed: {e}")

    def reset(self):
        self._py.__init__()
        if self._b in ("rust", "cpp"):
            try:
                self._c.maximizer_reset()
            except Exception as e:
                logger.warning(f"Rust fallback: maximizer_reset failed: {e}")


class MasterChain:
    """
    Drop-in replacement for modules.master.chain.MasterChain.
    Uses Rust → C++ → Python fallback chain.

    V5.8: Keeps Python-side module objects for full API compat with ui_panel.py.
    Rust handles the heavy audio processing; Python objects hold UI state.
    """

    def __init__(self, ffmpeg_path="ffmpeg"):
        self._backend = _BACKEND
        self._ffmpeg = ffmpeg_path
        self._progress_callback = None
        self._meter_callback = None
        self._input_path = None
        self._intensity = 50
        self._target_lufs = -14.0
        self._target_tp = -1.0
        self._normalize_loudness = True
        self._platform = "YouTube"
        self.recommendation = None
        self.input_analysis = None
        self.output_analysis = None
        self.output_path = None

        # Python-side loudness meter
        from .loudness import LoudnessMeter
        self.loudness_meter = LoudnessMeter()

        if self._backend == "rust":
            self._native = longplay.PyMasterChain(ffmpeg_path)
        elif self._backend == "cpp":
            self._native = longplay_cpp.CppMasterChain()
        else:
            raise ImportError("No native backend available")

        # Proxy modules that delegate to both Rust and Python objects
        self.equalizer = _EQProxy(self._native, self._backend)
        self.dynamics = _DynamicsProxy(self._native, self._backend)
        self.imager = _ImagerProxy(self._native, self._backend)
        self.maximizer = _MaximizerProxy(self._native, self._backend)

    @property
    def backend_name(self):
        return self._backend

    @property
    def ffmpeg_path(self):
        return self._ffmpeg

    @property
    def input_path(self):
        return self._input_path

    @property
    def intensity(self):
        return self._intensity

    @intensity.setter
    def intensity(self, value):
        self._intensity = value
        try:
            if self._backend == "rust":
                self._native.set_intensity(value)
            elif self._backend == "cpp":
                self._native.set_intensity(value)
        except Exception as e:
            logger.warning(f"Rust fallback: set_intensity failed: {e}")

    @property
    def target_lufs(self):
        return self._target_lufs

    @target_lufs.setter
    def target_lufs(self, value):
        self._target_lufs = value
        if self._backend == "rust" and self._native:
            try:
                self._native.set_target_lufs(float(value))
            except Exception as e:
                logger.warning(f"Rust fallback: set_target_lufs failed: {e}")

    @property
    def target_tp(self):
        return self._target_tp

    @target_tp.setter
    def target_tp(self, value):
        self._target_tp = value
        if self._backend == "rust" and self._native:
            try:
                self._native.set_target_tp(float(value))
            except Exception as e:
                logger.warning(f"Rust fallback: set_target_tp failed: {e}")

    @property
    def normalize_loudness(self):
        return self._normalize_loudness

    @normalize_loudness.setter
    def normalize_loudness(self, value):
        self._normalize_loudness = value

    @property
    def platform(self):
        return self._platform

    def set_meter_callback(self, callback):
        """Store meter callback for real-time meter updates during processing."""
        self._meter_callback = callback

    def load_audio(self, path):
        self._input_path = path
        if self._backend == "rust":
            return self._native.load_audio(path)
        elif self._backend == "cpp":
            self._native.load_audio(path)
            return True

    # V5.5 FIX: Normalize platform keys between Python (Title Case) and Rust (lowercase)
    _PLATFORM_KEY_MAP = {
        "Spotify": "spotify",
        "Apple Music": "apple_music",
        "YouTube": "youtube",
        "Tidal": "tidal",
        "Amazon": "amazon",
        "Amazon Music": "amazon",
        "SoundCloud": "soundcloud",
        "Radio": "radio",
        "CD": "cd",
        "CD / Digital Release": "cd",
        "Club": "club",
        "Podcast": "podcast",
        "Podcasts": "podcast",
        "Broadcast (EBU R128)": "broadcast",
        "Deezer": "deezer",
        "Vinyl": "vinyl",
        "Custom": "custom",
    }

    def _normalize_platform(self, platform):
        """Convert Python title-case platform names to Rust lowercase keys."""
        return self._PLATFORM_KEY_MAP.get(platform, platform.lower().replace(" ", "_"))

    def set_platform(self, platform):
        self._platform = platform
        # Update targets from genre_profiles
        from .genre_profiles import PLATFORM_TARGETS
        if platform in PLATFORM_TARGETS:
            t = PLATFORM_TARGETS[platform]
            self._target_lufs = t["target_lufs"]
            self._target_tp = t["true_peak"]
        normalized = self._normalize_platform(platform)
        try:
            self._native.set_platform(normalized)
        except Exception as e:
            logger.warning(f"Rust fallback: set_platform failed: {e}")

    def set_intensity(self, intensity):
        self.intensity = intensity

    def set_genre(self, genre):
        """Apply genre preset to all modules."""
        from .genre_profiles import GENRE_PROFILES
        if genre in GENRE_PROFILES:
            profile = GENRE_PROFILES[genre]
            self._target_lufs = profile.get("target_lufs", -14.0)

    def _get_python_chain(self):
        """Lazy-load Python chain.py as fallback for Rust failures."""
        if not hasattr(self, '_py_chain'):
            from .chain import MasterChain as PyChain
            self._py_chain = PyChain(self._ffmpeg)
        return self._py_chain

    def _sync_settings_to_python(self):
        """Sync current settings from Proxy objects to Python chain for fallback."""
        py = self._get_python_chain()
        py.intensity = self._intensity
        py.target_lufs = self._target_lufs
        py.target_tp = self._target_tp
        py.normalize_loudness = self._normalize_loudness
        if self._input_path:
            py.input_path = self._input_path
        # Sync module settings from proxy Python objects
        py.equalizer = self.equalizer._py
        py.dynamics = self.dynamics._py
        py.imager = self.imager._py
        py.maximizer = self.maximizer._py
        if self._meter_callback:
            py.set_meter_callback(self._meter_callback)
        return py

    def ai_recommend(self, genre=None, platform=None, intensity=None):
        norm_platform = self._normalize_platform(platform or "YouTube")
        norm_genre = (genre or "Pop").lower()
        # Try Rust first
        try:
            if self._backend == "rust":
                # Rust uses 0-100 scale like Python
                return self._native.ai_recommend(norm_genre, norm_platform, intensity or 50.0)
            elif self._backend == "cpp":
                return self._native.ai_recommend(norm_genre, norm_platform, intensity or 50.0)
        except Exception as e:
            print(f"[RUST CHAIN] ai_recommend error: {e}, falling back to Python")
        # Fallback to Python AIAssist
        try:
            if self._input_path:
                from .ai_assist import AIAssist
                ai = AIAssist(self._ffmpeg)
                rec = ai.analyze_and_recommend(
                    self._input_path,
                    genre=genre or "All-Purpose Mastering",
                    platform=platform or "YouTube",
                    intensity=intensity or 50,
                )
                return rec
        except Exception as e2:
            print(f"[RUST CHAIN] Python ai_recommend fallback error: {e2}")
        return None

    def apply_recommendation(self, rec):
        try:
            self._native.apply_recommendation(rec)
        except Exception as e:
            print(f"[RUST CHAIN] apply_recommendation error (non-fatal): {e}")

    def preview(self, start_sec=0.0, duration_sec=10.0, callback=None):
        # Try Rust first
        try:
            if self._backend == "rust":
                result = self._native.preview(start_sec, duration_sec, callback)
                if result:
                    return result
            elif self._backend == "cpp":
                result = self._native.preview(start_sec, duration_sec)
                if result:
                    return result
        except Exception as e:
            print(f"[RUST CHAIN] preview error: {e}, falling back to Python")
        # Fallback to Python chain
        try:
            py = self._sync_settings_to_python()
            result = py.preview(start_sec=start_sec, duration_sec=duration_sec, callback=callback)
            return result
        except Exception as e2:
            print(f"[RUST CHAIN] Python preview fallback error: {e2}")
            return None

    @property
    def progress_callback(self):
        return self._progress_callback

    @progress_callback.setter
    def progress_callback(self, callback):
        self._progress_callback = callback

    def render(self, output_path=None, callback=None):
        if output_path is None:
            if self.output_path:
                output_path = self.output_path
            else:
                output_path = os.path.join(
                    os.path.dirname(self._input_path) if self._input_path else ".",
                    "mastered_output.wav")
        cb = callback or self._progress_callback
        # Try Rust first
        try:
            if self._backend == "rust":
                result = self._native.render(output_path, cb)
            elif self._backend == "cpp":
                result = self._native.render(output_path, cb)
            # Verify output file is not empty (Rust may create header-only WAV)
            if result and os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                self.output_path = output_path
                return output_path
            else:
                print(f"[RUST CHAIN] Rust render produced empty/small file, falling back to Python")
        except Exception as e:
            print(f"[RUST CHAIN] render error: {e}, falling back to Python")
        # Fallback to Python chain
        try:
            py = self._sync_settings_to_python()
            result = py.render(output_path=output_path, callback=cb)
            if result and os.path.exists(result):
                self.output_path = result
                return result
        except Exception as e2:
            print(f"[RUST CHAIN] Python render fallback error: {e2}")
        return None

    def get_ab_comparison(self):
        try:
            return self._native.get_ab_comparison()
        except Exception as e:
            logger.warning(f"Rust fallback: get_ab_comparison failed: {e}")
            return None

    def get_chain_summary(self):
        try:
            if self._backend == "rust":
                return self._native.get_summary()
            elif self._backend == "cpp":
                return self._native.get_chain_summary()
        except Exception as e:
            logger.warning(f"Rust fallback: get_chain_summary failed: {e}")
            return "Rust MasterChain"

    def save_settings(self, filepath):
        try:
            return self._native.save_settings(filepath)
        except Exception as e:
            print(f"[RUST CHAIN] save_settings error: {e}")
            # Fallback: save Python-side state
            import json
            settings = {
                "chain": {"intensity": self._intensity, "target_lufs": self._target_lufs,
                          "target_tp": self._target_tp, "platform": self._platform},
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2)
            return True

    def load_settings(self, filepath):
        try:
            return self._native.load_settings(filepath)
        except Exception as e:
            print(f"[RUST CHAIN] load_settings error: {e}")
            return False

    def reset_all(self):
        self._intensity = 50
        self._target_lufs = -14.0
        self._target_tp = -1.0
        self._normalize_loudness = True
        self._platform = "YouTube"
        self.recommendation = None
        self.input_analysis = None
        self.output_analysis = None
        self.equalizer.reset()
        self.dynamics.reset()
        self.imager.reset()
        self.maximizer.reset()
        try:
            self._native.reset_all()
        except Exception as e:
            logger.warning(f"Rust fallback: reset_all failed: {e}")

    def build_filter_chain(self):
        """Backward compat — returns empty list for Rust backend."""
        return []

    def build_ffmpeg_command(self, input_path, output_path):
        """Backward compat — returns basic ffmpeg command."""
        return ["ffmpeg", "-i", input_path, "-y", output_path]
