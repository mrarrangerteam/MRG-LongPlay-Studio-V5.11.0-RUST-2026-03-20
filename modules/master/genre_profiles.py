"""
LongPlay Studio V5.0 — Genre Profiles Database
Pre-defined mastering parameter sets per music genre.
Each profile contains target values for all mastering modules.
"""

# IRC (Intelligent Release Control) Modes — Ozone 12 Style
# Each IRC mode has sub-modes for fine character control
# Maps to different limiter behavior characteristics via FFmpeg alimiter

IRC_MODES = {
    # ══════════════════════════════════════════════════
    #  IRC MODES — matching iZotope Ozone 12 exactly
    #  6 modes: IRC 1-5 + IRC LL
    #  Sub-modes ONLY on IRC 3 & IRC 4 (3 each: Pumping/Balanced/Crisp)
    # ══════════════════════════════════════════════════

    # --- IRC 1: Transparent peak limiter ---
    # Algorithm: Simple look-ahead brickwall. No spectral weighting.
    # Similar to: Waves L1/L2 transparent, FabFilter Pro-L "Safe"
    "IRC 1": {
        "name": "IRC 1",
        "description": "Transparent — Clean peak limiting with minimal coloration. Best for acoustic/jazz.",
        "attack": 5.0,
        "release": 200,
        "lookahead": 10,
        "knee": 0.5,
        "level_in": 1.0,
        "sub_modes": [],
        "algorithm": "peak_limiter",
    },
    # --- IRC 2: Program-dependent adaptive release ---
    # Algorithm: Dual-envelope follower. Release adapts to signal content.
    # Similar to: FabFilter Pro-L "Modern", Sonnox Oxford Limiter
    "IRC 2": {
        "name": "IRC 2",
        "description": "Adaptive — Program-dependent release for musical results. All-purpose mastering.",
        "attack": 3.0,
        "release": 100,
        "lookahead": 5,
        "knee": 1.0,
        "level_in": 1.0,
        "sub_modes": [],
        "algorithm": "adaptive_release",
    },
    # --- IRC 3: Multi-band frequency-weighted limiter (MOST POPULAR) ---
    # Algorithm: 4-band crossover (120/1k/8k Hz), independent limiter per band.
    # Similar to: Ozone Maximizer default, Waves L3 Multimaximizer
    "IRC 3": {
        "name": "IRC 3",
        "description": "Multi-band — Frequency-weighted limiting, spectral preservation.",
        "attack": 3.0,
        "release": 120,
        "lookahead": 5,
        "knee": 2.0,
        "level_in": 1.05,
        "sub_modes": ["Pumping", "Balanced", "Crisp", "Clipping"],
        "algorithm": "multiband_limiter",
    },
    "IRC 3 - Pumping": {
        "name": "IRC 3 — Pumping",
        "description": "Fast release — audible pump, great for EDM/Dance music.",
        "attack": 3.0,
        "release": 48,
        "lookahead": 5,
        "knee": 4.0,
        "level_in": 1.1,
        "parent": "IRC 3",
    },
    "IRC 3 - Balanced": {
        "name": "IRC 3 — Balanced",
        "description": "Natural release — default for most material. Smooth and musical.",
        "attack": 3.0,
        "release": 120,
        "lookahead": 5,
        "knee": 2.0,
        "level_in": 1.05,
        "parent": "IRC 3",
    },
    "IRC 3 - Crisp": {
        "name": "IRC 3 — Crisp",
        "description": "Transient-preserving — best for acoustic/vocal/jazz. Keeps snap and detail.",
        "attack": 3.0,
        "release": 216,
        "lookahead": 10,
        "knee": 1.5,
        "level_in": 1.05,
        "parent": "IRC 3",
    },
    "IRC 3 - Clipping": {
        "name": "IRC 3 — Clipping",
        "description": "Hard clip — instant attack, no lookahead. Maximum loudness, audible distortion.",
        "attack": 0.0,
        "release": 0,
        "lookahead": 0,
        "knee": 0,
        "level_in": 1.15,
        "parent": "IRC 3",
    },
    # --- IRC 4: Aggressive saturation + limiting ---
    # Algorithm: 3-stage: harmonic saturation → soft clip → brickwall limiter.
    # Similar to: Slate FG-X, Waves L2, FabFilter Pro-L "Aggressive"
    "IRC 4": {
        "name": "IRC 4",
        "description": "Aggressive — Saturation + limiting for maximum loudness with character.",
        "attack": 0.5,
        "release": 50,
        "lookahead": 3,
        "knee": 3.0,
        "level_in": 1.1,
        "sub_modes": ["Classic", "Modern", "Transient"],
        "algorithm": "aggressive_saturate",
    },
    "IRC 4 - Classic": {
        "name": "IRC 4 — Classic",
        "description": "Classic warm limiting — soft knee, multiband, musical compression.",
        "attack": 2.0,
        "release": 150,
        "lookahead": 2,
        "knee": 6.0,
        "level_in": 1.1,
        "parent": "IRC 4",
    },
    "IRC 4 - Modern": {
        "name": "IRC 4 — Modern",
        "description": "Modern clean limiting — balanced transients, transparent loudness.",
        "attack": 1.0,
        "release": 100,
        "lookahead": 1.5,
        "knee": 4.0,
        "level_in": 1.12,
        "parent": "IRC 4",
    },
    "IRC 4 - Transient": {
        "name": "IRC 4 — Transient",
        "description": "Transient-preserving — slow attack preserves punch and snap.",
        "attack": 5.0,
        "release": 80,
        "lookahead": 3,
        "knee": 4.0,
        "level_in": 1.1,
        "parent": "IRC 4",
    },
    # --- IRC 5: Maximum Density (NEW - Ozone 12) ---
    # Algorithm: 4-band compression (upward + downward) → 4-band limiting.
    # Similar to: No direct equivalent — Ozone 12 exclusive
    "IRC 5": {
        "name": "IRC 5",
        "description": "Maximum Density — Multi-band compression + limiting, loudest possible.",
        "attack": 2.0,
        "release": 100,
        "lookahead": 5,
        "knee": 6.0,
        "level_in": 1.2,
        "sub_modes": [],
        "algorithm": "maximum_density",
    },
    # --- IRC LL: Low Latency ---
    # Algorithm: Zero-lookahead feedback limiter. Very fast attack.
    # Similar to: Waves L1+ low-latency, Plugin Alliance bx_limiter
    "IRC LL": {
        "name": "IRC LL",
        "description": "Low Latency — Zero-lookahead for real-time monitoring. Fast response.",
        "attack": 0.05,
        "release": 30,
        "lookahead": 0,
        "knee": 1.5,
        "level_in": 1.0,
        "sub_modes": [],
        "algorithm": "feedback_limiter",
    },
}

# Quick lookup: top-level IRC modes (for UI dropdown)
IRC_TOP_MODES = ["IRC 1", "IRC 2", "IRC 3", "IRC 4", "IRC 5", "IRC LL"]

# Tone Presets for Maximizer
TONE_PRESETS = {
    "Transparent": {
        "description": "No tonal coloration",
        "pre_eq": {},  # no pre-EQ
    },
    "Warm": {
        "description": "Gentle low-end warmth, smooth highs",
        "pre_eq": {
            "low_shelf": {"freq": 200, "gain": 1.5, "type": "lowshelf"},
            "high_shelf": {"freq": 8000, "gain": -1.0, "type": "highshelf"},
        },
    },
    "Bright": {
        "description": "Enhanced presence and air",
        "pre_eq": {
            "presence": {"freq": 3000, "gain": 1.5, "width": 1.5, "type": "equalizer"},
            "air": {"freq": 12000, "gain": 2.0, "type": "highshelf"},
        },
    },
    "Punchy": {
        "description": "Enhanced low-mid punch, slight high boost",
        "pre_eq": {
            "punch": {"freq": 100, "gain": 2.0, "width": 0.8, "type": "equalizer"},
            "presence": {"freq": 4000, "gain": 1.0, "width": 2.0, "type": "equalizer"},
        },
    },
    "Analog": {
        "description": "Vintage warmth, gentle saturation character",
        "pre_eq": {
            "sub_roll": {"freq": 30, "gain": -2.0, "type": "highpass"},
            "warmth": {"freq": 250, "gain": 1.0, "width": 1.0, "type": "equalizer"},
            "air_roll": {"freq": 14000, "gain": -1.5, "type": "highshelf"},
        },
    },
}

# Genre Mastering Profiles
GENRE_PROFILES = {
    # === Electronic ===
    "EDM": {
        "category": "Electronic",
        "target_lufs": -9.0,
        "true_peak_ceiling": -0.3,
        "irc_mode": "IRC 4",
        "tone": "Punchy",
        "eq": {
            "bands": [
                {"freq": 40, "gain": 2.0, "width": 0.7, "type": "equalizer"},
                {"freq": 150, "gain": -1.0, "width": 1.0, "type": "equalizer"},
                {"freq": 1000, "gain": -0.5, "width": 2.0, "type": "equalizer"},
                {"freq": 5000, "gain": 1.5, "width": 1.5, "type": "equalizer"},
                {"freq": 12000, "gain": 2.0, "type": "highshelf"},
            ],
        },
        "compressor": {
            "threshold": -12, "ratio": 4.0, "attack": 5, "release": 50,
            "makeup": 3.0,
        },
        "stereo_width": 130,  # percentage (100 = original)
        "intensity_default": 70,
    },
    "House": {
        "category": "Electronic",
        "target_lufs": -10.0,
        "true_peak_ceiling": -0.3,
        "irc_mode": "IRC 3",
        "tone": "Warm",
        "eq": {
            "bands": [
                {"freq": 60, "gain": 2.5, "width": 0.6, "type": "equalizer"},
                {"freq": 300, "gain": -1.0, "width": 1.5, "type": "equalizer"},
                {"freq": 3000, "gain": 1.0, "width": 2.0, "type": "equalizer"},
                {"freq": 10000, "gain": 1.5, "type": "highshelf"},
            ],
        },
        "compressor": {
            "threshold": -14, "ratio": 3.0, "attack": 10, "release": 80,
            "makeup": 2.0,
        },
        "stereo_width": 120,
        "intensity_default": 60,
    },
    "Techno": {
        "category": "Electronic",
        "target_lufs": -9.0,
        "true_peak_ceiling": -0.3,
        "irc_mode": "IRC 4",
        "tone": "Punchy",
        "eq": {
            "bands": [
                {"freq": 50, "gain": 3.0, "width": 0.5, "type": "equalizer"},
                {"freq": 200, "gain": -1.5, "width": 1.0, "type": "equalizer"},
                {"freq": 2000, "gain": 0.5, "width": 2.0, "type": "equalizer"},
                {"freq": 8000, "gain": 1.0, "width": 2.0, "type": "equalizer"},
            ],
        },
        "compressor": {
            "threshold": -10, "ratio": 5.0, "attack": 3, "release": 40,
            "makeup": 4.0,
        },
        "stereo_width": 110,
        "intensity_default": 75,
    },
    "Dubstep": {
        "category": "Electronic",
        "target_lufs": -8.0,
        "true_peak_ceiling": -0.1,
        "irc_mode": "IRC 5",
        "tone": "Punchy",
        "eq": {
            "bands": [
                {"freq": 35, "gain": 3.0, "width": 0.5, "type": "equalizer"},
                {"freq": 100, "gain": 2.0, "width": 0.8, "type": "equalizer"},
                {"freq": 500, "gain": -2.0, "width": 1.0, "type": "equalizer"},
                {"freq": 3000, "gain": 2.0, "width": 1.5, "type": "equalizer"},
                {"freq": 10000, "gain": 1.5, "type": "highshelf"},
            ],
        },
        "compressor": {
            "threshold": -8, "ratio": 6.0, "attack": 2, "release": 30,
            "makeup": 5.0,
        },
        "stereo_width": 140,
        "intensity_default": 85,
    },
    "Future Bass": {
        "category": "Electronic",
        "target_lufs": -9.0,
        "true_peak_ceiling": -0.3,
        "irc_mode": "IRC 4",
        "tone": "Bright",
        "eq": {
            "bands": [
                {"freq": 60, "gain": 2.0, "width": 0.7, "type": "equalizer"},
                {"freq": 400, "gain": -1.0, "width": 1.5, "type": "equalizer"},
                {"freq": 2500, "gain": 2.0, "width": 1.5, "type": "equalizer"},
                {"freq": 8000, "gain": 2.5, "type": "highshelf"},
            ],
        },
        "compressor": {
            "threshold": -12, "ratio": 3.5, "attack": 8, "release": 60,
            "makeup": 3.0,
        },
        "stereo_width": 135,
        "intensity_default": 70,
    },
    "Electropop": {
        "category": "Electronic",
        "target_lufs": -10.0,
        "true_peak_ceiling": -0.3,
        "irc_mode": "IRC 3",
        "tone": "Bright",
        "eq": {
            "bands": [
                {"freq": 80, "gain": 1.5, "width": 0.8, "type": "equalizer"},
                {"freq": 500, "gain": -0.5, "width": 2.0, "type": "equalizer"},
                {"freq": 3000, "gain": 1.5, "width": 2.0, "type": "equalizer"},
                {"freq": 10000, "gain": 2.0, "type": "highshelf"},
            ],
        },
        "compressor": {
            "threshold": -14, "ratio": 3.0, "attack": 10, "release": 80,
            "makeup": 2.0,
        },
        "stereo_width": 125,
        "intensity_default": 60,
    },
    "Hyperpop": {
        "category": "Electronic",
        "target_lufs": -7.0,
        "true_peak_ceiling": -0.1,
        "irc_mode": "IRC 5",
        "tone": "Bright",
        "eq": {
            "bands": [
                {"freq": 50, "gain": 3.0, "width": 0.5, "type": "equalizer"},
                {"freq": 800, "gain": -2.0, "width": 1.0, "type": "equalizer"},
                {"freq": 3000, "gain": 3.0, "width": 1.5, "type": "equalizer"},
                {"freq": 12000, "gain": 3.0, "type": "highshelf"},
            ],
        },
        "compressor": {
            "threshold": -6, "ratio": 8.0, "attack": 1, "release": 20,
            "makeup": 6.0,
        },
        "stereo_width": 150,
        "intensity_default": 90,
    },
    "Dance Pop": {
        "category": "Electronic",
        "target_lufs": -9.0,
        "true_peak_ceiling": -0.3,
        "irc_mode": "IRC 3",
        "tone": "Warm",
        "eq": {
            "bands": [
                {"freq": 60, "gain": 2.0, "width": 0.7, "type": "equalizer"},
                {"freq": 300, "gain": -0.5, "width": 1.5, "type": "equalizer"},
                {"freq": 2000, "gain": 1.0, "width": 2.0, "type": "equalizer"},
                {"freq": 10000, "gain": 1.5, "type": "highshelf"},
            ],
        },
        "compressor": {
            "threshold": -14, "ratio": 3.0, "attack": 8, "release": 60,
            "makeup": 2.5,
        },
        "stereo_width": 120,
        "intensity_default": 65,
    },

    # === Rock ===
    "Rock": {
        "category": "Rock",
        "target_lufs": -11.0,
        "true_peak_ceiling": -0.5,
        "irc_mode": "IRC 3",
        "tone": "Punchy",
        "eq": {
            "bands": [
                {"freq": 80, "gain": 1.5, "width": 0.8, "type": "equalizer"},
                {"freq": 300, "gain": -1.0, "width": 1.5, "type": "equalizer"},
                {"freq": 2500, "gain": 1.5, "width": 2.0, "type": "equalizer"},
                {"freq": 8000, "gain": 1.0, "type": "highshelf"},
            ],
        },
        "compressor": {
            "threshold": -16, "ratio": 3.0, "attack": 10, "release": 100,
            "makeup": 2.0,
        },
        "stereo_width": 110,
        "intensity_default": 55,
    },
    "Classic Rock": {
        "category": "Rock",
        "target_lufs": -12.0,
        "true_peak_ceiling": -0.5,
        "irc_mode": "IRC 2",
        "tone": "Analog",
        "eq": {
            "bands": [
                {"freq": 100, "gain": 1.0, "width": 1.0, "type": "equalizer"},
                {"freq": 500, "gain": -0.5, "width": 2.0, "type": "equalizer"},
                {"freq": 3000, "gain": 1.0, "width": 2.0, "type": "equalizer"},
                {"freq": 10000, "gain": -0.5, "type": "highshelf"},
            ],
        },
        "compressor": {
            "threshold": -18, "ratio": 2.5, "attack": 15, "release": 120,
            "makeup": 1.5,
        },
        "stereo_width": 105,
        "intensity_default": 45,
    },
    "Alt Rock": {
        "category": "Rock",
        "target_lufs": -11.0,
        "true_peak_ceiling": -0.5,
        "irc_mode": "IRC 3",
        "tone": "Warm",
        "eq": {
            "bands": [
                {"freq": 80, "gain": 1.5, "width": 0.8, "type": "equalizer"},
                {"freq": 400, "gain": -1.0, "width": 1.5, "type": "equalizer"},
                {"freq": 2000, "gain": 1.0, "width": 2.0, "type": "equalizer"},
                {"freq": 6000, "gain": 1.5, "width": 2.0, "type": "equalizer"},
            ],
        },
        "compressor": {
            "threshold": -15, "ratio": 3.0, "attack": 10, "release": 80,
            "makeup": 2.0,
        },
        "stereo_width": 115,
        "intensity_default": 55,
    },
    "Indie Rock": {
        "category": "Rock",
        "target_lufs": -12.0,
        "true_peak_ceiling": -1.0,
        "irc_mode": "IRC 2",
        "tone": "Warm",
        "eq": {
            "bands": [
                {"freq": 100, "gain": 1.0, "width": 1.0, "type": "equalizer"},
                {"freq": 800, "gain": 0.5, "width": 2.0, "type": "equalizer"},
                {"freq": 3000, "gain": 1.0, "width": 2.0, "type": "equalizer"},
                {"freq": 10000, "gain": 0.5, "type": "highshelf"},
            ],
        },
        "compressor": {
            "threshold": -18, "ratio": 2.5, "attack": 15, "release": 100,
            "makeup": 1.5,
        },
        "stereo_width": 115,
        "intensity_default": 45,
    },
    "Post Rock": {
        "category": "Rock",
        "target_lufs": -14.0,
        "true_peak_ceiling": -1.0,
        "irc_mode": "IRC 1",
        "tone": "Warm",
        "eq": {
            "bands": [
                {"freq": 60, "gain": 1.0, "width": 0.8, "type": "equalizer"},
                {"freq": 300, "gain": -0.5, "width": 2.0, "type": "equalizer"},
                {"freq": 2000, "gain": 0.5, "width": 3.0, "type": "equalizer"},
                {"freq": 8000, "gain": 1.0, "type": "highshelf"},
            ],
        },
        "compressor": {
            "threshold": -20, "ratio": 2.0, "attack": 20, "release": 150,
            "makeup": 1.0,
        },
        "stereo_width": 130,
        "intensity_default": 35,
    },
    "Punk Rock": {
        "category": "Rock",
        "target_lufs": -10.0,
        "true_peak_ceiling": -0.3,
        "irc_mode": "IRC 4",
        "tone": "Punchy",
        "eq": {
            "bands": [
                {"freq": 80, "gain": 2.0, "width": 0.7, "type": "equalizer"},
                {"freq": 400, "gain": -1.5, "width": 1.0, "type": "equalizer"},
                {"freq": 2500, "gain": 2.0, "width": 1.5, "type": "equalizer"},
                {"freq": 6000, "gain": 1.5, "width": 2.0, "type": "equalizer"},
            ],
        },
        "compressor": {
            "threshold": -12, "ratio": 4.0, "attack": 5, "release": 50,
            "makeup": 3.0,
        },
        "stereo_width": 105,
        "intensity_default": 75,
    },
    "Metal": {
        "category": "Rock",
        "target_lufs": -9.0,
        "true_peak_ceiling": -0.3,
        "irc_mode": "IRC 4",
        "tone": "Punchy",
        "eq": {
            "bands": [
                {"freq": 60, "gain": 2.5, "width": 0.6, "type": "equalizer"},
                {"freq": 300, "gain": -2.0, "width": 1.0, "type": "equalizer"},
                {"freq": 800, "gain": -1.0, "width": 1.5, "type": "equalizer"},
                {"freq": 3000, "gain": 2.0, "width": 1.5, "type": "equalizer"},
                {"freq": 8000, "gain": 1.5, "type": "highshelf"},
            ],
        },
        "compressor": {
            "threshold": -10, "ratio": 5.0, "attack": 3, "release": 40,
            "makeup": 4.0,
        },
        "stereo_width": 115,
        "intensity_default": 80,
    },

    # === Pop ===
    "Pop": {
        "category": "Pop",
        "target_lufs": -10.0,
        "true_peak_ceiling": -0.3,
        "irc_mode": "IRC 3",
        "tone": "Bright",
        "eq": {
            "bands": [
                {"freq": 80, "gain": 1.5, "width": 0.8, "type": "equalizer"},
                {"freq": 250, "gain": -1.0, "width": 1.5, "type": "equalizer"},
                {"freq": 2000, "gain": 1.0, "width": 2.0, "type": "equalizer"},
                {"freq": 5000, "gain": 1.5, "width": 2.0, "type": "equalizer"},
                {"freq": 12000, "gain": 1.5, "type": "highshelf"},
            ],
        },
        "compressor": {
            "threshold": -14, "ratio": 3.0, "attack": 10, "release": 80,
            "makeup": 2.5,
        },
        "stereo_width": 120,
        "intensity_default": 60,
    },
    "K-pop/J-pop": {
        "category": "Pop",
        "target_lufs": -9.0,
        "true_peak_ceiling": -0.3,
        "irc_mode": "IRC 4",
        "tone": "Bright",
        "eq": {
            "bands": [
                {"freq": 60, "gain": 2.0, "width": 0.7, "type": "equalizer"},
                {"freq": 300, "gain": -1.0, "width": 1.5, "type": "equalizer"},
                {"freq": 2500, "gain": 2.0, "width": 1.5, "type": "equalizer"},
                {"freq": 8000, "gain": 2.0, "width": 2.0, "type": "equalizer"},
                {"freq": 14000, "gain": 2.0, "type": "highshelf"},
            ],
        },
        "compressor": {
            "threshold": -12, "ratio": 4.0, "attack": 5, "release": 50,
            "makeup": 3.0,
        },
        "stereo_width": 130,
        "intensity_default": 75,
    },
    "Pop Country": {
        "category": "Pop",
        "target_lufs": -11.0,
        "true_peak_ceiling": -0.5,
        "irc_mode": "IRC 2",
        "tone": "Warm",
        "eq": {
            "bands": [
                {"freq": 100, "gain": 1.0, "width": 1.0, "type": "equalizer"},
                {"freq": 400, "gain": -0.5, "width": 2.0, "type": "equalizer"},
                {"freq": 2000, "gain": 1.0, "width": 2.0, "type": "equalizer"},
                {"freq": 5000, "gain": 1.0, "width": 2.0, "type": "equalizer"},
            ],
        },
        "compressor": {
            "threshold": -16, "ratio": 2.5, "attack": 15, "release": 100,
            "makeup": 1.5,
        },
        "stereo_width": 110,
        "intensity_default": 50,
    },
    "Latin Pop": {
        "category": "Pop",
        "target_lufs": -10.0,
        "true_peak_ceiling": -0.3,
        "irc_mode": "IRC 3",
        "tone": "Warm",
        "eq": {
            "bands": [
                {"freq": 80, "gain": 2.0, "width": 0.7, "type": "equalizer"},
                {"freq": 300, "gain": -0.5, "width": 1.5, "type": "equalizer"},
                {"freq": 1500, "gain": 1.0, "width": 2.0, "type": "equalizer"},
                {"freq": 5000, "gain": 1.0, "width": 2.0, "type": "equalizer"},
            ],
        },
        "compressor": {
            "threshold": -14, "ratio": 3.0, "attack": 8, "release": 70,
            "makeup": 2.5,
        },
        "stereo_width": 115,
        "intensity_default": 60,
    },

    # === Hip-Hop / Urban ===
    "Hip-Hop": {
        "category": "Hip-Hop",
        "target_lufs": -10.0,
        "true_peak_ceiling": -0.3,
        "irc_mode": "IRC 4",
        "tone": "Punchy",
        "eq": {
            "bands": [
                {"freq": 50, "gain": 3.0, "width": 0.5, "type": "equalizer"},
                {"freq": 200, "gain": -1.5, "width": 1.0, "type": "equalizer"},
                {"freq": 3000, "gain": 1.5, "width": 2.0, "type": "equalizer"},
                {"freq": 10000, "gain": 1.0, "type": "highshelf"},
            ],
        },
        "compressor": {
            "threshold": -12, "ratio": 4.0, "attack": 5, "release": 50,
            "makeup": 3.0,
        },
        "stereo_width": 115,
        "intensity_default": 70,
    },
    "Classic Hip-Hop": {
        "category": "Hip-Hop",
        "target_lufs": -11.0,
        "true_peak_ceiling": -0.5,
        "irc_mode": "IRC 3",
        "tone": "Analog",
        "eq": {
            "bands": [
                {"freq": 60, "gain": 2.0, "width": 0.7, "type": "equalizer"},
                {"freq": 300, "gain": -1.0, "width": 1.5, "type": "equalizer"},
                {"freq": 1000, "gain": 0.5, "width": 2.0, "type": "equalizer"},
                {"freq": 5000, "gain": 0.5, "width": 2.0, "type": "equalizer"},
            ],
        },
        "compressor": {
            "threshold": -15, "ratio": 3.0, "attack": 10, "release": 80,
            "makeup": 2.0,
        },
        "stereo_width": 105,
        "intensity_default": 55,
    },
    "Trap": {
        "category": "Hip-Hop",
        "target_lufs": -8.0,
        "true_peak_ceiling": -0.1,
        "irc_mode": "IRC 5",
        "tone": "Punchy",
        "eq": {
            "bands": [
                {"freq": 35, "gain": 4.0, "width": 0.4, "type": "equalizer"},
                {"freq": 150, "gain": -2.0, "width": 1.0, "type": "equalizer"},
                {"freq": 3000, "gain": 2.0, "width": 1.5, "type": "equalizer"},
                {"freq": 8000, "gain": 2.0, "type": "highshelf"},
            ],
        },
        "compressor": {
            "threshold": -8, "ratio": 6.0, "attack": 2, "release": 25,
            "makeup": 5.0,
        },
        "stereo_width": 120,
        "intensity_default": 85,
    },
    "RnB/Soul": {
        "category": "Hip-Hop",
        "target_lufs": -12.0,
        "true_peak_ceiling": -1.0,
        "irc_mode": "IRC 2",
        "tone": "Warm",
        "eq": {
            "bands": [
                {"freq": 80, "gain": 1.5, "width": 0.8, "type": "equalizer"},
                {"freq": 300, "gain": -0.5, "width": 2.0, "type": "equalizer"},
                {"freq": 2000, "gain": 0.5, "width": 2.0, "type": "equalizer"},
                {"freq": 10000, "gain": 0.5, "type": "highshelf"},
            ],
        },
        "compressor": {
            "threshold": -18, "ratio": 2.0, "attack": 15, "release": 120,
            "makeup": 1.0,
        },
        "stereo_width": 120,
        "intensity_default": 40,
    },
    "Reggaeton": {
        "category": "Hip-Hop",
        "target_lufs": -9.0,
        "true_peak_ceiling": -0.3,
        "irc_mode": "IRC 4",
        "tone": "Warm",
        "eq": {
            "bands": [
                {"freq": 60, "gain": 3.0, "width": 0.6, "type": "equalizer"},
                {"freq": 200, "gain": -1.0, "width": 1.5, "type": "equalizer"},
                {"freq": 2000, "gain": 1.0, "width": 2.0, "type": "equalizer"},
                {"freq": 8000, "gain": 1.0, "type": "highshelf"},
            ],
        },
        "compressor": {
            "threshold": -12, "ratio": 4.0, "attack": 5, "release": 50,
            "makeup": 3.0,
        },
        "stereo_width": 110,
        "intensity_default": 70,
    },

    # === Acoustic / Traditional ===
    "Jazz": {
        "category": "Acoustic",
        "target_lufs": -16.0,
        "true_peak_ceiling": -1.0,
        "irc_mode": "IRC 1",
        "tone": "Transparent",
        "eq": {
            "bands": [
                {"freq": 80, "gain": 0.5, "width": 1.0, "type": "equalizer"},
                {"freq": 500, "gain": -0.5, "width": 2.0, "type": "equalizer"},
                {"freq": 3000, "gain": 0.5, "width": 3.0, "type": "equalizer"},
            ],
        },
        "compressor": {
            "threshold": -22, "ratio": 1.5, "attack": 25, "release": 200,
            "makeup": 0.5,
        },
        "stereo_width": 100,
        "intensity_default": 25,
    },
    "Vocal Jazz": {
        "category": "Acoustic",
        "target_lufs": -15.0,
        "true_peak_ceiling": -1.0,
        "irc_mode": "IRC 1",
        "tone": "Warm",
        "eq": {
            "bands": [
                {"freq": 100, "gain": 0.5, "width": 1.0, "type": "equalizer"},
                {"freq": 400, "gain": -0.5, "width": 2.0, "type": "equalizer"},
                {"freq": 2500, "gain": 1.0, "width": 2.0, "type": "equalizer"},
                {"freq": 8000, "gain": 0.5, "type": "highshelf"},
            ],
        },
        "compressor": {
            "threshold": -20, "ratio": 2.0, "attack": 20, "release": 150,
            "makeup": 1.0,
        },
        "stereo_width": 105,
        "intensity_default": 30,
    },
    "Folk": {
        "category": "Acoustic",
        "target_lufs": -14.0,
        "true_peak_ceiling": -1.0,
        "irc_mode": "IRC 1",
        "tone": "Warm",
        "eq": {
            "bands": [
                {"freq": 100, "gain": 0.5, "width": 1.0, "type": "equalizer"},
                {"freq": 500, "gain": -0.5, "width": 2.0, "type": "equalizer"},
                {"freq": 3000, "gain": 1.0, "width": 2.0, "type": "equalizer"},
                {"freq": 8000, "gain": 0.5, "type": "highshelf"},
            ],
        },
        "compressor": {
            "threshold": -20, "ratio": 2.0, "attack": 20, "release": 150,
            "makeup": 1.0,
        },
        "stereo_width": 100,
        "intensity_default": 30,
    },
    "Country": {
        "category": "Acoustic",
        "target_lufs": -12.0,
        "true_peak_ceiling": -1.0,
        "irc_mode": "IRC 2",
        "tone": "Warm",
        "eq": {
            "bands": [
                {"freq": 100, "gain": 1.0, "width": 1.0, "type": "equalizer"},
                {"freq": 500, "gain": -0.5, "width": 2.0, "type": "equalizer"},
                {"freq": 2500, "gain": 1.0, "width": 2.0, "type": "equalizer"},
                {"freq": 8000, "gain": 0.5, "type": "highshelf"},
            ],
        },
        "compressor": {
            "threshold": -18, "ratio": 2.5, "attack": 15, "release": 120,
            "makeup": 1.5,
        },
        "stereo_width": 105,
        "intensity_default": 40,
    },
    "Reggae": {
        "category": "Acoustic",
        "target_lufs": -12.0,
        "true_peak_ceiling": -1.0,
        "irc_mode": "IRC 2",
        "tone": "Warm",
        "eq": {
            "bands": [
                {"freq": 80, "gain": 2.0, "width": 0.7, "type": "equalizer"},
                {"freq": 400, "gain": -1.0, "width": 1.5, "type": "equalizer"},
                {"freq": 1500, "gain": 1.0, "width": 2.0, "type": "equalizer"},
                {"freq": 5000, "gain": 0.5, "width": 2.0, "type": "equalizer"},
            ],
        },
        "compressor": {
            "threshold": -16, "ratio": 3.0, "attack": 10, "release": 80,
            "makeup": 2.0,
        },
        "stereo_width": 100,
        "intensity_default": 50,
    },
    "Orchestral": {
        "category": "Acoustic",
        "target_lufs": -18.0,
        "true_peak_ceiling": -1.0,
        "irc_mode": "IRC 1",
        "tone": "Transparent",
        "eq": {
            "bands": [
                {"freq": 60, "gain": 0.5, "width": 1.0, "type": "equalizer"},
                {"freq": 300, "gain": -0.5, "width": 2.0, "type": "equalizer"},
                {"freq": 5000, "gain": 0.5, "width": 3.0, "type": "equalizer"},
            ],
        },
        "compressor": {
            "threshold": -24, "ratio": 1.5, "attack": 30, "release": 250,
            "makeup": 0.5,
        },
        "stereo_width": 110,
        "intensity_default": 20,
    },

    # === Ambient / Chill ===
    "Ambient": {
        "category": "Ambient",
        "target_lufs": -18.0,
        "true_peak_ceiling": -1.0,
        "irc_mode": "IRC 1",
        "tone": "Transparent",
        "eq": {
            "bands": [
                {"freq": 60, "gain": 0.5, "width": 0.8, "type": "equalizer"},
                {"freq": 500, "gain": -0.5, "width": 3.0, "type": "equalizer"},
                {"freq": 8000, "gain": 0.5, "type": "highshelf"},
            ],
        },
        "compressor": {
            "threshold": -24, "ratio": 1.5, "attack": 30, "release": 300,
            "makeup": 0.5,
        },
        "stereo_width": 140,
        "intensity_default": 20,
    },
    "LoFi": {
        "category": "Ambient",
        "target_lufs": -14.0,
        "true_peak_ceiling": -1.0,
        "irc_mode": "IRC 1",
        "tone": "Analog",
        "eq": {
            "bands": [
                {"freq": 80, "gain": 1.5, "width": 0.8, "type": "equalizer"},
                {"freq": 500, "gain": 0.5, "width": 2.0, "type": "equalizer"},
                {"freq": 3000, "gain": -1.0, "width": 2.0, "type": "equalizer"},
                {"freq": 12000, "gain": -2.0, "type": "highshelf"},
            ],
        },
        "compressor": {
            "threshold": -18, "ratio": 2.0, "attack": 15, "release": 100,
            "makeup": 1.5,
        },
        "stereo_width": 110,
        "intensity_default": 35,
    },

    # === All-Purpose ===
    "All-Purpose Mastering": {
        "category": "General",
        "target_lufs": -14.0,
        "true_peak_ceiling": -1.0,
        "irc_mode": "IRC 2",
        "tone": "Transparent",
        "eq": {
            "bands": [
                {"freq": 80, "gain": 0.5, "width": 1.0, "type": "equalizer"},
                {"freq": 300, "gain": -0.5, "width": 2.0, "type": "equalizer"},
                {"freq": 3000, "gain": 0.5, "width": 2.0, "type": "equalizer"},
                {"freq": 10000, "gain": 0.5, "type": "highshelf"},
            ],
        },
        "compressor": {
            "threshold": -18, "ratio": 2.0, "attack": 15, "release": 100,
            "makeup": 1.0,
        },
        "stereo_width": 105,
        "intensity_default": 40,
    },

    # === V5.8 D-1: Expanded Genre Presets ===
    "Shoegaze": {
        "category": "Rock", "target_lufs": -10.0, "true_peak_ceiling": -0.5,
        "irc_mode": "IRC 3", "tone": "Warm",
        "eq": {"bands": [{"freq": 200, "gain": 2.0, "width": 1.5, "type": "equalizer"},
                         {"freq": 3000, "gain": -1.5, "width": 2.0, "type": "equalizer"},
                         {"freq": 8000, "gain": 2.0, "type": "highshelf"}]},
        "compressor": {"threshold": -10, "ratio": 3.0, "attack": 20, "release": 150, "makeup": 2.0},
        "stereo_width": 160, "intensity_default": 65,
    },
    "Synthwave": {
        "category": "Electronic", "target_lufs": -10.0, "true_peak_ceiling": -0.5,
        "irc_mode": "IRC 3", "tone": "Warm",
        "eq": {"bands": [{"freq": 80, "gain": 2.5, "width": 0.7, "type": "equalizer"},
                         {"freq": 2000, "gain": 1.0, "width": 2.0, "type": "equalizer"},
                         {"freq": 10000, "gain": 1.5, "type": "highshelf"}]},
        "compressor": {"threshold": -14, "ratio": 3.5, "attack": 8, "release": 60, "makeup": 2.5},
        "stereo_width": 140, "intensity_default": 60,
    },
    "Phonk": {
        "category": "Hip-Hop", "target_lufs": -8.0, "true_peak_ceiling": -0.3,
        "irc_mode": "IRC 4", "tone": "Punchy",
        "eq": {"bands": [{"freq": 50, "gain": 3.0, "width": 0.6, "type": "equalizer"},
                         {"freq": 800, "gain": -2.0, "width": 1.5, "type": "equalizer"},
                         {"freq": 6000, "gain": 1.5, "width": 1.5, "type": "equalizer"}]},
        "compressor": {"threshold": -10, "ratio": 5.0, "attack": 3, "release": 40, "makeup": 4.0},
        "stereo_width": 110, "intensity_default": 80,
    },
    "Drill": {
        "category": "Hip-Hop", "target_lufs": -8.0, "true_peak_ceiling": -0.3,
        "irc_mode": "IRC 4", "tone": "Punchy",
        "eq": {"bands": [{"freq": 60, "gain": 2.5, "width": 0.7, "type": "equalizer"},
                         {"freq": 400, "gain": -1.5, "width": 1.5, "type": "equalizer"},
                         {"freq": 3000, "gain": 1.0, "width": 2.0, "type": "equalizer"}]},
        "compressor": {"threshold": -12, "ratio": 4.5, "attack": 5, "release": 50, "makeup": 3.5},
        "stereo_width": 105, "intensity_default": 75,
    },
    "Lo-Fi Hip Hop": {
        "category": "Hip-Hop", "target_lufs": -14.0, "true_peak_ceiling": -1.0,
        "irc_mode": "IRC 2", "tone": "Warm",
        "eq": {"bands": [{"freq": 100, "gain": 1.5, "width": 1.0, "type": "equalizer"},
                         {"freq": 8000, "gain": -2.0, "type": "highshelf"}]},
        "compressor": {"threshold": -18, "ratio": 2.0, "attack": 20, "release": 150, "makeup": 1.0},
        "stereo_width": 110, "intensity_default": 40,
    },
    "Thai Pop": {
        "category": "Asian", "target_lufs": -12.0, "true_peak_ceiling": -0.5,
        "irc_mode": "IRC 3", "tone": "Bright",
        "eq": {"bands": [{"freq": 100, "gain": 1.0, "width": 1.0, "type": "equalizer"},
                         {"freq": 3000, "gain": 1.5, "width": 2.0, "type": "equalizer"},
                         {"freq": 10000, "gain": 1.0, "type": "highshelf"}]},
        "compressor": {"threshold": -14, "ratio": 3.0, "attack": 10, "release": 80, "makeup": 2.0},
        "stereo_width": 115, "intensity_default": 55,
    },
    "Thai Luk Thung": {
        "category": "Asian", "target_lufs": -12.0, "true_peak_ceiling": -0.5,
        "irc_mode": "IRC 2", "tone": "Warm",
        "eq": {"bands": [{"freq": 200, "gain": 1.5, "width": 1.0, "type": "equalizer"},
                         {"freq": 2000, "gain": 1.0, "width": 2.0, "type": "equalizer"},
                         {"freq": 6000, "gain": 0.5, "width": 2.0, "type": "equalizer"}]},
        "compressor": {"threshold": -16, "ratio": 2.5, "attack": 12, "release": 100, "makeup": 1.5},
        "stereo_width": 100, "intensity_default": 50,
    },
    "Thai Mor Lam": {
        "category": "Asian", "target_lufs": -12.0, "true_peak_ceiling": -0.5,
        "irc_mode": "IRC 2", "tone": "Warm",
        "eq": {"bands": [{"freq": 150, "gain": 1.0, "width": 1.0, "type": "equalizer"},
                         {"freq": 1500, "gain": 1.5, "width": 2.0, "type": "equalizer"}]},
        "compressor": {"threshold": -16, "ratio": 2.5, "attack": 15, "release": 100, "makeup": 1.5},
        "stereo_width": 95, "intensity_default": 45,
    },
    "Thai Isan": {
        "category": "Asian", "target_lufs": -12.0, "true_peak_ceiling": -0.5,
        "irc_mode": "IRC 2", "tone": "Warm",
        "eq": {"bands": [{"freq": 200, "gain": 1.0, "width": 1.0, "type": "equalizer"},
                         {"freq": 2500, "gain": 1.0, "width": 2.0, "type": "equalizer"}]},
        "compressor": {"threshold": -16, "ratio": 2.0, "attack": 15, "release": 120, "makeup": 1.0},
        "stereo_width": 95, "intensity_default": 45,
    },
    "Bollywood": {
        "category": "Asian", "target_lufs": -11.0, "true_peak_ceiling": -0.5,
        "irc_mode": "IRC 3", "tone": "Bright",
        "eq": {"bands": [{"freq": 80, "gain": 1.5, "width": 0.8, "type": "equalizer"},
                         {"freq": 3000, "gain": 2.0, "width": 2.0, "type": "equalizer"},
                         {"freq": 10000, "gain": 1.5, "type": "highshelf"}]},
        "compressor": {"threshold": -12, "ratio": 3.5, "attack": 8, "release": 60, "makeup": 2.5},
        "stereo_width": 120, "intensity_default": 60,
    },
    "Afrobeats": {
        "category": "World", "target_lufs": -10.0, "true_peak_ceiling": -0.5,
        "irc_mode": "IRC 3", "tone": "Punchy",
        "eq": {"bands": [{"freq": 60, "gain": 2.0, "width": 0.7, "type": "equalizer"},
                         {"freq": 500, "gain": -1.0, "width": 1.5, "type": "equalizer"},
                         {"freq": 4000, "gain": 1.5, "width": 2.0, "type": "equalizer"}]},
        "compressor": {"threshold": -12, "ratio": 3.5, "attack": 8, "release": 60, "makeup": 2.5},
        "stereo_width": 115, "intensity_default": 65,
    },
    "Dembow": {
        "category": "Latin", "target_lufs": -8.0, "true_peak_ceiling": -0.3,
        "irc_mode": "IRC 4", "tone": "Punchy",
        "eq": {"bands": [{"freq": 50, "gain": 3.0, "width": 0.6, "type": "equalizer"},
                         {"freq": 3000, "gain": 1.0, "width": 2.0, "type": "equalizer"}]},
        "compressor": {"threshold": -10, "ratio": 5.0, "attack": 3, "release": 40, "makeup": 4.0},
        "stereo_width": 110, "intensity_default": 80,
    },
    "Cumbia": {
        "category": "Latin", "target_lufs": -12.0, "true_peak_ceiling": -0.5,
        "irc_mode": "IRC 2", "tone": "Warm",
        "eq": {"bands": [{"freq": 100, "gain": 1.5, "width": 1.0, "type": "equalizer"},
                         {"freq": 2000, "gain": 1.0, "width": 2.0, "type": "equalizer"}]},
        "compressor": {"threshold": -16, "ratio": 2.5, "attack": 12, "release": 100, "makeup": 1.5},
        "stereo_width": 105, "intensity_default": 50,
    },
    "Bossa Nova": {
        "category": "Jazz", "target_lufs": -16.0, "true_peak_ceiling": -1.0,
        "irc_mode": "IRC 1", "tone": "Transparent",
        "eq": {"bands": [{"freq": 200, "gain": 0.5, "width": 1.5, "type": "equalizer"},
                         {"freq": 5000, "gain": 0.5, "width": 2.0, "type": "equalizer"}]},
        "compressor": {"threshold": -20, "ratio": 1.5, "attack": 25, "release": 200, "makeup": 0.5},
        "stereo_width": 110, "intensity_default": 30,
    },
    "Samba": {
        "category": "Latin", "target_lufs": -12.0, "true_peak_ceiling": -0.5,
        "irc_mode": "IRC 3", "tone": "Warm",
        "eq": {"bands": [{"freq": 80, "gain": 1.5, "width": 0.8, "type": "equalizer"},
                         {"freq": 3000, "gain": 1.0, "width": 2.0, "type": "equalizer"}]},
        "compressor": {"threshold": -14, "ratio": 3.0, "attack": 8, "release": 70, "makeup": 2.0},
        "stereo_width": 110, "intensity_default": 55,
    },
    "Tango": {
        "category": "Latin", "target_lufs": -14.0, "true_peak_ceiling": -1.0,
        "irc_mode": "IRC 2", "tone": "Warm",
        "eq": {"bands": [{"freq": 150, "gain": 1.0, "width": 1.0, "type": "equalizer"},
                         {"freq": 2500, "gain": 1.0, "width": 2.0, "type": "equalizer"}]},
        "compressor": {"threshold": -16, "ratio": 2.0, "attack": 15, "release": 120, "makeup": 1.0},
        "stereo_width": 100, "intensity_default": 40,
    },
    "Drone": {
        "category": "Experimental", "target_lufs": -16.0, "true_peak_ceiling": -1.0,
        "irc_mode": "IRC 1", "tone": "Transparent",
        "eq": {"bands": [{"freq": 60, "gain": 2.0, "width": 0.5, "type": "equalizer"},
                         {"freq": 8000, "gain": -1.0, "type": "highshelf"}]},
        "compressor": {"threshold": -24, "ratio": 1.5, "attack": 50, "release": 300, "makeup": 0.5},
        "stereo_width": 150, "intensity_default": 30,
    },
    "Math Rock": {
        "category": "Rock", "target_lufs": -12.0, "true_peak_ceiling": -0.5,
        "irc_mode": "IRC 3", "tone": "Bright",
        "eq": {"bands": [{"freq": 80, "gain": 1.0, "width": 0.8, "type": "equalizer"},
                         {"freq": 3000, "gain": 1.5, "width": 2.0, "type": "equalizer"},
                         {"freq": 8000, "gain": 1.0, "type": "highshelf"}]},
        "compressor": {"threshold": -14, "ratio": 3.0, "attack": 5, "release": 60, "makeup": 2.0},
        "stereo_width": 115, "intensity_default": 60,
    },
    "Emo": {
        "category": "Rock", "target_lufs": -10.0, "true_peak_ceiling": -0.5,
        "irc_mode": "IRC 3", "tone": "Warm",
        "eq": {"bands": [{"freq": 100, "gain": 1.5, "width": 1.0, "type": "equalizer"},
                         {"freq": 2000, "gain": 1.5, "width": 2.0, "type": "equalizer"},
                         {"freq": 6000, "gain": 1.0, "width": 2.0, "type": "equalizer"}]},
        "compressor": {"threshold": -12, "ratio": 3.5, "attack": 8, "release": 70, "makeup": 2.5},
        "stereo_width": 120, "intensity_default": 60,
    },
    "Screamo": {
        "category": "Rock", "target_lufs": -8.0, "true_peak_ceiling": -0.3,
        "irc_mode": "IRC 4", "tone": "Aggressive",
        "eq": {"bands": [{"freq": 60, "gain": 2.0, "width": 0.7, "type": "equalizer"},
                         {"freq": 2000, "gain": 2.0, "width": 2.0, "type": "equalizer"},
                         {"freq": 8000, "gain": 1.5, "type": "highshelf"}]},
        "compressor": {"threshold": -10, "ratio": 5.0, "attack": 3, "release": 40, "makeup": 4.0},
        "stereo_width": 125, "intensity_default": 80,
    },
    "J-Pop": {
        "category": "Asian", "target_lufs": -10.0, "true_peak_ceiling": -0.5,
        "irc_mode": "IRC 3", "tone": "Bright",
        "eq": {"bands": [{"freq": 80, "gain": 1.0, "width": 0.8, "type": "equalizer"},
                         {"freq": 3000, "gain": 2.0, "width": 2.0, "type": "equalizer"},
                         {"freq": 12000, "gain": 1.5, "type": "highshelf"}]},
        "compressor": {"threshold": -12, "ratio": 3.5, "attack": 8, "release": 60, "makeup": 2.5},
        "stereo_width": 120, "intensity_default": 65,
    },
}


# Platform loudness targets
PLATFORM_TARGETS = {
    "YouTube": {"target_lufs": -14.0, "true_peak": -1.0},
    "Spotify": {"target_lufs": -14.0, "true_peak": -1.0},
    "Apple Music": {"target_lufs": -16.0, "true_peak": -1.0},
    "Amazon Music": {"target_lufs": -14.0, "true_peak": -2.0},
    "Tidal": {"target_lufs": -14.0, "true_peak": -1.0},
    "Deezer": {"target_lufs": -15.0, "true_peak": -1.0},
    "Podcasts": {"target_lufs": -16.0, "true_peak": -1.5},
    "Vinyl": {"target_lufs": -12.0, "true_peak": -0.5},
    "CD / Digital Release": {"target_lufs": -9.0, "true_peak": 0.0},
    "Podcast": {"target_lufs": -16.0, "true_peak": -1.0},
    "Broadcast (EBU R128)": {"target_lufs": -23.0, "true_peak": -1.0},
    "Custom": {"target_lufs": -14.0, "true_peak": -1.0},
}


def get_genre_list():
    """Return sorted list of genre names grouped by category."""
    categories = {}
    for name, profile in GENRE_PROFILES.items():
        cat = profile["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(name)
    return categories


def get_genre_profile(genre_name):
    """Get complete profile for a genre, or All-Purpose if not found."""
    return GENRE_PROFILES.get(genre_name, GENRE_PROFILES["All-Purpose Mastering"])


def get_irc_mode(mode_name):
    """Get IRC mode parameters. Supports both old (IRC II) and new (IRC 2) naming."""
    # Direct lookup
    if mode_name in IRC_MODES:
        return IRC_MODES[mode_name]
    # Legacy name mapping (Roman → Arabic)
    legacy_map = {"IRC I": "IRC 1", "IRC II": "IRC 2", "IRC III": "IRC 3",
                  "IRC IV": "IRC 4", "IRC V": "IRC 5"}
    mapped = legacy_map.get(mode_name)
    if mapped and mapped in IRC_MODES:
        return IRC_MODES[mapped]
    return IRC_MODES["IRC 2"]


def get_irc_sub_modes(mode_name):
    """Get list of sub-mode names for a given IRC mode."""
    mode = IRC_MODES.get(mode_name, {})
    return mode.get("sub_modes", [])


def get_tone_preset(tone_name):
    """Get tone preset parameters."""
    return TONE_PRESETS.get(tone_name, TONE_PRESETS["Transparent"])


# ══════════════════════════════════════════════════════════════
#  MASTERING PRESETS — 15 One-Click Presets
#  Dynamics + Imager + Maximizer ONLY (No EQ — Suno tone is good)
#  Each preset is designed for a specific use case / genre feel.
# ══════════════════════════════════════════════════════════════

MASTERING_PRESETS = {
    # ─── TRANSPARENT / REFERENCE ───
    "Transparent Master": {
        "description": "Clean, transparent mastering — preserves original mix character",
        "category": "Reference",
        "dynamics": {
            "threshold": -18, "ratio": 1.5, "attack": 20, "release": 200,
            "makeup": 1.0, "knee": 6,
        },
        "imager": {"width": 105, "mono_bass_freq": 0},
        "maximizer": {
            "gain_db": 3.0, "ceiling": -1.0, "character": 2.0,
            "irc_mode": "IRC 1", "irc_sub_mode": None,
            "transient_emphasis_pct": 0, "upward_compress_db": 0,
            "soft_clip_enabled": False,
        },
    },

    "Streaming Optimized": {
        "description": "Balanced loudness for Spotify/Apple Music/YouTube (-14 LUFS target)",
        "category": "Reference",
        "dynamics": {
            "threshold": -16, "ratio": 2.0, "attack": 15, "release": 150,
            "makeup": 1.5, "knee": 4,
        },
        "imager": {"width": 115, "mono_bass_freq": 100},
        "maximizer": {
            "gain_db": 5.0, "ceiling": -1.0, "character": 3.0,
            "irc_mode": "IRC 3", "irc_sub_mode": "Balanced",
            "transient_emphasis_pct": 10, "upward_compress_db": 0,
            "soft_clip_enabled": False,
        },
    },

    # ─── POP / MAINSTREAM ───
    "Pop Radio Ready": {
        "description": "Loud, polished, radio-ready pop mastering",
        "category": "Pop",
        "dynamics": {
            "threshold": -12, "ratio": 3.0, "attack": 8, "release": 80,
            "makeup": 3.0, "knee": 3,
        },
        "imager": {"width": 130, "mono_bass_freq": 150},
        "maximizer": {
            "gain_db": 7.5, "ceiling": -0.5, "character": 4.5,
            "irc_mode": "IRC 3", "irc_sub_mode": "Balanced",
            "transient_emphasis_pct": 15, "upward_compress_db": 1.0,
            "soft_clip_enabled": True, "soft_clip_pct": 20,
        },
    },

    "K-Pop Bright": {
        "description": "Bright, wide, punchy — K-Pop / J-Pop style mastering",
        "category": "Pop",
        "dynamics": {
            "threshold": -14, "ratio": 2.5, "attack": 5, "release": 60,
            "makeup": 2.5, "knee": 3,
        },
        "imager": {"width": 145, "mono_bass_freq": 120},
        "maximizer": {
            "gain_db": 7.0, "ceiling": -0.5, "character": 5.0,
            "irc_mode": "IRC 3", "irc_sub_mode": "Crisp",
            "transient_emphasis_pct": 25, "upward_compress_db": 1.5,
            "soft_clip_enabled": False,
        },
    },

    # ─── HIP-HOP / R&B ───
    "Hip-Hop Loud": {
        "description": "Heavy, loud mastering — trap, hip-hop, drill",
        "category": "Hip-Hop",
        "dynamics": {
            "threshold": -10, "ratio": 4.0, "attack": 3, "release": 40,
            "makeup": 4.0, "knee": 2,
        },
        "imager": {"width": 120, "mono_bass_freq": 200},
        "maximizer": {
            "gain_db": 9.0, "ceiling": -0.3, "character": 6.5,
            "irc_mode": "IRC 4", "irc_sub_mode": "Pumping",
            "transient_emphasis_pct": 20, "upward_compress_db": 2.0,
            "soft_clip_enabled": True, "soft_clip_pct": 35,
        },
    },

    "R&B Smooth": {
        "description": "Smooth, warm, controlled — R&B / Neo Soul",
        "category": "Hip-Hop",
        "dynamics": {
            "threshold": -16, "ratio": 2.0, "attack": 15, "release": 120,
            "makeup": 2.0, "knee": 6,
        },
        "imager": {"width": 125, "mono_bass_freq": 100},
        "maximizer": {
            "gain_db": 5.5, "ceiling": -1.0, "character": 2.5,
            "irc_mode": "IRC 2", "irc_sub_mode": None,
            "transient_emphasis_pct": 5, "upward_compress_db": 0.5,
            "soft_clip_enabled": False,
        },
    },

    # ─── ELECTRONIC / EDM ───
    "EDM Banger": {
        "description": "Maximum loudness & energy — EDM, house, techno",
        "category": "Electronic",
        "dynamics": {
            "threshold": -8, "ratio": 5.0, "attack": 1, "release": 30,
            "makeup": 5.0, "knee": 1,
        },
        "imager": {"width": 155, "mono_bass_freq": 180},
        "maximizer": {
            "gain_db": 10.0, "ceiling": -0.3, "character": 7.5,
            "irc_mode": "IRC 4", "irc_sub_mode": "Pumping",
            "transient_emphasis_pct": 10, "upward_compress_db": 3.0,
            "soft_clip_enabled": True, "soft_clip_pct": 50,
        },
    },

    "Lo-Fi Chill": {
        "description": "Warm, compressed, cozy — lo-fi beats, chillhop",
        "category": "Electronic",
        "dynamics": {
            "threshold": -14, "ratio": 3.5, "attack": 25, "release": 250,
            "makeup": 2.5, "knee": 8,
        },
        "imager": {"width": 110, "mono_bass_freq": 80},
        "maximizer": {
            "gain_db": 4.0, "ceiling": -1.5, "character": 2.0,
            "irc_mode": "IRC 2", "irc_sub_mode": None,
            "transient_emphasis_pct": 0, "upward_compress_db": 1.0,
            "soft_clip_enabled": True, "soft_clip_pct": 15,
        },
    },

    # ─── ROCK / METAL ───
    "Rock Punchy": {
        "description": "Punchy, dynamic mastering — rock, alternative, indie",
        "category": "Rock",
        "dynamics": {
            "threshold": -14, "ratio": 3.0, "attack": 10, "release": 60,
            "makeup": 3.0, "knee": 2,
        },
        "imager": {"width": 125, "mono_bass_freq": 120},
        "maximizer": {
            "gain_db": 6.5, "ceiling": -0.5, "character": 5.0,
            "irc_mode": "IRC 3", "irc_sub_mode": "Crisp",
            "transient_emphasis_pct": 30, "upward_compress_db": 0.5,
            "soft_clip_enabled": False,
        },
    },

    "Metal Wall": {
        "description": "Brick-wall loud — metal, hardcore, heavy rock",
        "category": "Rock",
        "dynamics": {
            "threshold": -8, "ratio": 6.0, "attack": 1, "release": 25,
            "makeup": 5.0, "knee": 0,
        },
        "imager": {"width": 135, "mono_bass_freq": 150},
        "maximizer": {
            "gain_db": 10.0, "ceiling": -0.3, "character": 8.0,
            "irc_mode": "IRC 5", "irc_sub_mode": None,
            "transient_emphasis_pct": 15, "upward_compress_db": 3.0,
            "soft_clip_enabled": True, "soft_clip_pct": 40,
        },
    },

    # ─── ACOUSTIC / JAZZ / CLASSICAL ───
    "Acoustic Natural": {
        "description": "Minimal processing — acoustic, folk, singer-songwriter",
        "category": "Acoustic",
        "dynamics": {
            "threshold": -22, "ratio": 1.3, "attack": 30, "release": 300,
            "makeup": 0.5, "knee": 8,
        },
        "imager": {"width": 105, "mono_bass_freq": 0},
        "maximizer": {
            "gain_db": 2.0, "ceiling": -1.5, "character": 1.0,
            "irc_mode": "IRC 1", "irc_sub_mode": None,
            "transient_emphasis_pct": 0, "upward_compress_db": 0,
            "soft_clip_enabled": False,
        },
    },

    "Jazz Open": {
        "description": "Open, dynamic, spacious — jazz, classical, ambient",
        "category": "Acoustic",
        "dynamics": {
            "threshold": -20, "ratio": 1.5, "attack": 25, "release": 250,
            "makeup": 1.0, "knee": 10,
        },
        "imager": {"width": 130, "mono_bass_freq": 0},
        "maximizer": {
            "gain_db": 2.5, "ceiling": -1.0, "character": 1.5,
            "irc_mode": "IRC 1", "irc_sub_mode": None,
            "transient_emphasis_pct": 5, "upward_compress_db": 0,
            "soft_clip_enabled": False,
        },
    },

    # ─── SPECIAL USE CASES ───
    "Vocal Focus": {
        "description": "Vocal-forward mastering — podcasts, vocal covers, spoken word",
        "category": "Special",
        "dynamics": {
            "threshold": -18, "ratio": 2.5, "attack": 10, "release": 100,
            "makeup": 2.0, "knee": 5,
        },
        "imager": {"width": 90, "mono_bass_freq": 0},
        "maximizer": {
            "gain_db": 4.5, "ceiling": -1.0, "character": 3.0,
            "irc_mode": "IRC 2", "irc_sub_mode": None,
            "transient_emphasis_pct": 10, "upward_compress_db": 1.0,
            "soft_clip_enabled": False,
        },
    },

    "TikTok / Reels": {
        "description": "Phone-optimized — loud, punchy, mono-compatible for social media",
        "category": "Special",
        "dynamics": {
            "threshold": -12, "ratio": 3.5, "attack": 5, "release": 50,
            "makeup": 3.5, "knee": 2,
        },
        "imager": {"width": 100, "mono_bass_freq": 200},
        "maximizer": {
            "gain_db": 8.0, "ceiling": -0.5, "character": 5.5,
            "irc_mode": "IRC 3", "irc_sub_mode": "Balanced",
            "transient_emphasis_pct": 20, "upward_compress_db": 2.0,
            "soft_clip_enabled": True, "soft_clip_pct": 25,
        },
    },

    "Maximum Density": {
        "description": "Absolute maximum loudness — competition masters, DJ edits",
        "category": "Special",
        "dynamics": {
            "threshold": -6, "ratio": 8.0, "attack": 0.5, "release": 20,
            "makeup": 6.0, "knee": 0,
        },
        "imager": {"width": 120, "mono_bass_freq": 200},
        "maximizer": {
            "gain_db": 12.0, "ceiling": -0.1, "character": 9.0,
            "irc_mode": "IRC 5", "irc_sub_mode": None,
            "transient_emphasis_pct": 5, "upward_compress_db": 4.0,
            "soft_clip_enabled": True, "soft_clip_pct": 60,
        },
    },
}

MASTERING_PRESET_NAMES = list(MASTERING_PRESETS.keys())

MASTERING_PRESET_CATEGORIES = {
    "Reference": ["Transparent Master", "Streaming Optimized"],
    "Pop": ["Pop Radio Ready", "K-Pop Bright"],
    "Hip-Hop": ["Hip-Hop Loud", "R&B Smooth"],
    "Electronic": ["EDM Banger", "Lo-Fi Chill"],
    "Rock": ["Rock Punchy", "Metal Wall"],
    "Acoustic": ["Acoustic Natural", "Jazz Open"],
    "Special": ["Vocal Focus", "TikTok / Reels", "Maximum Density"],
}


def get_mastering_preset(name):
    """Get a mastering preset by name."""
    return MASTERING_PRESETS.get(name, MASTERING_PRESETS["Streaming Optimized"])
