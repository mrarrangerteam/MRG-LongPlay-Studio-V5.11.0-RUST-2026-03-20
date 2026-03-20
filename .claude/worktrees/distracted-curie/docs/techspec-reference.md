# LongPlay Studio — Technical Specification Reference

## K-Weighting Filter Coefficients (48 kHz, ITU-R BS.1770-4)

### Stage 1: Pre-filter (shelving)
```
b0 =  1.53512485958697
b1 = -2.69169618940638
b2 =  1.19839281085285
a1 = -1.69065929318241
a2 =  0.73248077421585
```

### Stage 2: High-pass (RLB weighting)
```
b0 =  1.0
b1 = -2.0
b2 =  1.0
a1 = -1.99004745483398
a2 =  0.99007225036621
```

### Implementation Notes
- Apply Stage 1 then Stage 2 as cascaded biquad filters
- Use double-precision (float64) for all intermediate calculations
- K-weighted signal is used for LUFS gating, not for output

## LUFS Measurement Windows

| Measurement   | Window   | Overlap | Gate Threshold         |
|---------------|----------|---------|------------------------|
| Momentary     | 400 ms   | 75%     | None (ungated)         |
| Short-term    | 3000 ms  | 75%     | None (ungated)         |
| Integrated    | Full     | N/A     | Absolute: -70 LUFS, Relative: -10 LU below ungated |

### Loudness Range (LRA)
- Measure short-term loudness over full program
- Apply absolute gate (-70 LUFS) then relative gate (-20 LU)
- LRA = difference between 95th and 10th percentile of gated short-term values

## IRC Sub-Modes (Ozone 12 Reference)

### IRC I — Transparent
No sub-modes. Basic look-ahead limiting.

### IRC II — Balanced
No sub-modes. Multi-stage look-ahead with release shaping.

### IRC III — Advanced
| Sub-Mode  | Character                                    |
|-----------|----------------------------------------------|
| Pumping   | Allows natural gain pumping, musical feel     |
| Balanced  | Even-handed limiting (default)                |
| Crisp     | Preserves transients, brighter limiting       |
| Clipping  | Hard clip before limiting, aggressive tone    |

### IRC IV — Modern
| Sub-Mode   | Character                                   |
|------------|---------------------------------------------|
| Classic    | Traditional brickwall behavior              |
| Modern     | Frequency-dependent release (default)       |
| Transient  | Maximum transient preservation              |

### IRC V — Intelligent (LongPlay Custom)
AI-driven adaptive limiting. No sub-modes — auto-adjusts based on content analysis.

## Platform Loudness Targets

| Platform       | Target LUFS | True Peak (dBTP) | Notes                    |
|----------------|-------------|-------------------|--------------------------|
| YouTube        | -14.0       | -1.0              | Normalized down only     |
| Spotify        | -14.0       | -1.0              | ReplayGain + limiter     |
| Apple Music    | -16.0       | -1.0              | Sound Check              |
| Amazon Music   | -14.0       | -2.0              | Conservative TP          |
| Tidal           | -14.0       | -1.0              | MQA passthrough          |
| Deezer         | -15.0       | -1.0              | ReplayGain               |
| Podcasts       | -16.0       | -1.5              | Mono-compatible          |
| Broadcast (EBU)| -23.0       | -1.0              | EBU R128                 |
| CD / Digital   | -9.0        | -0.3              | No normalization         |
| Vinyl          | -12.0       | -0.5              | Dynamic range preserved  |
| Custom         | user        | user              | User-defined             |

## Logic Pro X Channel Strip Meter Specifications

### Visual Design
- Two tall vertical bars (L/R stereo)
- Gradient: green (below -12 dB) → yellow (-12 to -3 dB) → red (-3 to 0 dB)
- Segmented display (2px segments, 1px gap)

### Peak Hold
- White horizontal line
- Hold time: 2000 ms
- Decay rate: 20 dB/sec after hold expires

### Clip Indicator
- Red dot above bar when peak exceeds ceiling
- Sticky until manual reset (click to clear)

### Scale Markings
- dB values: 0, -3, -6, -12, -24, -48
- Displayed on both sides of meter pair

### Update Rate
- 30 fps (33ms QTimer interval)
- Thread-safe meter buffer with lock

## Waves WLM Plus Meter Specifications

### Numeric Displays
- Short-term LUFS (large, primary)
- Integrated LUFS (large)
- Loudness Range in LU (medium)

### LED-Segment Bars
- Momentary LUFS: horizontal bar, -60 to 0 LUFS
- True Peak: horizontal bar, -60 to +3 dBTP
- Gain Reduction: horizontal bar, 0 to -24 dB

### Color Coding
- Green: within target ±2 LU
- Yellow: within target ±5 LU
- Red: outside target ±5 LU or True Peak over ceiling

## OzoneRotaryKnob Specifications

### Visual Design
- Dark circle background (#1A1A1E)
- 270° value arc (135° to 405°, gap at bottom)
- Arc color: teal (#00B4D8) for active, dim (#0077B6) for track
- Center value text (Menlo font, bold)
- Unit label below value
- Parameter name above knob

### Interaction
- Mouse drag: vertical movement maps to value change
- Scroll wheel: fine adjustment (1% of range per tick)
- Double-click: reset to default value
- Shift+drag: fine mode (0.1x sensitivity)

### Sizes
- Standard: 64x64 px
- Large: 80x80 px (for Gain knob)

## Vectorscope Specifications

### Visual Design
- Half-circle (or full circle) display
- X-axis: L-R (stereo difference)
- Y-axis: L+R (stereo sum)
- Phosphor-style dots with decay (green → dim)
- Correlation bar below (-1 to +1)

### Signal Processing
- Lissajous display: plot (L-R, L+R) for each sample
- Downsample to ~4096 points for display performance
- Phosphor decay: 85% retention per frame (alpha blending)

## Transfer Curve Specifications

### Visual Design
- Square plot: input dB (X) vs output dB (Y)
- Range: -60 to 0 dB both axes
- 1:1 reference line (diagonal, dim gray)
- Compression curve: follows threshold/ratio/knee
- Knee region: smooth quadratic interpolation
- Active region highlighted in teal

### Computation
For soft-knee compressor at point x (dB):
```
if x < (threshold - knee/2):
    y = x  (no compression)
elif x > (threshold + knee/2):
    y = threshold + (x - threshold) / ratio
else:
    # Knee region (quadratic interpolation)
    t = (x - threshold + knee/2) / knee
    y = x + (1/ratio - 1) * (x - threshold + knee/2)^2 / (2 * knee)
```
