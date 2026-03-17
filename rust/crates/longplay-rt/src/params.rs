//! Lock-free parameter store using atomic floats.
//!
//! Python GUI thread writes parameters via `set_*()`.
//! Audio callback thread reads them via `get_*()`.
//! No locks, no allocations — just atomic loads/stores.

use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering};

/// Atomic f32 using AtomicU32 bit transmute.
#[derive(Debug)]
pub struct AtomicF32 {
    bits: AtomicU32,
}

impl AtomicF32 {
    pub fn new(val: f32) -> Self {
        Self {
            bits: AtomicU32::new(val.to_bits()),
        }
    }

    #[inline]
    pub fn load(&self) -> f32 {
        f32::from_bits(self.bits.load(Ordering::Relaxed))
    }

    #[inline]
    pub fn store(&self, val: f32) {
        self.bits.store(val.to_bits(), Ordering::Relaxed);
    }
}

/// All real-time controllable parameters for the mastering chain.
///
/// Each field is atomic — safe to read from audio thread while
/// Python GUI thread writes new values.
pub struct RtParams {
    // -- Maximizer --
    pub gain_db: AtomicF32,
    pub ceiling_db: AtomicF32,
    pub irc_mode: AtomicI32,

    // -- Imager --
    pub width_pct: AtomicF32,
    pub low_width: AtomicF32,
    pub mid_width: AtomicF32,
    pub high_width: AtomicF32,
    pub imager_multiband: AtomicBool,

    // -- EQ bands (8 bands, gain only for real-time) --
    pub eq_gains: [AtomicF32; 8],
    pub eq_bypass: AtomicBool,

    // -- Limiter --
    pub limiter_ceiling_db: AtomicF32,
    pub limiter_bypass: AtomicBool,

    // -- Transport --
    pub volume: AtomicF32,

    // -- Dirty flag: set by writer, read+cleared by audio thread --
    pub dirty: AtomicBool,
}

impl RtParams {
    pub fn new() -> Self {
        Self {
            gain_db: AtomicF32::new(0.0),
            ceiling_db: AtomicF32::new(-1.0),
            irc_mode: AtomicI32::new(3),

            width_pct: AtomicF32::new(100.0),
            low_width: AtomicF32::new(100.0),
            mid_width: AtomicF32::new(100.0),
            high_width: AtomicF32::new(100.0),
            imager_multiband: AtomicBool::new(false),

            eq_gains: std::array::from_fn(|_| AtomicF32::new(0.0)),
            eq_bypass: AtomicBool::new(false),

            limiter_ceiling_db: AtomicF32::new(-1.0),
            limiter_bypass: AtomicBool::new(false),

            volume: AtomicF32::new(1.0),

            dirty: AtomicBool::new(false),
        }
    }

    /// Mark parameters as changed (called by setter methods).
    #[inline]
    pub fn mark_dirty(&self) {
        self.dirty.store(true, Ordering::Relaxed);
    }

    /// Check and clear dirty flag (called by audio thread).
    #[inline]
    pub fn take_dirty(&self) -> bool {
        self.dirty.swap(false, Ordering::Relaxed)
    }
}

impl Default for RtParams {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atomic_f32_round_trip() {
        let a = AtomicF32::new(3.14);
        assert!((a.load() - 3.14).abs() < 1e-6);
        a.store(-0.5);
        assert!((a.load() - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn test_params_defaults() {
        let p = RtParams::new();
        assert!((p.gain_db.load() - 0.0).abs() < 1e-6);
        assert!((p.ceiling_db.load() - (-1.0)).abs() < 1e-6);
        assert_eq!(p.irc_mode.load(Ordering::Relaxed), 3);
        assert!((p.width_pct.load() - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_dirty_flag() {
        let p = RtParams::new();
        assert!(!p.take_dirty());
        p.mark_dirty();
        assert!(p.take_dirty());
        assert!(!p.take_dirty()); // cleared after take
    }
}
