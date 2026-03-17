use crate::EPSILON;

/// Convert decibels to linear amplitude
pub fn db_to_linear(db: f64) -> f64 {
    10.0_f64.powf(db / 20.0)
}

/// Convert linear amplitude to decibels
pub fn linear_to_db(linear: f64) -> f64 {
    if linear <= EPSILON {
        return -200.0;
    }
    20.0 * linear.log10()
}

/// Convert decibels to linear amplitude (f32)
pub fn db_to_linear_f(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

/// Convert linear amplitude to decibels (f32)
pub fn linear_to_db_f(linear: f32) -> f32 {
    if linear <= 1e-10 {
        return -200.0;
    }
    20.0 * linear.log10()
}

/// Convert milliseconds to samples
pub fn ms_to_samples(ms: f64, sample_rate: i32) -> f64 {
    ms * sample_rate as f64 / 1000.0
}

/// Convert samples to milliseconds
pub fn samples_to_ms(samples: f64, sample_rate: i32) -> f64 {
    samples * 1000.0 / sample_rate as f64
}

/// Make a number odd (round up if even)
pub fn make_odd(n: i32) -> i32 {
    if n % 2 == 0 { n + 1 } else { n }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_db_to_linear() {
        assert!((db_to_linear(0.0) - 1.0).abs() < 1e-10);
        assert!((db_to_linear(-6.0) - 0.5011872).abs() < 1e-5);
        assert!((db_to_linear(20.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_to_db() {
        assert!((linear_to_db(1.0) - 0.0).abs() < 1e-10);
        assert!((linear_to_db(0.5) - (-6.0206)).abs() < 1e-3);
        assert_eq!(linear_to_db(0.0), -200.0);
    }

    #[test]
    fn test_ms_to_samples() {
        assert!((ms_to_samples(1000.0, 44100) - 44100.0).abs() < 1e-10);
        assert!((ms_to_samples(10.0, 48000) - 480.0).abs() < 1e-10);
    }
}
