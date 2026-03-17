//! Cooley-Tukey radix-2 FFT implementation
//! Ported from C++ analyzer.cpp

use std::f32::consts::PI;

/// Complex number for FFT computation
#[derive(Debug, Clone, Copy)]
pub struct ComplexNum {
    pub real: f32,
    pub imag: f32,
}

impl ComplexNum {
    pub fn new(real: f32, imag: f32) -> Self {
        Self { real, imag }
    }

    pub fn zero() -> Self {
        Self { real: 0.0, imag: 0.0 }
    }

    /// Compute magnitude: sqrt(real^2 + imag^2)
    pub fn magnitude(&self) -> f32 {
        (self.real * self.real + self.imag * self.imag).sqrt()
    }
}

impl std::ops::Add for ComplexNum {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        ComplexNum {
            real: self.real + other.real,
            imag: self.imag + other.imag,
        }
    }
}

impl std::ops::Sub for ComplexNum {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        ComplexNum {
            real: self.real - other.real,
            imag: self.imag - other.imag,
        }
    }
}

impl std::ops::Mul for ComplexNum {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        ComplexNum {
            real: self.real * other.real - self.imag * other.imag,
            imag: self.real * other.imag + self.imag * other.real,
        }
    }
}

/// Reverse bits of an integer value for bit-reversal permutation
///
/// # Arguments
/// * `value` - The integer value to bit-reverse
/// * `nbits` - Number of bits to reverse
pub fn reverse_bits(value: u32, nbits: u32) -> u32 {
    let mut result: u32 = 0;
    let mut v = value;
    for _ in 0..nbits {
        result = (result << 1) | (v & 1);
        v >>= 1;
    }
    result
}

/// Check if a number is a power of two
pub fn is_power_of_two(n: usize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

/// Find the next power of two greater than or equal to n
pub fn next_power_of_two(n: usize) -> usize {
    if is_power_of_two(n) {
        return n;
    }
    let mut result: usize = 1;
    while result < n {
        result <<= 1;
    }
    result
}

/// Count trailing zeros (equivalent to __builtin_ctz)
fn count_trailing_zeros(n: usize) -> u32 {
    if n == 0 {
        return 0;
    }
    n.trailing_zeros()
}

/// In-place Cooley-Tukey radix-2 FFT
///
/// Data must have a power-of-2 length. Performs bit-reversal permutation
/// followed by the butterfly computation stages.
///
/// # Panics
/// Panics if `data.len()` is not a power of 2 or is 0.
pub fn fft_cooley_tukey(data: &mut [ComplexNum]) {
    let n = data.len();

    if !is_power_of_two(n) || n == 0 {
        panic!("FFT size must be a power of 2");
    }

    let log2_n = count_trailing_zeros(n);

    // Bit-reversal permutation
    for i in 0..n {
        let j = reverse_bits(i as u32, log2_n) as usize;
        if i < j {
            data.swap(i, j);
        }
    }

    // Cooley-Tukey butterfly stages
    for s in 1..=log2_n {
        let m = 1usize << s; // 2^s
        let angle_step = -2.0 * PI / m as f32;

        let mut k = 0;
        while k < n {
            for j in 0..m / 2 {
                let angle = angle_step * j as f32;
                let w = ComplexNum::new(angle.cos(), angle.sin());

                let t = w * data[k + j + m / 2];
                let u = data[k + j];

                data[k + j] = u + t;
                data[k + j + m / 2] = u - t;
            }
            k += m;
        }
    }
}

/// Real FFT returning magnitude spectrum (non-negative frequencies only)
///
/// Pads input to next power of 2, computes the full FFT, then returns
/// magnitudes for bins 0..N/2+1.
///
/// # Arguments
/// * `input` - Real-valued input signal
///
/// # Returns
/// Magnitude spectrum with N/2+1 bins
pub fn rfft(input: &[f32]) -> Vec<f32> {
    let fft_size = next_power_of_two(input.len());
    let mut data = vec![ComplexNum::zero(); fft_size];

    for (i, &sample) in input.iter().enumerate() {
        data[i].real = sample;
    }

    fft_cooley_tukey(&mut data);

    // Extract magnitude spectrum (non-negative frequencies only)
    let mut result = Vec::with_capacity(fft_size / 2 + 1);
    for i in 0..=fft_size / 2 {
        result.push(data[i].magnitude());
    }

    result
}

/// Apply Hann window to a signal in-place
///
/// window[n] = 0.5 * (1 - cos(2*PI*n / (N-1)))
pub fn apply_hann_window(signal: &mut [f32]) {
    let n = signal.len();
    if n <= 1 {
        return;
    }
    for i in 0..n {
        let window = 0.5 * (1.0 - (2.0 * PI as f64 * i as f64 / (n as f64 - 1.0)).cos());
        signal[i] *= window as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_power_of_two() {
        assert!(is_power_of_two(1));
        assert!(is_power_of_two(2));
        assert!(is_power_of_two(4));
        assert!(is_power_of_two(1024));
        assert!(!is_power_of_two(0));
        assert!(!is_power_of_two(3));
        assert!(!is_power_of_two(5));
    }

    #[test]
    fn test_next_power_of_two() {
        assert_eq!(next_power_of_two(1), 1);
        assert_eq!(next_power_of_two(2), 2);
        assert_eq!(next_power_of_two(3), 4);
        assert_eq!(next_power_of_two(5), 8);
        assert_eq!(next_power_of_two(4096), 4096);
        assert_eq!(next_power_of_two(4097), 8192);
    }

    #[test]
    fn test_reverse_bits() {
        // 0b000 -> 0b000
        assert_eq!(reverse_bits(0, 3), 0);
        // 0b001 -> 0b100
        assert_eq!(reverse_bits(1, 3), 4);
        // 0b010 -> 0b010
        assert_eq!(reverse_bits(2, 3), 2);
        // 0b011 -> 0b110
        assert_eq!(reverse_bits(3, 3), 6);
    }

    #[test]
    fn test_complex_ops() {
        let a = ComplexNum::new(1.0, 2.0);
        let b = ComplexNum::new(3.0, 4.0);

        let sum = a + b;
        assert!((sum.real - 4.0).abs() < 1e-6);
        assert!((sum.imag - 6.0).abs() < 1e-6);

        let diff = a - b;
        assert!((diff.real - (-2.0)).abs() < 1e-6);
        assert!((diff.imag - (-2.0)).abs() < 1e-6);

        // (1+2i)*(3+4i) = 3+4i+6i+8i^2 = 3+10i-8 = -5+10i
        let prod = a * b;
        assert!((prod.real - (-5.0)).abs() < 1e-6);
        assert!((prod.imag - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_fft_dc_signal() {
        // A constant signal should have all energy in bin 0
        let input = vec![1.0f32; 8];
        let spectrum = rfft(&input);
        assert!(spectrum[0] > 7.9); // ~8.0
        for i in 1..spectrum.len() {
            assert!(spectrum[i] < 0.01, "bin {} should be near zero", i);
        }
    }

    #[test]
    fn test_hann_window() {
        let mut signal = vec![1.0f32; 4];
        apply_hann_window(&mut signal);
        // Endpoints should be near zero, center should be near 1
        assert!(signal[0].abs() < 1e-6);
        assert!(signal[3].abs() < 1e-6);
        assert!((signal[1] - 0.75).abs() < 0.01);
        assert!((signal[2] - 0.75).abs() < 0.01);
    }
}
