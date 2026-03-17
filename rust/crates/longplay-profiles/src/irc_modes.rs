//! IRC Mode definitions matching iZotope Ozone 12.
//!
//! 6 modes: IRC 1-5 + IRC LL
//! Sub-modes only on IRC 3 and IRC 4: Pumping / Balanced / Crisp

#[derive(Debug, Clone)]
pub struct IRCSubMode {
    pub name: String,
    pub description: String,
    pub release_factor: f64,
    pub character: f64,
}

#[derive(Debug, Clone)]
pub struct IRCMode {
    pub name: String,
    pub display_name: String,
    pub description: String,
    pub base_attack_ms: f64,
    pub base_release_ms: f64,
    pub lookahead_ms: f64,
    pub algorithm: String,
    pub sub_modes: Vec<IRCSubMode>,
}

/// Returns all 6 IRC modes matching Ozone 12.
pub fn get_irc_modes() -> Vec<IRCMode> {
    vec![
        // IRC 1: Transparent — clean peak limiting
        // Algorithm: Simple look-ahead brickwall limiter. No spectral weighting.
        // Similar to: Waves L1/L2 transparent mode, FabFilter Pro-L "Safe"
        IRCMode {
            name: "IRC1".into(),
            display_name: "IRC 1".into(),
            description: "Transparent — Clean peak limiting with minimal coloration".into(),
            base_attack_ms: 5.0,
            base_release_ms: 200.0,
            lookahead_ms: 10.0,
            algorithm: "peak_limiter".into(),
            sub_modes: vec![], // No sub-modes
        },
        // IRC 2: Adaptive — program-dependent release
        // Algorithm: Dual-envelope follower (fast transient + slow sustained).
        // Release adapts automatically based on signal content.
        // Similar to: FabFilter Pro-L "Modern", Sonnox Oxford Limiter
        IRCMode {
            name: "IRC2".into(),
            display_name: "IRC 2".into(),
            description: "Adaptive — Program-dependent release for musical results".into(),
            base_attack_ms: 3.0,
            base_release_ms: 100.0,
            lookahead_ms: 5.0,
            algorithm: "adaptive_release".into(),
            sub_modes: vec![], // No sub-modes
        },
        // IRC 3: Multi-band — frequency-weighted limiting (MOST POPULAR)
        // Algorithm: 4-band crossover (120/1k/8k Hz), independent limiter per band.
        // Prevents bass transients from pumping highs.
        // Similar to: Ozone Maximizer default, Waves L3 Multimaximizer
        // Sub-modes: Pumping (fast release), Balanced (default), Crisp (transient preserve)
        IRCMode {
            name: "IRC3".into(),
            display_name: "IRC 3".into(),
            description: "Multi-band — Frequency-weighted limiting, spectral preservation".into(),
            base_attack_ms: 3.0,
            base_release_ms: 120.0,
            lookahead_ms: 5.0,
            algorithm: "multiband_limiter".into(),
            sub_modes: vec![
                IRCSubMode {
                    name: "Pumping".into(),
                    description: "Fast release — audible pump, great for EDM/Dance".into(),
                    release_factor: 0.4,
                    character: 6.0,
                },
                IRCSubMode {
                    name: "Balanced".into(),
                    description: "Natural release — default for most material".into(),
                    release_factor: 1.0,
                    character: 3.0,
                },
                IRCSubMode {
                    name: "Crisp".into(),
                    description: "Transient-preserving — best for acoustic/vocal/jazz".into(),
                    release_factor: 1.8,
                    character: 1.0,
                },
            ],
        },
        // IRC 4: Aggressive — saturation + limiting
        // Algorithm: 3-stage: harmonic saturation → soft clip → brickwall limiter.
        // Creates loud, characterful sound with even-harmonic warmth.
        // Similar to: Slate FG-X, Waves L2, FabFilter Pro-L "Aggressive"
        // Sub-modes: Pumping (fast), Balanced (default), Crisp (transient)
        IRCMode {
            name: "IRC4".into(),
            display_name: "IRC 4".into(),
            description: "Aggressive — Saturation + limiting for maximum loudness with character".into(),
            base_attack_ms: 0.5,
            base_release_ms: 50.0,
            lookahead_ms: 3.0,
            algorithm: "aggressive_saturate".into(),
            sub_modes: vec![
                IRCSubMode {
                    name: "Pumping".into(),
                    description: "Fast aggressive release — pumping loudness".into(),
                    release_factor: 0.4,
                    character: 8.0,
                },
                IRCSubMode {
                    name: "Balanced".into(),
                    description: "Balanced aggression — loud but controlled".into(),
                    release_factor: 1.0,
                    character: 5.0,
                },
                IRCSubMode {
                    name: "Crisp".into(),
                    description: "Aggressive with transient snap — punch + loudness".into(),
                    release_factor: 1.8,
                    character: 4.0,
                },
            ],
        },
        // IRC 5: Maximum Density (NEW in Ozone 12)
        // Algorithm: 4-band compression (upward + downward) THEN 4-band limiting.
        // Maximizes RMS loudness by reducing dynamic range before limiting.
        // Similar to: No direct equivalent — Ozone 12 exclusive concept
        IRCMode {
            name: "IRC5".into(),
            display_name: "IRC 5".into(),
            description: "Maximum Density — Multi-band compression + limiting, loudest possible".into(),
            base_attack_ms: 2.0,
            base_release_ms: 100.0,
            lookahead_ms: 5.0,
            algorithm: "maximum_density".into(),
            sub_modes: vec![], // No sub-modes
        },
        // IRC LL: Low Latency
        // Algorithm: Zero-lookahead feedback limiter. Very fast attack.
        // Lower quality but suitable for real-time monitoring.
        // Similar to: Waves L1+ in low-latency mode, Plugin Alliance bx_limiter True Peak
        IRCMode {
            name: "IRCLL".into(),
            display_name: "IRC LL".into(),
            description: "Low Latency — Zero-lookahead for real-time monitoring".into(),
            base_attack_ms: 0.05,
            base_release_ms: 30.0,
            lookahead_ms: 0.0,
            algorithm: "feedback_limiter".into(),
            sub_modes: vec![], // No sub-modes
        },
    ]
}

/// Get a specific IRC mode by name.
pub fn get_irc_mode(name: &str) -> Option<IRCMode> {
    let clean = name.to_uppercase().replace(' ', "").replace('-', "");
    get_irc_modes().into_iter().find(|m| {
        let m_clean = m.name.to_uppercase().replace(' ', "").replace('-', "");
        m_clean == clean
    })
}

/// Get sub-modes for a given IRC mode name.
pub fn get_irc_sub_modes(mode_name: &str) -> Vec<String> {
    match get_irc_mode(mode_name) {
        Some(mode) => mode.sub_modes.iter().map(|s| s.name.clone()).collect(),
        None => vec![],
    }
}

/// Get all IRC mode display names (for UI dropdown).
pub fn get_irc_mode_names() -> Vec<String> {
    get_irc_modes().iter().map(|m| m.display_name.clone()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_six_modes() {
        let modes = get_irc_modes();
        assert_eq!(modes.len(), 6);
    }

    #[test]
    fn test_mode_names() {
        let names = get_irc_mode_names();
        assert_eq!(names, vec!["IRC 1", "IRC 2", "IRC 3", "IRC 4", "IRC 5", "IRC LL"]);
    }

    #[test]
    fn test_irc3_has_3_sub_modes() {
        let subs = get_irc_sub_modes("IRC3");
        assert_eq!(subs.len(), 3);
        assert_eq!(subs, vec!["Pumping", "Balanced", "Crisp"]);
    }

    #[test]
    fn test_irc4_has_3_sub_modes() {
        let subs = get_irc_sub_modes("IRC4");
        assert_eq!(subs.len(), 3);
        assert_eq!(subs, vec!["Pumping", "Balanced", "Crisp"]);
    }

    #[test]
    fn test_irc1_no_sub_modes() {
        let subs = get_irc_sub_modes("IRC1");
        assert_eq!(subs.len(), 0);
    }

    #[test]
    fn test_irc5_no_sub_modes() {
        let subs = get_irc_sub_modes("IRC5");
        assert_eq!(subs.len(), 0);
    }

    #[test]
    fn test_ircll_no_sub_modes() {
        let subs = get_irc_sub_modes("IRCLL");
        assert_eq!(subs.len(), 0);
    }

    #[test]
    fn test_get_mode_by_name() {
        assert!(get_irc_mode("IRC 3").is_some());
        assert!(get_irc_mode("IRC3").is_some());
        assert!(get_irc_mode("IRC LL").is_some());
        assert!(get_irc_mode("IRCLL").is_some());
    }
}
