//! The training corpus is code, not a blob: a seeded template grammar.
//!
//! Every sentence is a filled template. Change the seed and you get a different
//! corpus; the vocabulary is bounded so the `W1` grid stays readable.

use crate::data::Snippet;
use crate::rng::Rng;

/// Fraction of labels deliberately flipped to create a train/test gap.
pub const NOISE: f32 = 0.08;

// Shared function words — appear in both positive and negative sentences.
const FUNC: [&str; 12] = [
    "the", "a", "was", "seemed", "i", "found", "felt", "today", "really", "very", "overall", "and",
];

// Subject nouns — shared across classes.
const NOUNS: [&str; 14] = [
    "service",
    "delivery",
    "price",
    "staff",
    "food",
    "room",
    "wait",
    "quality",
    "experience",
    "product",
    "support",
    "packaging",
    "selection",
    "value",
];

pub const POS_ADJ: [&str; 12] = [
    "great",
    "fine",
    "fast",
    "reasonable",
    "friendly",
    "excellent",
    "helpful",
    "quick",
    "good",
    "pleasant",
    "clean",
    "reliable",
];

pub const NEG_ADJ: [&str; 12] = [
    "terrible",
    "slow",
    "overpriced",
    "rude",
    "awful",
    "poor",
    "bad",
    "disappointing",
    "dirty",
    "unhelpful",
    "late",
    "broken",
];

fn adj_pair(rng: &mut Rng, positive: bool) -> (&'static str, &'static str) {
    let table: &[&'static str] = if positive { &POS_ADJ } else { &NEG_ADJ };
    let a = table[rng.below(table.len())];
    let mut b = table[rng.below(table.len())];
    if b == a {
        b = table[(rng.below(table.len()) + 1) % table.len()];
    }
    (a, b)
}

fn sentence(rng: &mut Rng, positive: bool) -> String {
    let n = NOUNS[rng.below(NOUNS.len())];
    let (a, a2) = adj_pair(rng, positive);
    match rng.below(11) {
        0 => format!("{} {} was {}", FUNC[0], n, a),
        1 => format!("{} was {}", n, a),
        2 => format!("{} {} {} {}", FUNC[4], FUNC[0], n, a),
        3 => format!("{} {} {} {}", FUNC[0], n, FUNC[3], a),
        4 => format!("{} {} {} {} {}", FUNC[0], n, FUNC[6], a, FUNC[7]),
        5 => format!("{} {} {} {} {}", FUNC[0], n, FUNC[2], FUNC[8], a),
        6 => format!("{} {} {} {} {}", FUNC[0], n, FUNC[2], FUNC[9], a),
        7 => format!("{} {} {} {} {}", FUNC[10], FUNC[0], n, FUNC[2], a),
        8 => format!("{} {} {} {}", FUNC[5], FUNC[0], n, a),
        9 => format!("{} {} {} {} {} {}", FUNC[0], n, FUNC[2], a, FUNC[11], a2),
        _ => format!("{} {} {} {}", FUNC[1], n, FUNC[2], a),
    }
}

/// Generate `n` labeled sentences, half positive and half negative, deterministic
/// in `seed`.
pub fn generate(seed: u64, n: usize) -> Vec<Snippet> {
    let mut rng = Rng::new(seed);
    let mut out: Vec<Snippet> = Vec::with_capacity(n);
    for i in 0..n {
        let positive = i % 2 == 0;
        out.push(Snippet {
            text: sentence(&mut rng, positive),
            label: if positive { 1 } else { 0 },
        });
    }
    // Shuffle so classes are interleaved.
    for i in (1..out.len()).rev() {
        let j = rng.below(i + 1);
        out.swap(i, j);
    }

    // Deliberate label noise (~8%, balanced): the model can memorize the
    // training set but cannot reach a perfect test score. This is the
    // generalization gap made visible. Class counts stay 50/50.
    let per_class = ((n as f32) * NOISE / 2.0).round() as usize;
    let mut p = 0;
    let mut neg = 0;
    for s in out.iter_mut() {
        if s.label == 1 && p < per_class {
            s.label = 0;
            p += 1;
        } else if s.label == 0 && neg < per_class {
            s.label = 1;
            neg += 1;
        }
    }
    out
}
