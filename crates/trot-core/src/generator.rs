//! The training corpus is code, not a blob: a seeded template grammar.
//!
//! Every sentence is a filled template. Change the seed and you get a different
//! corpus; the vocabulary is a fixed 50-token lexicon so the `W1` grid keeps the
//! same shape and stays readable.
//!
//! Nouns and adjectives are paired through an explicit compatibility table,
//! templates are complete sentences (capitalized, final period), duplicate texts
//! are rejected, and label noise is a uniform per-class selection after the
//! corpus shuffle.

use std::collections::HashSet;

use crate::data::Snippet;
use crate::rng::Rng;

/// Default fraction of deliberately flipped labels (v3 default is clean).
pub const NOISE: f32 = 0.0;
/// One-click preset that makes the train/test gap visible.
pub const NOISE_PRESET: f32 = 0.08;

/// The fixed v3 vocabulary, sorted ascending. `vocab_tokens[i]` is `W1` column `i`.
pub const LEXICON: [&str; 50] = [
    "and",
    "awful",
    "bad",
    "broken",
    "clean",
    "delivery",
    "dirty",
    "disappointing",
    "excellent",
    "experience",
    "fast",
    "felt",
    "fine",
    "food",
    "found",
    "frankly",
    "friendly",
    "good",
    "great",
    "helpful",
    "honestly",
    "i",
    "late",
    "overall",
    "overpriced",
    "packaging",
    "pleasant",
    "poor",
    "price",
    "product",
    "quality",
    "quick",
    "really",
    "reasonable",
    "reliable",
    "room",
    "rude",
    "seemed",
    "selection",
    "service",
    "slow",
    "staff",
    "support",
    "terrible",
    "the",
    "today",
    "unhelpful",
    "value",
    "wait",
    "was",
];

/// Subject nouns, in the order used by the compatibility table and fallback scan.
pub const NOUNS: [&str; 14] = [
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
    "friendly",
    "helpful",
    "fast",
    "quick",
    "clean",
    "reasonable",
    "great",
    "fine",
    "excellent",
    "good",
    "pleasant",
    "reliable",
];

pub const NEG_ADJ: [&str; 12] = [
    "rude",
    "unhelpful",
    "slow",
    "late",
    "dirty",
    "broken",
    "overpriced",
    "terrible",
    "awful",
    "poor",
    "bad",
    "disappointing",
];

const MANNER_POS: [&str; 2] = ["friendly", "helpful"];
const MANNER_NEG: [&str; 2] = ["rude", "unhelpful"];
const SPEED_POS: [&str; 2] = ["fast", "quick"];
const SPEED_NEG: [&str; 2] = ["slow", "late"];
const CONDITION_POS: [&str; 1] = ["clean"];
const CONDITION_NEG: [&str; 2] = ["dirty", "broken"];
const PRICE_POS: [&str; 1] = ["reasonable"];
const PRICE_NEG: [&str; 1] = ["overpriced"];
const GENERAL_POS: [&str; 6] = ["great", "fine", "excellent", "good", "pleasant", "reliable"];
const GENERAL_NEG: [&str; 5] = ["terrible", "awful", "poor", "bad", "disappointing"];

/// Adjective categories in the contract order: manner, speed, condition,
/// price/value, general evaluation. Each pair is `(positive, negative)`.
const CATEGORIES: [(&[&str], &[&str]); 5] = [
    (&MANNER_POS, &MANNER_NEG),
    (&SPEED_POS, &SPEED_NEG),
    (&CONDITION_POS, &CONDITION_NEG),
    (&PRICE_POS, &PRICE_NEG),
    (&GENERAL_POS, &GENERAL_NEG),
];

#[derive(Clone, Copy)]
enum Rule {
    None,
    All,
    Only(&'static [&'static str]),
}

#[derive(Clone, Copy)]
struct NounRule {
    noun: &'static str,
    manner: Rule,
    speed: Rule,
    condition: Rule,
    price: Rule,
    general: Rule,
}

impl NounRule {
    fn rule(&self, category: usize) -> Rule {
        match category {
            0 => self.manner,
            1 => self.speed,
            2 => self.condition,
            3 => self.price,
            _ => self.general,
        }
    }
}

const NOUN_RULES: [NounRule; 14] = [
    NounRule {
        noun: "service",
        manner: Rule::All,
        speed: Rule::Only(&["fast", "quick", "slow"]),
        condition: Rule::None,
        price: Rule::All,
        general: Rule::All,
    },
    NounRule {
        noun: "delivery",
        manner: Rule::None,
        speed: Rule::All,
        condition: Rule::None,
        price: Rule::All,
        general: Rule::All,
    },
    NounRule {
        noun: "price",
        manner: Rule::None,
        speed: Rule::None,
        condition: Rule::None,
        price: Rule::All,
        general: Rule::All,
    },
    NounRule {
        noun: "staff",
        manner: Rule::All,
        speed: Rule::None,
        condition: Rule::None,
        price: Rule::None,
        general: Rule::All,
    },
    NounRule {
        noun: "food",
        manner: Rule::None,
        speed: Rule::None,
        condition: Rule::None,
        price: Rule::All,
        general: Rule::All,
    },
    NounRule {
        noun: "room",
        manner: Rule::None,
        speed: Rule::None,
        condition: Rule::Only(&["clean", "dirty"]),
        price: Rule::All,
        general: Rule::All,
    },
    NounRule {
        noun: "wait",
        manner: Rule::None,
        speed: Rule::Only(&["fast", "quick", "slow"]),
        condition: Rule::None,
        price: Rule::None,
        general: Rule::All,
    },
    NounRule {
        noun: "quality",
        manner: Rule::None,
        speed: Rule::None,
        condition: Rule::None,
        price: Rule::Only(&["reasonable"]),
        general: Rule::All,
    },
    NounRule {
        noun: "experience",
        manner: Rule::None,
        speed: Rule::None,
        condition: Rule::None,
        price: Rule::None,
        general: Rule::All,
    },
    NounRule {
        noun: "product",
        manner: Rule::None,
        speed: Rule::None,
        condition: Rule::All,
        price: Rule::All,
        general: Rule::All,
    },
    NounRule {
        noun: "support",
        manner: Rule::All,
        speed: Rule::All,
        condition: Rule::None,
        price: Rule::None,
        general: Rule::All,
    },
    NounRule {
        noun: "packaging",
        manner: Rule::None,
        speed: Rule::None,
        condition: Rule::All,
        price: Rule::None,
        general: Rule::All,
    },
    NounRule {
        noun: "selection",
        manner: Rule::None,
        speed: Rule::None,
        condition: Rule::None,
        price: Rule::None,
        general: Rule::All,
    },
    NounRule {
        noun: "value",
        manner: Rule::None,
        speed: Rule::None,
        condition: Rule::None,
        price: Rule::All,
        general: Rule::All,
    },
];

/// T1..T23 in contract order. `{adj2}` appears only in T21.
pub const TEMPLATES: [&str; 23] = [
    "The {noun} was {adj}.",
    "The {noun} seemed {adj}.",
    "I found the {noun} {adj}.",
    "I felt the {noun} was {adj}.",
    "Overall the {noun} was {adj}.",
    "Today the {noun} seemed {adj}.",
    "I found the {noun} {adj} today.",
    "Overall I found the {noun} {adj}.",
    "The {noun} really was {adj}.",
    "Honestly the {noun} was {adj}.",
    "Frankly the {noun} seemed {adj}.",
    "The {noun} was {adj} overall.",
    "Today I found the {noun} {adj}.",
    "Overall the {noun} seemed {adj}.",
    "I really felt the {noun} was {adj}.",
    "The {noun} seemed {adj} today.",
    "Honestly I found the {noun} {adj}.",
    "Frankly the {noun} was {adj}.",
    "The {noun} was {adj} today.",
    "I felt the {noun} was {adj} today.",
    "The {noun} was {adj} and {adj2}.",
    "Honestly the {noun} seemed {adj}.",
    "Frankly I felt the {noun} was {adj}.",
];

const T21: usize = 20;
const N_MIN: usize = 100;
const N_MAX: usize = 5000;
const DUPLICATE_RETRIES: usize = 64;

/// Adjectives legally paired with `noun_index` for the requested polarity.
pub fn adjective_pool(noun_index: usize, positive: bool) -> Vec<&'static str> {
    let rule = &NOUN_RULES[noun_index];
    let mut pool = Vec::with_capacity(12);
    for (category, &(pos, neg)) in CATEGORIES.iter().enumerate() {
        let source = if positive { pos } else { neg };
        match rule.rule(category) {
            Rule::None => {}
            Rule::All => pool.extend_from_slice(source),
            Rule::Only(allowed) => {
                pool.extend(source.iter().copied().filter(|adj| allowed.contains(adj)))
            }
        }
    }
    pool
}

fn render(template_index: usize, noun: &str, adj: &str, adj2: Option<&str>) -> String {
    let mut text = TEMPLATES[template_index].replace("{noun}", noun);
    if let Some(second) = adj2 {
        text = text.replace("{adj2}", second);
    }
    text.replace("{adj}", adj)
}

/// Every legal sentence for one clean polarity, in deterministic fallback order:
/// template, noun, first adjective, then (for T21) second adjective.
pub fn candidate_space(positive: bool) -> Vec<String> {
    let mut candidates = Vec::new();
    for template_index in 0..TEMPLATES.len() {
        for noun_index in 0..NOUNS.len() {
            let pool = adjective_pool(noun_index, positive);
            if template_index == T21 {
                for (a, adj) in pool.iter().enumerate() {
                    for (b, adj2) in pool.iter().enumerate() {
                        if a != b {
                            candidates.push(render(
                                template_index,
                                NOUN_RULES[noun_index].noun,
                                adj,
                                Some(adj2),
                            ));
                        }
                    }
                }
            } else {
                for adj in &pool {
                    candidates.push(render(
                        template_index,
                        NOUN_RULES[noun_index].noun,
                        adj,
                        None,
                    ));
                }
            }
        }
    }
    candidates
}

fn draw_text(rng: &mut Rng, positive: bool) -> String {
    let template_index = rng.below(TEMPLATES.len());
    let noun_index = rng.below(NOUNS.len());
    let pool = adjective_pool(noun_index, positive);
    let adj_index = rng.below(pool.len());
    let adj = pool[adj_index];
    if template_index == T21 {
        let mut adj2_index = rng.below(pool.len() - 1);
        if adj2_index >= adj_index {
            adj2_index += 1;
        }
        render(
            template_index,
            NOUN_RULES[noun_index].noun,
            adj,
            Some(pool[adj2_index]),
        )
    } else {
        render(template_index, NOUN_RULES[noun_index].noun, adj, None)
    }
}

fn fallback_text(
    i: usize,
    positive: bool,
    accepted: &HashSet<String>,
    cache: &mut Option<Vec<String>>,
) -> String {
    let candidates = cache.get_or_insert_with(|| candidate_space(positive));
    let mut cursor = (i * 97) % candidates.len();
    for _ in 0..candidates.len() {
        let text = &candidates[cursor];
        if !accepted.contains(text) {
            return text.clone();
        }
        cursor = (cursor + 1) % candidates.len();
    }
    panic!(
        "trot corpus candidate space exhausted for {} polarity",
        if positive { "positive" } else { "negative" }
    );
}

fn shuffle<T>(items: &mut [T], rng: &mut Rng) {
    for i in (1..items.len()).rev() {
        let j = rng.below(i + 1);
        items.swap(i, j);
    }
}

fn effective_n(n: usize) -> usize {
    let clamped = n.clamp(N_MIN, N_MAX);
    clamped - clamped % 2
}

/// Generate `n` labeled sentences, half positive and half negative, deterministic
/// in `seed`, with the default amount of deliberate label noise.
pub fn generate(seed: u64, n: usize) -> Vec<Snippet> {
    generate_with_noise(seed, n, NOISE)
}

/// Generate `n` labeled sentences with `noise` in `[0, 1]` — the fraction of
/// labels flipped (balanced across classes, so the 50/50 split is preserved).
pub fn generate_with_noise(seed: u64, n: usize, noise: f32) -> Vec<Snippet> {
    let n = effective_n(n);
    let noise = noise.clamp(0.0, 1.0);
    let mut rng = Rng::new(seed);
    let mut out: Vec<Snippet> = Vec::with_capacity(n);
    let mut accepted: HashSet<String> = HashSet::with_capacity(n);
    let mut positive_fallback: Option<Vec<String>> = None;
    let mut negative_fallback: Option<Vec<String>> = None;

    for i in 0..n {
        let positive = i % 2 == 0;
        let mut chosen = None;
        for _ in 0..DUPLICATE_RETRIES {
            let text = draw_text(&mut rng, positive);
            if !accepted.contains(&text) {
                chosen = Some(text);
                break;
            }
        }
        let text = match chosen {
            Some(text) => text,
            None => fallback_text(
                i,
                positive,
                &accepted,
                if positive {
                    &mut positive_fallback
                } else {
                    &mut negative_fallback
                },
            ),
        };
        accepted.insert(text.clone());
        out.push(Snippet {
            text,
            label: if positive { 1 } else { 0 },
            flipped: false,
        });
    }

    // Shuffle so classes are interleaved before any label work.
    shuffle(&mut out, &mut rng);

    // Deliberate label noise (balanced): the model can memorize the training
    // set but cannot reach a perfect test score. Select uniformly within each
    // class after the shuffle so flips are not clustered in generation order.
    let k = (((n as f32) * noise / 2.0).round() as usize).min(n / 2);
    if k > 0 {
        let mut positives: Vec<usize> = (0..out.len()).filter(|&i| out[i].label == 1).collect();
        let mut negatives: Vec<usize> = (0..out.len()).filter(|&i| out[i].label == 0).collect();
        shuffle(&mut positives, &mut rng);
        shuffle(&mut negatives, &mut rng);
        for &i in positives.iter().take(k) {
            out[i].label = 0;
            out[i].flipped = true;
        }
        for &i in negatives.iter().take(k) {
            out[i].label = 1;
            out[i].flipped = true;
        }
    }
    out
}
