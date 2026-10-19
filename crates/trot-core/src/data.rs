//! Data: tokenizing, vocabulary, and the deterministic train/test split.

use crate::generator;
use crate::rng::Rng;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Snippet {
    pub text: String,
    pub label: u8, // 0 = negative, 1 = positive
}

#[derive(Clone, Debug)]
pub struct Dataset {
    pub snippets: Vec<Snippet>,
    pub vocab: Vec<String>, // sorted, unique
    pub train_idx: Vec<usize>,
    pub test_idx: Vec<usize>,
}

/// `text` -> lowercase alphanumeric tokens, punctuation stripped.
pub fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

impl Dataset {
    /// Build a dataset from snippets: derive vocab, then split 80/20 with a seed.
    pub fn from_snippets(snippets: Vec<Snippet>, seed: u64) -> Self {
        let mut vocab: Vec<String> = Vec::new();
        for s in &snippets {
            for t in tokenize(&s.text) {
                vocab.push(t);
            }
        }
        vocab.sort();
        vocab.dedup();

        let mut order: Vec<usize> = (0..snippets.len()).collect();
        let mut rng = Rng::new(seed ^ 0x5EED_5EED);
        // Fisher-Yates
        for i in (1..order.len()).rev() {
            let j = rng.below(i + 1);
            order.swap(i, j);
        }
        let test_n = (snippets.len() as f32 * 0.2).round() as usize;
        let test_idx = order[..test_n].to_vec();
        let mut train_idx = order[test_n..].to_vec();
        train_idx.sort_unstable();

        Dataset {
            snippets,
            vocab,
            train_idx,
            test_idx,
        }
    }

    /// Deterministically generated corpus with the default label noise.
    pub fn generated(seed: u64, n: usize) -> Self {
        Self::generated_with(seed, n, generator::NOISE)
    }

    /// Deterministically generated corpus with a configurable label-noise rate.
    pub fn generated_with(seed: u64, n: usize, noise: f32) -> Self {
        Self::from_snippets(generator::generate_with_noise(seed, n, noise), seed)
    }

    /// True if the snippet at `index` is in the test split.
    pub fn is_test(&self, index: usize) -> bool {
        self.test_idx.binary_search(&index).is_ok()
    }

    /// Positive/negative snippet counts for each vocabulary token.
    pub fn token_stats(&self) -> Vec<(usize, usize)> {
        let mut stats = vec![(0usize, 0usize); self.vocab.len()];
        for s in &self.snippets {
            let x = self.vectorize(&s.text);
            let slot = if s.label == 1 { 0 } else { 1 };
            for (i, &v) in x.iter().enumerate() {
                if v != 0.0 {
                    if slot == 0 {
                        stats[i].0 += 1;
                    } else {
                        stats[i].1 += 1;
                    }
                }
            }
        }
        stats
    }

    pub fn vocab_len(&self) -> usize {
        self.vocab.len()
    }

    pub fn positives(&self) -> usize {
        self.snippets.iter().filter(|s| s.label == 1).count()
    }

    pub fn negatives(&self) -> usize {
        self.snippets.iter().filter(|s| s.label == 0).count()
    }

    /// Presence vector `x in {0,1}^V`.
    pub fn vectorize(&self, text: &str) -> Vec<f32> {
        let mut x = vec![0.0f32; self.vocab.len()];
        for t in tokenize(text) {
            if let Ok(i) = self.vocab.binary_search(&t) {
                x[i] = 1.0;
            }
        }
        x
    }
}
