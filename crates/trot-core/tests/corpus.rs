use std::collections::HashSet;

use trot_core::data::{tokenize, Dataset};
use trot_core::generator;
use trot_core::model::Model;
use trot_core::rng::Rng;

fn candidate_set(positive: bool) -> HashSet<String> {
    generator::candidate_space(positive).into_iter().collect()
}

fn text_polarity(text: &str) -> u8 {
    let mut positive = false;
    let mut negative = false;
    for token in tokenize(text) {
        if generator::POS_ADJ.contains(&token.as_str()) {
            positive = true;
        }
        if generator::NEG_ADJ.contains(&token.as_str()) {
            negative = true;
        }
    }
    assert!(
        positive ^ negative,
        "sentence must use one adjective polarity: {text}"
    );
    if positive {
        1
    } else {
        0
    }
}

fn clean_label(snippet: &trot_core::data::Snippet) -> u8 {
    if snippet.flipped {
        1 - snippet.label
    } else {
        snippet.label
    }
}

fn assert_clean_text(text: &str) {
    assert!(
        text.chars().next().is_some_and(|c| c.is_ascii_uppercase()),
        "not capitalized: {text}"
    );
    assert!(text.ends_with('.'), "missing final period: {text}");
    assert!(!text.ends_with(".."), "double final period: {text}");
    assert!(!text.contains("  "), "double space: {text}");
    let tokens = tokenize(text);
    for banned in ["a", "an", "very"] {
        assert!(
            !tokens.iter().any(|token| token == banned),
            "banned token {banned}: {text}"
        );
    }
    assert!(
        !tokens
            .windows(2)
            .any(|pair| pair[0] == "i" && pair[1] == "the"),
        "banned `i the` sequence: {text}"
    );
}

#[test]
fn generator_is_deterministic_balanced_and_bounded() {
    let a = generator::generate(42, 256);
    let b = generator::generate(42, 256);
    assert_eq!(a.len(), 256);
    assert_eq!(a, b);
    assert_eq!(a.iter().filter(|s| s.label == 1).count(), 128);
    assert_eq!(a.iter().filter(|s| s.label == 0).count(), 128);
    assert!(a.iter().all(|s| !s.flipped), "default noise must be 0.0");
    assert_ne!(
        a,
        generator::generate(43, 256),
        "seed must change the corpus"
    );

    let ds = Dataset::generated(42, 256);
    assert_eq!(ds.vocab_len(), 50, "v3 vocabulary is fixed at 50 tokens");
    assert_eq!(ds.train_idx.len(), 205);
    assert_eq!(ds.test_idx.len(), 51);
    let mut rng = Rng::new(42);
    let model = Model::new(ds.vocab_len(), 8, &mut rng);
    assert_eq!(model.w1.len(), 8 * 50);
    assert_eq!(model.param_count(), 417);
}

#[test]
fn lexicon_is_the_fixed_sorted_contract() {
    let mut sorted = generator::LEXICON.to_vec();
    sorted.sort_unstable();
    assert_eq!(sorted, generator::LEXICON);
    assert_eq!(generator::LEXICON.len(), 50);
    for banned in ["a", "an", "very"] {
        assert!(!generator::LEXICON.contains(&banned));
    }

    let ds = Dataset::generated(42, 256);
    let expected: Vec<String> = generator::LEXICON
        .iter()
        .map(|token| (*token).to_string())
        .collect();
    assert_eq!(ds.vocab, expected);
    for snippet in &ds.snippets {
        for token in tokenize(&snippet.text) {
            assert!(
                generator::LEXICON.contains(&token.as_str()),
                "token {token} outside the fixed lexicon"
            );
        }
    }
}

#[test]
fn corpus_candidate_space_matches_contract() {
    let positive = generator::candidate_space(true);
    let negative = generator::candidate_space(false);
    assert_eq!(positive.len(), 3_164);
    assert_eq!(negative.len(), 2_630);
    assert_eq!(
        positive.iter().collect::<HashSet<_>>().len(),
        positive.len(),
        "positive candidate texts must be unique"
    );
    assert_eq!(
        negative.iter().collect::<HashSet<_>>().len(),
        negative.len(),
        "negative candidate texts must be unique"
    );
}

#[test]
fn corpus_is_clean_unique_and_compatible() {
    let corpus = generator::generate_with_noise(42, 256, 0.0);
    let positive = candidate_set(true);
    let negative = candidate_set(false);
    let mut seen = HashSet::new();

    for snippet in &corpus {
        assert_clean_text(&snippet.text);
        assert!(
            seen.insert(snippet.text.clone()),
            "duplicate text: {}",
            snippet.text
        );
        let polarity = text_polarity(&snippet.text);
        assert_eq!(polarity, snippet.label, "label mismatch: {}", snippet.text);
        let legal = if polarity == 1 { &positive } else { &negative };
        assert!(
            legal.contains(&snippet.text),
            "illegal noun/adjective pairing or template: {}",
            snippet.text
        );
    }
    assert_eq!(seen.len(), 256);
}

#[test]
fn corpus_is_unique_at_every_supported_size() {
    for n in [100usize, 256, 1000, 5000] {
        let corpus = generator::generate_with_noise(11, n, 0.08);
        assert_eq!(corpus.len(), n);
        let unique: HashSet<&str> = corpus.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(unique.len(), n, "duplicate text at n={n}");
        assert_eq!(corpus.iter().filter(|s| s.label == 1).count(), n / 2);
        assert_eq!(corpus.iter().filter(|s| s.label == 0).count(), n / 2);
    }
}

#[test]
fn noise_flip_counts_and_spread() {
    for (noise, expected_k) in [(0.08f32, 10usize), (0.20, 26)] {
        let corpus = generator::generate_with_noise(42, 256, noise);
        let flipped = corpus.iter().filter(|s| s.flipped).count();
        assert_eq!(flipped, expected_k * 2, "noise={noise}");
        assert_eq!(corpus.iter().filter(|s| s.label == 1).count(), 128);
        assert_eq!(corpus.iter().filter(|s| s.label == 0).count(), 128);

        let mut flips_by_clean_class = [Vec::new(), Vec::new()];
        let mut order_by_clean_class = [Vec::new(), Vec::new()];
        for (i, snippet) in corpus.iter().enumerate() {
            let polarity = text_polarity(&snippet.text);
            order_by_clean_class[polarity as usize].push(i);
            if snippet.flipped {
                flips_by_clean_class[polarity as usize].push(i);
            }
            assert_eq!(
                polarity,
                clean_label(snippet),
                "flipped polarity mismatch: {}",
                snippet.text
            );
        }

        for class in 0..2 {
            assert_eq!(flips_by_clean_class[class].len(), expected_k);
            let first_k: Vec<usize> = order_by_clean_class[class]
                .iter()
                .copied()
                .take(expected_k)
                .collect();
            assert_ne!(
                flips_by_clean_class[class], first_k,
                "flips clustered in the first k rows of class {class}"
            );
        }

        let flipped_indices: Vec<usize> = corpus
            .iter()
            .enumerate()
            .filter(|(_, s)| s.flipped)
            .map(|(i, _)| i)
            .collect();
        assert!(
            flipped_indices.iter().any(|&i| i >= corpus.len() / 4),
            "flips must not cluster in the leading rows"
        );
    }

    let clean = generator::generate_with_noise(42, 256, 0.0);
    assert!(clean.iter().all(|s| !s.flipped));
}

#[test]
fn noise_only_changes_labels_and_flags() {
    let clean = generator::generate_with_noise(42, 256, 0.0);
    let noisy = generator::generate_with_noise(42, 256, 0.08);
    assert_eq!(clean.len(), noisy.len());
    for (a, b) in clean.iter().zip(noisy.iter()) {
        assert_eq!(a.text, b.text);
        assert_eq!(a.label, clean_label(b));
        assert!(!a.flipped);
    }
    assert_eq!(
        noisy.iter().filter(|s| s.flipped).count(),
        20,
        "8% at n=256 is 10 flips per class"
    );
}

#[test]
fn corpus_controls_respected() {
    let clean = generator::generate_with_noise(7, 256, 0.0);
    assert_eq!(clean.iter().filter(|s| s.label == 1).count(), 128);
    assert_eq!(clean.iter().filter(|s| s.label == 0).count(), 128);

    // The fixed lexicon holds at every supported corpus size.
    for n in [100usize, 256, 1000, 5000] {
        let ds = Dataset::generated_with(11, n, 0.08);
        assert_eq!(ds.vocab_len(), 50, "n={n} vocab={}", ds.vocab_len());
        assert_eq!(ds.snippets.len(), n);
        assert_eq!(ds.positives() + ds.negatives(), n);
    }

    // Noise is balanced: counts stay 50/50 at the default rate.
    let noisy = Dataset::generated_with(7, 1000, 0.08);
    assert_eq!(noisy.positives(), 500);
    assert_eq!(noisy.negatives(), 500);
}

#[test]
fn token_stats_match_final_labels() {
    let ds = Dataset::generated_with(42, 256, 0.08);
    let stats = ds.token_stats();
    assert_eq!(stats.len(), 50);
    for (i, &(pos, neg)) in stats.iter().enumerate() {
        let mut expected = (0usize, 0usize);
        for snippet in &ds.snippets {
            if tokenize(&snippet.text).iter().any(|t| t == &ds.vocab[i]) {
                if snippet.label == 1 {
                    expected.0 += 1;
                } else {
                    expected.1 += 1;
                }
            }
        }
        assert_eq!((pos, neg), expected, "token {}", ds.vocab[i]);
    }
}
