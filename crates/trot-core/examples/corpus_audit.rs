//! Print every generated snippet with QA flags and a summary defect count.
//!
//! ```text
//! cargo run -p trot-core --example corpus_audit -- 42 256 0.0
//! cargo run -p trot-core --example corpus_audit -- 42 256 0.08
//! ```
//!
//! A clean corpus prints `ok` for every row and exits 0. Any defect is counted
//! and the process exits 1 so the same command can guard a script.

use std::collections::HashSet;
use std::process::ExitCode;

use trot_core::data::{tokenize, Dataset, Snippet};
use trot_core::generator;

fn clean_label(snippet: &Snippet) -> u8 {
    if snippet.flipped {
        1 - snippet.label
    } else {
        snippet.label
    }
}

fn text_polarity(text: &str) -> Option<u8> {
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
    match (positive, negative) {
        (true, false) => Some(1),
        (false, true) => Some(0),
        _ => None,
    }
}

fn grammar_ok(text: &str) -> bool {
    if text.contains("  ") {
        return false;
    }
    let tokens = tokenize(text);
    if tokens
        .iter()
        .any(|token| matches!(token.as_str(), "a" | "an" | "very"))
    {
        return false;
    }
    !tokens
        .windows(2)
        .any(|pair| pair[0] == "i" && pair[1] == "the")
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let seed: u64 = args.first().and_then(|v| v.parse().ok()).unwrap_or(42);
    let n: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(256);
    let noise: f32 = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(0.0);

    let corpus = generator::generate_with_noise(seed, n, noise);
    let dataset = Dataset::from_snippets(corpus, seed);
    let positive: HashSet<String> = generator::candidate_space(true).into_iter().collect();
    let negative: HashSet<String> = generator::candidate_space(false).into_iter().collect();
    let sets = [negative, positive];

    println!("TROT corpus audit");
    println!(
        "seed={seed}  n={}  noise={noise:.2}  snippets={}",
        n,
        dataset.snippets.len()
    );
    println!();
    println!(
        "{:>4}  {:^5}  {:<5}  {:<6}  {:<22}  text",
        "idx", "label", "split", "flip", "flags"
    );

    let mut seen = HashSet::new();
    let mut duplicate = 0usize;
    let mut capitalization = 0usize;
    let mut punctuation = 0usize;
    let mut grammar = 0usize;
    let mut collocation = 0usize;
    let mut polarity = 0usize;
    let mut flips = 0usize;

    for (i, snippet) in dataset.snippets.iter().enumerate() {
        let clean = clean_label(snippet);
        let observed = text_polarity(&snippet.text);
        let is_duplicate = !seen.insert(snippet.text.clone());
        let capital_ok = snippet
            .text
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_uppercase());
        let punctuation_ok = snippet.text.ends_with('.') && !snippet.text.ends_with("..");
        let grammar_row_ok = grammar_ok(&snippet.text);
        let polarity_ok = observed == Some(clean);
        let collocation_ok = sets[clean as usize].contains(&snippet.text);

        let mut flags = Vec::new();
        if is_duplicate {
            duplicate += 1;
            flags.push("dup");
        }
        if !capital_ok {
            capitalization += 1;
            flags.push("cap");
        }
        if !punctuation_ok {
            punctuation += 1;
            flags.push("punct");
        }
        if !grammar_row_ok {
            grammar += 1;
            flags.push("grammar");
        }
        if !collocation_ok {
            collocation += 1;
            flags.push("colloc");
        }
        if !polarity_ok {
            polarity += 1;
            flags.push("polarity");
        }
        if snippet.flipped {
            flips += 1;
        }
        let status = if flags.is_empty() {
            "ok".to_string()
        } else {
            flags.join(",")
        };

        println!(
            "{:>4}  {:^5}  {:<5}  {:<6}  {:<22}  {}",
            i,
            snippet.label,
            if dataset.is_test(i) { "test" } else { "train" },
            if snippet.flipped { "flip" } else { "-" },
            status,
            snippet.text
        );
    }

    let defects = duplicate + capitalization + punctuation + grammar + collocation + polarity;
    println!();
    println!("summary");
    println!("  snippets             : {}", dataset.snippets.len());
    println!("  duplicates           : {duplicate}");
    println!("  capitalization       : {capitalization}");
    println!("  final punctuation    : {punctuation}");
    println!("  banned grammar       : {grammar}");
    println!("  noun/adjective pairs : {collocation}");
    println!("  polarity consistency : {polarity}");
    println!("  intentional flips    : {flips}");
    println!(
        "  total defects        : {defects} ({})",
        if defects == 0 { "clean" } else { "needs work" }
    );

    if defects == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
