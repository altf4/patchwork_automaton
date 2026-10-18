//! Replay captured requests against the existing release rlib, timing native
//! rules, observation construction and JSON serialization independently.
//! This does not modify or rebuild the RustMage engine or its shared library.
use rustmage::{Config, Game};
use rustmage::protocol::{Request, Response};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

#[derive(Default)]
struct Metric { calls: u64, ns: u128 }

fn timed<T>(stats: &mut BTreeMap<String, Metric>, name: &str, f: impl FnOnce() -> T) -> T {
    let start = Instant::now();
    let result = f();
    let ns = start.elapsed().as_nanos();
    let metric = stats.entry(name.to_owned()).or_default();
    metric.calls += 1;
    metric.ns += ns;
    result
}

fn adler32(bytes: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for chunk in bytes.chunks(5552) {
        for &byte in chunk { a += u32::from(byte); b += a; }
        a %= 65521; b %= 65521;
    }
    (b << 16) | a
}

fn main() {
    let paths: Vec<_> = std::env::args().skip(1).collect();
    assert!(!paths.is_empty(), "pass game trace JSONL paths");
    let mut stats = BTreeMap::new();
    let (mut requests, mut bytes) = (0u64, 0u64);
    let started = Instant::now();
    for path in &paths {
        let mut lines = BufReader::new(File::open(path).unwrap()).lines();
        let first: Value = serde_json::from_str(&lines.next().unwrap().unwrap()).unwrap();
        let config: Config = serde_json::from_value(first["config"].clone()).unwrap();
        let mut game = timed(&mut stats, "game_initialization", || Game::new(config).unwrap());
        for line in lines {
            let record: Value = serde_json::from_str(&line.unwrap()).unwrap();
            let input = record["request"].as_str().unwrap();
            let request: Request = timed(&mut stats, "request_json_parsing", || serde_json::from_str(input).unwrap());
            let mut response = Response { schema: 2, revision: game.revision(),
                observation: None, snapshot: None, error: None, capacity: None };
            let (op, viewer) = match request {
                Request::Observe { viewer } => ("observe", viewer),
                Request::Step { actor, revision, index, viewer } => {
                    timed(&mut stats, "rules_apply/step", || game.apply_index(actor, revision, index)).unwrap();
                    ("step", viewer)
                }
                Request::Act { actor, revision, action, viewer } => {
                    assert_eq!(revision, game.revision());
                    timed(&mut stats, "rules_apply/act", || game.apply(actor, action)).unwrap();
                    ("act", viewer)
                }
                _ => panic!("trace contains an operation outside the collection protocol"),
            };
            response.observation = Some(timed(&mut stats, &format!("observation_construction/{op}"), || game.observe(viewer)));
            response.revision = game.revision();
            let output = timed(&mut stats, &format!("response_json_serialization/{op}"), || serde_json::to_string(&response).unwrap());
            // Timing excludes validation and trace-file IO. This paired checksum
            // and byte-length check verifies wire equivalence, not authenticity.
            assert_eq!(output.len() as u64, record["response_bytes"].as_u64().unwrap(), "{path}, request {requests}");
            assert_eq!(adler32(output.as_bytes()) as u64, record["response_adler32"].as_u64().unwrap(), "{path}, request {requests}");
            requests += 1;
            bytes += output.len() as u64;
            timed(&mut stats, "response_deallocation", || drop((response, output)));
        }
    }
    let stages: BTreeMap<_, _> = stats.iter().map(|(name, metric)|
        (name, json!({"calls": metric.calls, "seconds": metric.ns as f64 / 1e9}))).collect();
    println!("{}", serde_json::to_string_pretty(&json!({
        "trace_files": paths, "games": paths.len(), "requests_verified": requests,
        "response_bytes_verified": bytes, "stages": stages,
        "replay_wall_seconds_including_trace_io_and_validation": started.elapsed().as_secs_f64(),
        "notes": "Single-thread replay against the pinned release rlib. Stage timings exclude trace IO, checksum verification, FFI string copying and Python. They are separate measurements, not additive to live collector timings."
    })).unwrap());
}
