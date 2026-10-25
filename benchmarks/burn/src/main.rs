mod model;
mod optim;

use burn::prelude::*;
use model::{Archive, Inputs, Network};
use optim::{Adam, gradients};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};

fn backend() -> String {
    "Flex SIMD Rayon".into()
}

fn device() -> Device {
    Device::flex()
}

struct Writer {
    file: File,
    arrays: BTreeMap<String, Value>,
    offset: usize,
}
impl Writer {
    fn new(path: &Path) -> Self {
        assert!(!path.exists(), "Output already exists");
        fs::create_dir_all(path).unwrap();
        Self {
            file: File::create(path.join("arrays.bin")).unwrap(),
            arrays: BTreeMap::new(),
            offset: 0,
        }
    }
    fn add<const D: usize>(&mut self, name: &str, t: Tensor<D>) {
        let shape = t.dims();
        let values = t.into_data().try_to_vec::<f32>().unwrap();
        self.arrays.insert(
            name.into(),
            json!({"offset": self.offset, "shape": shape.to_vec(), "elements": values.len()}),
        );
        for value in values {
            self.file.write_all(&value.to_le_bytes()).unwrap();
            self.offset += 4;
        }
    }
    fn finish(mut self, path: &Path, losses: Value) {
        self.file.flush().unwrap();
        fs::write(
            path.join("manifest.json"),
            serde_json::to_vec_pretty(&json!({"arrays": self.arrays, "losses": losses})).unwrap(),
        )
        .unwrap();
    }
}

fn materialize(net: &Network, input: &Inputs) {
    let (logits, actor_memory) = net.actor(input);
    let (value, critic_memory) = net.critic(input);
    std::hint::black_box(logits.into_data());
    std::hint::black_box(value.into_data());
    std::hint::black_box(Tensor::cat(vec![actor_memory, critic_memory], 1).into_data());
}

fn validate(archive: &Archive, path: &Path) {
    let device = device();
    let net = Network::load(archive, &device, false);
    let mut writer = Writer::new(path);
    for fixture in &archive.manifest.fixtures {
        let input = Inputs::load(archive, &fixture.name, &device);
        let (logits, actor_memory) = net.actor(&input);
        let (value, critic_memory) = net.critic(&input);
        writer.add(&format!("{}/logits", fixture.name), logits);
        writer.add(&format!("{}/value", fixture.name), value);
        writer.add(
            &format!("{}/memory_out", fixture.name),
            Tensor::cat(vec![actor_memory, critic_memory], 1),
        );
        eprintln!("Validated forward {}", fixture.name);
    }
    let device = device.autodiff();
    let mut losses = serde_json::Map::new();
    for name in ["typical-32", "large-32"] {
        let mut net = Network::load(archive, &device, true);
        let input = Inputs::load(archive, name, &device);
        for actor in [true, false] {
            let (loss, norm, grads) = gradients(&net, &input, actor);
            let kind = if actor { "actor" } else { "critic" };
            losses.insert(format!("{name}/{kind}"), json!({"loss":loss, "norm":norm}));
            for (key, value) in grads {
                writer.add(&format!("{name}/grad/{key}"), value);
            }
            eprintln!("Validated gradients {name}/{kind}: loss {loss}, norm {norm}");
        }
        let mut actor = Adam::initialized(&net, true);
        let mut critic = Adam::initialized(&net, false);
        for step in 1..=3 {
            actor.step(&mut net, &input, true);
            critic.step(&mut net, &input, false);
            for (key, value) in &net.weights {
                writer.add(&format!("{name}/after-{step}/{key}"), value.clone());
            }
        }
        for optimizer in [&actor, &critic] {
            for (key, value) in &optimizer.first {
                writer.add(&format!("{name}/adam-m/{key}"), value.clone());
            }
            for (key, value) in &optimizer.second {
                writer.add(&format!("{name}/adam-v/{key}"), value.clone());
            }
        }
    }
    writer.finish(path, Value::Object(losses));
}

fn cpu_seconds() -> f64 {
    // Linux process CPU time, including Rayon workers.
    static TICKS: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    let ticks = *TICKS.get_or_init(|| {
        let output = std::process::Command::new("getconf")
            .arg("CLK_TCK")
            .output()
            .unwrap();
        assert!(output.status.success());
        String::from_utf8(output.stdout)
            .unwrap()
            .trim()
            .parse()
            .unwrap()
    });
    let text = fs::read_to_string("/proc/self/stat").unwrap();
    let fields: Vec<_> = text
        .rsplit_once(')')
        .unwrap()
        .1
        .split_whitespace()
        .collect();
    (fields[11].parse::<f64>().unwrap() + fields[12].parse::<f64>().unwrap()) / ticks
}

fn bench(archive: &Archive, path: &Path, trials: usize, train_steps: usize) {
    let mut jobs: Vec<_> = archive
        .manifest
        .fixtures
        .iter()
        .map(|f| (f.name.clone(), "inference"))
        .collect();
    for name in ["typical-32", "large-32"] {
        for op in ["actor_step", "critic_step"] {
            jobs.push((name.into(), op));
        }
    }
    let mut rows = Vec::new();
    for (name, op) in jobs {
        let training = op != "inference";
        let device = if training {
            device().autodiff()
        } else {
            device()
        };
        let input = Inputs::load(archive, &name, &device);
        let baseline = Network::load(archive, &device, training);
        let mut net = baseline.clone();
        let mut optimizer = Adam::initialized(&net, op == "actor_step");
        let run = |net: &mut Network, optimizer: &mut Adam| {
            if training {
                std::hint::black_box(optimizer.step(net, &input, op == "actor_step"));
            } else {
                materialize(net, &input);
            }
        };
        for _ in 0..5 {
            run(&mut net, &mut optimizer);
        }
        let iterations = if training {
            train_steps
        } else {
            let start = Instant::now();
            for _ in 0..5 {
                run(&mut net, &mut optimizer);
            }
            (0.3 / (start.elapsed().as_secs_f64() / 5.))
                .round()
                .clamp(5., 256.) as usize
        };
        let mut walls = Vec::new();
        let mut cpus = Vec::new();
        for _ in 0..trials {
            let mut net = baseline.clone();
            let mut optimizer = Adam::initialized(&net, op == "actor_step");
            let cstart = cpu_seconds();
            let start = Instant::now();
            for _ in 0..iterations {
                run(&mut net, &mut optimizer);
            }
            device.sync().unwrap();
            walls.push(start.elapsed().as_secs_f64() / iterations as f64);
            cpus.push((cpu_seconds() - cstart) / iterations as f64);
        }
        let mut sorted = walls.clone();
        sorted.sort_by(f64::total_cmp);
        let median_ms = sorted[sorted.len() / 2] * 1000.;
        let row = json!({"fixture":name,"operation":op,"iterations":iterations,"wall_seconds":walls,"cpu_seconds":cpus,"median_ms":median_ms});
        println!("{row}");
        rows.push(row);
    }
    let max_rss = fs::read_to_string("/proc/self/status")
        .unwrap()
        .lines()
        .find(|l| l.starts_with("VmHWM:"))
        .unwrap()
        .to_string();
    fs::write(path, serde_json::to_vec_pretty(&json!({"framework":"burn","version":"0.22.1","backend":backend(), "rayon_threads":std::env::var("RAYON_NUM_THREADS").ok(),"fixture_binary_sha256":archive.manifest.binary_sha256,"rows":rows,"peak_rss":max_rss,"timing":"Warmed preloaded tensors; inference materializes all outputs; training includes forward/backward/global norm clipping/Adam and norm materialization. Parameter and optimizer reset excluded; work synchronized before stopping timer."})).unwrap()).unwrap();
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert!(
        args.len() >= 4,
        "usage: patchwork-burn-benchmark validate|bench FIXTURE OUTPUT [TRIALS] [TRAIN_STEPS]"
    );
    let archive = Archive::load(&PathBuf::from(&args[2]));
    match args[1].as_str() {
        "validate" => validate(&archive, Path::new(&args[3])),
        "bench" => bench(
            &archive,
            Path::new(&args[3]),
            args.get(4).map(|x| x.parse().unwrap()).unwrap_or(5),
            args.get(5).map(|x| x.parse().unwrap()).unwrap_or(12),
        ),
        other => panic!("Unknown mode {other}"),
    }
}
