use crate::arrays::{Array, Arrays};
use crate::{Result, config};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};

#[derive(Deserialize)]
pub struct Config {
    pub fingerprint: String,
    pub schema: u64,
    pub cards: Vec<String>,
    pub semantics: Vec<Vec<f32>>,
    pub mana_values: Vec<f64>,
    pub actions: Vec<(String, Vec<String>)>,
    pub zones: Vec<String>,
    pub steps: Vec<String>,
    pub decisions: Vec<String>,
    pub counters: Vec<String>,
    pub flags: Vec<String>,
    pub player_stats: Vec<String>,
    pub history_zones: Vec<String>,
}
impl Config {
    pub fn card(&self, name: &Value) -> usize {
        name.as_str()
            .and_then(|n| self.cards.iter().position(|c| c == n))
            .map_or(0, |i| i + 1)
    }
}
pub fn list(v: &Value) -> &[Value] {
    v.as_array().map_or(&[], Vec::as_slice)
}
pub fn truth(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
        Value::String(s) => !s.is_empty(),
        Value::Number(n) => n.as_f64() != Some(0.),
    }
}
pub fn numeric(row: &mut Vec<f32>, v: &Value) {
    let number = v
        .as_f64()
        .or_else(|| v.as_str().and_then(|s| s.parse().ok()));
    if let Some(n) = number.filter(|n| n.is_finite()) {
        row.extend([(n.abs().ln_1p().copysign(n) / 5.) as f32, 1.]);
    } else {
        row.extend([0., 0.]);
    }
}
pub fn number(row: &mut Vec<f32>, n: f64) {
    numeric(row, &json!(n));
}
pub fn hot(row: &mut Vec<f32>, value: &str, vocabulary: &[String]) {
    row.extend(vocabulary.iter().map(|v| (v == value) as u8 as f32));
}
pub fn one(row: &mut Vec<f32>, index: usize, count: usize) {
    row.extend((0..count).map(|i| (i == index) as u8 as f32));
}

fn entity(obj: &Value, side: usize, player: bool) -> Vec<f32> {
    let c = config();
    let mut row = Vec::with_capacity(105);
    hot(
        &mut row,
        if player {
            "player".into()
        } else {
            obj["zone"].as_str().unwrap_or("").to_lowercase()
        }
        .as_str(),
        &c.zones,
    );
    one(&mut row, side, 2);
    row.push(player as u8 as f32);
    row.extend(c.flags.iter().map(|k| truth(&obj[k]) as u8 as f32));
    for (key, bits) in [("kinds", 8), ("colors", 6), ("keywords", 28)] {
        let mask = obj[key].as_u64().unwrap_or(0);
        row.extend((0..bits).map(|b| ((mask & (1 << b)) != 0) as u8 as f32));
    }
    for k in ["power", "toughness", "damage"] {
        numeric(&mut row, &obj[k]);
    }
    let id = c.card(&obj["name"]);
    if id > 0 {
        number(&mut row, c.mana_values[id - 1]);
    } else {
        numeric(&mut row, &Value::Null);
    }
    for k in &c.counters {
        numeric(&mut row, obj["counters"].get(k).unwrap_or(&json!(0)));
    }
    for k in &c.player_stats {
        numeric(&mut row, if player { &obj[k] } else { &Value::Null });
    }
    let zero = json!(0);
    for i in 0..6 {
        numeric(
            &mut row,
            if player {
                obj["mana"].get(i).unwrap_or(&zero)
            } else {
                &zero
            },
        );
    }
    assert_eq!(row.len(), 105);
    row
}

#[derive(Clone, Serialize)]
pub struct Choice {
    pub option: String,
    pub value: Value,
    pub action: Value,
}
pub fn choices(raw: &Value, allow_concede: bool) -> Result<Vec<Choice>> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    let mut add = |option: String, value: Value, action: Value| {
        if seen.insert(action.to_string()) {
            out.push(Choice {
                option,
                value,
                action,
            });
        }
    };
    for (i, a) in list(&raw["actions"]).iter().enumerate() {
        if a["kind"] != "Concede" || allow_concede {
            add(format!("action:{i}"), Value::Null, a.clone());
        }
    }
    if !raw["x_choice"].is_null() {
        return Err("Domain outside mono-red schema: domain:x_choice".into());
    }
    for field in ["damage_assignment", "mana_lot_spend", "mana_payment"] {
        let d = &raw[field];
        if d.is_null() {
            continue;
        }
        let mut values = Vec::new();
        if field == "mana_payment" {
            let req = list(&d["required"]);
            let avail = list(&d["available"]);
            if req.len() != 6 || avail.len() != 6 {
                return Err("Invalid payment vectors".into());
            }
            let mut size = 1u64;
            let mut ranges = Vec::new();
            for (lo, hi) in req.iter().zip(avail) {
                let lo = lo.as_u64().ok_or("Invalid payment lower bound")?;
                let hi = hi.as_u64().ok_or("Invalid payment upper bound")?;
                size = size
                    .checked_mul(if hi >= lo { hi - lo + 1 } else { 0 })
                    .ok_or("Payment domain exceeds mono-red enumeration bound")?;
                ranges.push((lo, hi));
            }
            if size > 4096 {
                return Err("Payment domain exceeds mono-red enumeration bound".into());
            }
            let required: u64 = req.iter().map(|v| v.as_u64().unwrap()).sum();
            let total = required + d["generic"].as_u64().ok_or("Invalid generic payment")?;
            for mut index in 0..size {
                let mut pay = vec![0u64; 6];
                for i in (0..6).rev() {
                    let (lo, hi) = ranges[i];
                    pay[i] = lo + index % (hi - lo + 1);
                    index /= hi - lo + 1;
                }
                if pay.iter().sum::<u64>() == total {
                    values.push(json!(pay));
                }
            }
        } else if let Some(v) = d["values"].as_array() {
            values = v.clone();
        } else {
            let lo = d["minimum"]
                .as_u64()
                .ok_or("Numeric domain exceeds mono-red enumeration bound")?;
            let hi = d["maximum"]
                .as_u64()
                .ok_or("Numeric domain exceeds mono-red enumeration bound")?;
            if hi >= lo && hi - lo >= 4096 {
                return Err("Numeric domain exceeds mono-red enumeration bound".into());
            }
            if hi >= lo {
                values = (lo..=hi).map(|i| json!(i)).collect();
            }
        }
        for v in values {
            let a = match field {
                "mana_payment" => json!({"kind":"PayCosts","payment":v}),
                "mana_lot_spend" => json!({"kind":"ChooseManaLotSpend","amount":v}),
                _ => json!({"kind":"AssignDamage","target":d["target"],"amount":v}),
            };
            add(format!("domain:{field}"), v, a);
        }
    }
    if out.is_empty() {
        return Err("No playable action".into());
    }
    Ok(out)
}

fn resolve(
    target: &Value,
    lookup: &HashMap<String, usize>,
    stack: &HashMap<String, &Value>,
) -> Result<Option<usize>> {
    if !truth(target) {
        return Ok(None);
    }
    if let Some(player) = target.get("Player") {
        return lookup
            .get(&player.to_string())
            .copied()
            .map(Some)
            .ok_or_else(|| "Unknown player target".into());
    }
    for key in ["Card", "Spell"] {
        if let Some(v) = target.get(key) {
            return Ok(lookup.get(&v.to_string()).copied());
        }
    }
    if let Some(v) = target.get("Stack") {
        return Ok(stack
            .get(&v.to_string())
            .and_then(|s| lookup.get(&s["card"].to_string()))
            .copied());
    }
    Err("Unknown target representation".into())
}
fn role(row: &mut Vec<f32>, index: Option<usize>, name: &Value, ids: &[i32], rows: &[Vec<f32>]) {
    let c = config();
    if let Some(i) = index {
        row.push(1.);
        one(row, ids[i] as usize, c.cards.len() + 1);
        row.remove(row.len() - c.cards.len() - 1);
        row.extend(&rows[i]);
    } else {
        let id = c.card(name);
        row.push((id > 0) as u8 as f32);
        row.extend((1..=c.cards.len()).map(|i| (i == id) as u8 as f32));
        row.extend([0.; 105]);
    }
}

pub fn encode(raw: &Value, choices: &[Choice]) -> Result<Arrays> {
    let c = config();
    let me = &raw["viewer"];
    if raw["schema"].as_u64() != Some(c.schema) || raw["actor"] != *me || !raw["outcome"].is_null()
    {
        return Err("RustMage decision with supported schema required".into());
    }
    if !c.decisions.iter().any(|v| raw["decision"] == *v)
        || !c.steps.iter().any(|v| raw["step"] == *v)
    {
        return Err("Unsupported RustMage schema/decision/step".into());
    }
    let players = list(&raw["players"]);
    if players.len() != 2 {
        return Err("Two players required".into());
    }
    let viewer = if me == "P0" {
        0
    } else if me == "P1" {
        1
    } else {
        return Err("Invalid viewer".into());
    };
    let objects = list(&raw["objects"]);
    let n = objects.len() + 2;
    let a = choices.len();
    let mut rows = vec![
        entity(&players[viewer], 0, true),
        entity(&players[1 - viewer], 1, true),
    ];
    let mut ids = vec![0, 0];
    let mut lookup = HashMap::new();
    lookup.insert(json!(format!("P{viewer}")).to_string(), 0);
    lookup.insert(json!(format!("P{}", 1 - viewer)).to_string(), 1);
    for (i, obj) in objects.iter().enumerate() {
        let id = c.card(&obj["name"]);
        if id == 0 {
            return Err("Card outside Foundations mono-red vocabulary".into());
        }
        ids.push(id as i32);
        rows.push(entity(obj, (obj["controller"] != *me) as usize, false));
        lookup.insert(obj["id"].to_string(), i + 2);
    }
    let stack: Vec<_> = list(&raw["stack"])
        .iter()
        .chain(list(&raw["waiting_triggers"]))
        .collect();
    let stack_by_id: HashMap<_, _> = stack.iter().map(|s| (s["id"].to_string(), *s)).collect();
    let pending = if truth(&raw["announcing"]) {
        &raw["announcing"]
    } else {
        &raw["resolving"]
    };
    let mut relations = vec![0.; n * n * 4];
    let mut link = |i, j, ch| {
        relations[(i * n + j) * 4 + ch] = 1.;
    };
    for (i, obj) in objects.iter().enumerate() {
        link(
            i + 2,
            *lookup
                .get(&obj["controller"].to_string())
                .ok_or("Unknown controller")?,
            0,
        );
        if let Some(&j) = lookup.get(&obj["attached_to"]["card"].to_string()) {
            link(i + 2, j, 1);
        }
    }
    for s in stack.iter().copied().chain(std::iter::once(pending)) {
        if let Some(&i) = lookup.get(&s["card"].to_string()) {
            for t in list(&s["targets"]) {
                if let Some(j) = resolve(&t["target"], &lookup, &stack_by_id)? {
                    link(i, j, 2);
                }
            }
        }
    }
    for pair in list(&raw["stack"]).windows(2) {
        if let (Some(&i), Some(&j)) = (
            lookup.get(&pair[0]["card"].to_string()),
            lookup.get(&pair[1]["card"].to_string()),
        ) {
            link(i, j, 3);
        }
    }
    for pair in list(&raw["blocks"]) {
        if let (Some(&i), Some(&j)) = (
            lookup.get(&pair[0].to_string()),
            lookup.get(&pair[1].to_string()),
        ) {
            link(i, j, 1);
        }
    }
    let mut candidates = Vec::with_capacity(a * 408);
    let mut refs = vec![0.; a * n];
    for (ci, choice) in choices.iter().enumerate() {
        let action = &choice.action;
        let kind = action["kind"].as_str().ok_or("Missing action kind")?;
        let (ki, (_, fields)) = c
            .actions
            .iter()
            .enumerate()
            .find(|(_, v)| v.0 == kind)
            .ok_or("Unknown native action")?;
        let keys = action.as_object().ok_or("Action must be an object")?;
        if keys.len() != fields.len() + 1 || fields.iter().any(|k| !keys.contains_key(k)) {
            return Err("Unknown native action fields".into());
        }
        let source_item = if kind == "PutTrigger" {
            stack_by_id
                .get(&action["id"].to_string())
                .copied()
                .unwrap_or(&Value::Null)
        } else {
            pending
        };
        let source_id = action
            .get("card")
            .or_else(|| action.get("blocker"))
            .unwrap_or(&source_item["card"]);
        let source = lookup.get(&source_id.to_string()).copied();
        let target = resolve(&action["target"], &lookup, &stack_by_id)?;
        let other = lookup.get(&action["attacker"].to_string()).copied();
        let mut row = Vec::with_capacity(408);
        one(&mut row, ki, c.actions.len());
        role(&mut row, source, &source_item["source_card"], &ids, &rows);
        role(&mut row, target, &Value::Null, &ids, &rows);
        role(&mut row, other, &Value::Null, &ids, &rows);
        for k in ["ability", "amount"] {
            numeric(&mut row, &action[k]);
        }
        for i in 0..6 {
            numeric(&mut row, action["payment"].get(i).unwrap_or(&Value::Null));
        }
        for k in ["pay", "accept"] {
            if let Some(v) = action.get(k) {
                number(&mut row, truth(v) as u8 as f64);
            } else {
                numeric(&mut row, &Value::Null);
            }
        }
        for k in &c.counters {
            numeric(
                &mut row,
                action["counter_payment"].get(k).unwrap_or(&json!(0)),
            );
        }
        assert_eq!(row.len(), 408);
        candidates.extend(row);
        let linked: Vec<_> = [source, target, other].into_iter().flatten().collect();
        for &i in &linked {
            refs[ci * n + i] += 1.;
        }
        for v in &mut refs[ci * n..(ci + 1) * n] {
            *v /= linked.len().max(1) as f32;
        }
    }
    let own: Vec<_> = objects.iter().filter(|o| o["controller"] == *me).collect();
    let hand: Vec<_> = own
        .iter()
        .copied()
        .filter(|o| {
            o["zone"]
                .as_str()
                .unwrap_or("")
                .eq_ignore_ascii_case("hand")
        })
        .collect();
    let board: Vec<_> = own
        .iter()
        .copied()
        .filter(|o| {
            o["zone"]
                .as_str()
                .unwrap_or("")
                .eq_ignore_ascii_case("battlefield")
        })
        .collect();
    let mut context = Vec::with_capacity(87);
    hot(&mut context, raw["step"].as_str().unwrap(), &c.steps);
    hot(
        &mut context,
        raw["decision"].as_str().unwrap(),
        &c.decisions,
    );
    context.extend([
        (raw["active"] == *me) as u8 as f32,
        truth(pending) as u8 as f32,
        truth(&raw["announcing_payment"]) as u8 as f32,
    ]);
    numeric(&mut context, &raw["turn"]);
    for v in [
        list(&raw["stack"]).len(),
        list(&raw["waiting_triggers"]).len(),
        hand.len(),
        hand.iter().filter(|o| o["name"] == "Mountain").count(),
        board
            .iter()
            .filter(|o| o["name"] == "Mountain" && !truth(&o["tapped"]))
            .count(),
        board.iter().filter(|o| o["name"] != "Mountain").count(),
        hand.iter()
            .filter(|o| c.mana_values[c.card(&o["name"]) - 1] == 2.)
            .count(),
    ] {
        number(&mut context, v as f64);
    }
    let pending_id = c.card(&pending["source_card"]);
    context.extend((1..=c.cards.len()).map(|i| (i == pending_id) as u8 as f32));
    numeric(&mut context, &raw["targeting_group"]);
    numeric(&mut context, &pending["x"]);
    number(&mut context, truth(&pending["cast"]["kicker"]) as u8 as f64);
    for k in ["required", "available"] {
        for i in 0..6 {
            numeric(
                &mut context,
                raw["mana_payment"][k].get(i).unwrap_or(&Value::Null),
            );
        }
    }
    assert_eq!(context.len(), 87);
    let mut semantics = vec![0.; n * 25];
    for (i, &id) in ids.iter().enumerate().skip(2) {
        semantics[i * 25..(i + 1) * 25].copy_from_slice(&c.semantics[id as usize - 1]);
    }
    Ok(Arrays::new(vec![
        Array::floats(0, &[n, 105], rows.into_iter().flatten().collect()),
        Array::ints(1, &[n], ids),
        Array::bools(2, &[n], vec![1; n]),
        Array::floats(3, &[n, 1, 25], semantics),
        Array::bools(4, &[n, 1], vec![1; n]),
        Array::floats(5, &[n, n, 4], relations),
        Array::floats(6, &[87], context),
        Array::floats(7, &[a, 408], candidates),
        Array::floats(8, &[a, n], refs),
        Array::bools(9, &[a], vec![1; a]),
        Array::floats(10, &[408], vec![0.; 408]),
        Array::floats(11, &[1, 40], vec![0.; 40]),
        Array::ints(12, &[1], vec![0]),
        Array::bools(13, &[1], vec![0]),
    ]))
}
