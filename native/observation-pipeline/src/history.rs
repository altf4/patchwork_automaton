use crate::arrays::{Array, Arrays};
use crate::encode::{hot, list, numeric, one};
use crate::{Result, config};
use serde_json::Value;
use std::collections::HashMap;

struct Object {
    id: u64,
    card: i32,
    zone: String,
    side: usize,
}
struct State {
    revision: u64,
    objects: Vec<Object>,
    players: Vec<[Option<i64>; 3]>,
    turn: Value,
}
pub struct History {
    previous: Option<State>,
    pending: Vec<(i32, Vec<f32>)>,
    pub limit: usize,
    viewer: Option<String>,
}
impl History {
    pub fn new(limit: usize) -> Self {
        Self {
            previous: None,
            pending: Vec::new(),
            limit,
            viewer: None,
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn event(
        &mut self,
        kind: usize,
        card: i32,
        side: usize,
        old: &str,
        new: &str,
        value: Value,
        delta: Value,
        turn: &Value,
    ) -> Result<()> {
        if self.pending.len() >= self.limit {
            return Err(
                "Neural history overflow; consume decisions or increase the explicit limit".into(),
            );
        }
        let mut row = Vec::with_capacity(40);
        one(&mut row, kind, 7);
        one(&mut row, side, 3);
        hot(&mut row, old, &config().history_zones);
        hot(&mut row, new, &config().history_zones);
        numeric(&mut row, &value);
        numeric(&mut row, &delta);
        numeric(&mut row, turn);
        assert_eq!(row.len(), 40);
        self.pending.push((card, row));
        Ok(())
    }
    pub fn observe(&mut self, raw: &Value) -> Result<()> {
        let viewer = raw["viewer"].as_str().ok_or("Missing private viewer")?;
        if self.viewer.as_deref().is_some_and(|v| v != viewer) {
            return Err("Cannot mix private viewers in one history".into());
        }
        self.viewer = Some(viewer.into());
        let revision = raw["revision"].as_u64().ok_or("Invalid revision")?;
        if self
            .previous
            .as_ref()
            .is_some_and(|p| p.revision >= revision)
        {
            return Ok(());
        }
        if list(&raw["players"]).len() != 2 {
            return Ok(());
        }
        let objects: Vec<_> = list(&raw["objects"])
            .iter()
            .map(|o| {
                Ok(Object {
                    id: o["id"].as_u64().ok_or("Invalid object ID")?,
                    card: config().card(&o["name"]) as i32,
                    zone: o["zone"].as_str().unwrap_or("").to_lowercase(),
                    side: (o["controller"] != viewer) as usize,
                })
            })
            .collect::<Result<_>>()?;
        let players: Vec<_> = list(&raw["players"])
            .iter()
            .map(|p| {
                [
                    p["life"].as_i64(),
                    p["hand_count"].as_i64(),
                    p["library_count"].as_i64(),
                ]
            })
            .collect();
        let previous = self.previous.take();
        let old: HashMap<_, _> = previous
            .as_ref()
            .map(|p| p.objects.iter().map(|o| (o.id, o)).collect())
            .unwrap_or_default();
        let turn = &raw["turn"];
        for obj in &objects {
            match old.get(&obj.id) {
                Some(before) if before.card == obj.card => {
                    if before.zone != obj.zone {
                        self.event(
                            2,
                            obj.card,
                            obj.side,
                            &before.zone,
                            &obj.zone,
                            Value::Null,
                            Value::Null,
                            turn,
                        )?;
                    }
                }
                _ => self.event(
                    0,
                    obj.card,
                    obj.side,
                    "unknown",
                    &obj.zone,
                    Value::Null,
                    Value::Null,
                    turn,
                )?,
            }
        }
        let current: HashMap<_, _> = objects.iter().map(|o| (o.id, ())).collect();
        if let Some(p) = &previous {
            for obj in &p.objects {
                if !current.contains_key(&obj.id) {
                    self.event(
                        1,
                        obj.card,
                        obj.side,
                        &obj.zone,
                        "unknown",
                        Value::Null,
                        Value::Null,
                        turn,
                    )?;
                }
            }
        }
        for (i, p) in players.iter().enumerate() {
            for (stat, &value) in p.iter().enumerate() {
                let before = previous.as_ref().and_then(|p| p.players[i][stat]);
                if value != before {
                    self.event(
                        3 + stat,
                        0,
                        (viewer != format!("P{i}")) as usize,
                        "unknown",
                        "unknown",
                        value.map_or(Value::Null, Value::from),
                        value
                            .zip(before)
                            .map_or(Value::Null, |(v, b)| Value::from(v - b)),
                        turn,
                    )?;
                }
            }
        }
        if previous.as_ref().is_none_or(|p| p.turn != *turn) {
            self.event(
                6,
                0,
                2,
                "unknown",
                "unknown",
                turn.clone(),
                Value::Null,
                turn,
            )?;
        }
        self.previous = Some(State {
            revision,
            objects,
            players,
            turn: turn.clone(),
        });
        Ok(())
    }
    pub fn take(&mut self) -> Arrays {
        let pending = std::mem::take(&mut self.pending);
        let size = pending.len().max(1);
        let mut rows = vec![0.; size * 40];
        let mut cards = vec![0; size];
        let mut mask = vec![0; size];
        for (i, (card, row)) in pending.into_iter().enumerate() {
            cards[i] = card;
            mask[i] = 1;
            rows[i * 40..(i + 1) * 40].copy_from_slice(&row);
        }
        Arrays::new(vec![
            Array::floats(11, &[size, 40], rows),
            Array::ints(12, &[size], cards),
            Array::bools(13, &[size], mask),
        ])
    }
}
