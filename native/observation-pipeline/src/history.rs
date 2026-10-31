use crate::arrays::{Array, Arrays};
use crate::encode::{hot, list, numeric, one};
use crate::{Result, config};
use serde_json::Value;
use std::collections::HashMap;

struct Object {
    id: u64,
    card: usize,
    unknown_identity: Option<String>,
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
    pending: Vec<Vec<f32>>,
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
        card: usize,
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
        let c = config();
        let mut row = Vec::with_capacity(c.feature_spec.history);
        one(&mut row, kind, 7);
        one(&mut row, side, 3);
        hot(&mut row, old, &config().history_zones);
        hot(&mut row, new, &config().history_zones);
        numeric(&mut row, &value);
        numeric(&mut row, &delta);
        numeric(&mut row, turn);
        // Descriptor IDs are bookkeeping only. Model history receives the same
        // fixed numerical traits as current entities, never a card-name lookup.
        if kind <= 2 {
            row.extend(c.card_traits(card));
        } else {
            row.resize(row.len() + c.feature_spec.semantics, 0.);
        }
        assert_eq!(row.len(), c.feature_spec.history);
        self.pending.push(row);
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
                let c = config();
                let card = c.object_card(o);
                Ok(Object {
                    id: o["id"].as_u64().ok_or("Invalid object ID")?,
                    card,
                    unknown_identity: if card == 0 {
                        c.object_identity(o).map(str::to_owned)
                    } else {
                        None
                    },
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
                Some(before)
                    if before.card == obj.card
                        && before.unknown_identity == obj.unknown_identity =>
                {
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
        let width = config().feature_spec.history;
        let mut rows = vec![0.; size * width];
        let mut mask = vec![0; size];
        for (i, row) in pending.into_iter().enumerate() {
            mask[i] = 1;
            rows[i * width..(i + 1) * width].copy_from_slice(&row);
        }
        Arrays::new(vec![
            Array::floats(11, &[size, width], rows),
            Array::ints(12, &[size], vec![0; size]),
            Array::bools(13, &[size], mask),
        ])
    }
}
