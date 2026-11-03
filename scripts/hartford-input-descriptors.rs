//! Narrow, typed descriptors for the experimental input model. No rules changes.
use rustmage_core::cards::*;
use crate::CATALOGUE_VERSION;
use serde_json::{json, Value};

fn entry(rule: EntryTapped) -> Value {
    match rule {
        EntryTapped::Never => json!({"kind":"never"}),
        EntryTapped::Always => json!({"kind":"always"}),
        EntryTapped::UnlessAtMostTwo => json!({"kind":"land_count_at_most", "threshold":2}),
        EntryTapped::UnlessAtLeastTwo => json!({"kind":"land_count_at_least", "threshold":2}),
        EntryTapped::PayLife(n) => json!({"kind":"pay_life", "amount":n}),
        EntryTapped::UnlessBasic => json!({"kind":"controls_basic"}),
        EntryTapped::UnlessPlainsOrIsland => json!({"kind":"controls_subtype", "subtypes":["Plains","Island"]}),
        EntryTapped::UnlessControlsEitherSubtype(a,b) => json!({"kind":"controls_subtype", "subtypes":[a,b]}),
        EntryTapped::UnlessControlsSubtype(a) => json!({"kind":"controls_subtype", "subtypes":[a]}),
        EntryTapped::OpponentsTurn => json!({"kind":"opponents_turn"}),
        EntryTapped::DuringFirstThreeTurns => json!({"kind":"first_three_turns"}),
        EntryTapped::UnlessLowLife => json!({"kind":"low_life"}),
        EntryTapped::UnlessPlaneswalker => json!({"kind":"controls_planeswalker"}),
        EntryTapped::UnlessBehold(a) => json!({"kind":"behold_subtype", "subtypes":[a]}),
        EntryTapped::ChosenXAtMost(n) => json!({"kind":"chosen_x_at_most", "threshold":n}),
    }
}

fn target(rule: TargetRule) -> Value {
    match rule {
        TargetRule::Spell => json!({"kind":"spell"}),
        TargetRule::SpellKinds { kinds, excluded } => json!({"kind":"spell_kinds", "included":kinds,"excluded":excluded}),
        TargetRule::SpellCharacteristics { colors, min_mana, max_mana } =>
            json!({"kind":"spell_characteristics","colors":colors,"min_mana":min_mana,
                "max_mana":if max_mana==u64::MAX {None} else {Some(max_mana)}}),
        _ => json!({"kind":"other"}),
    }
}

fn spell(plan: TargetPlan, ops: &[Instruction], mode: Option<usize>) -> Value {
    let mut targets=Vec::new(); let mut i=0;
    while let Some(t)=plan.group(i) {
        targets.push(json!({"min":t.min,"max":t.max,"rule":target(t.rule)})); i+=1;
    }
    let mut counters=Vec::new();
    for (pc,op) in ops.iter().enumerate() {
        match *op {
            Instruction::Counter => counters.push(json!({"kind":"hard","pc":pc})),
            Instruction::CounterUnlessPay { cost, exile, unpaid_skip, .. } =>
                counters.push(json!({"kind":"unless_pay","pc":pc,"generic":cost.generic,
                    "colored":cost.colored,"exile":exile,"unpaid_skip":unpaid_skip})),
            Instruction::CounterUnlessPayScaled { exile, .. } =>
                counters.push(json!({"kind":"unless_pay_dynamic","pc":pc,"exile":exile})),
            _ => (),
        }
    }
    // Only a single direct counter instruction is represented completely here.
    // Other programs retain an explicit partial marker and their old traits.
    let complete=ops.len()==1 && counters.len()==1 && targets.len()==1 &&
        targets[0]["rule"]["kind"]!="other" && mode.is_none();
    json!({"mode":mode,"targets":targets,"counters":counters,"simple_counter_complete":complete})
}

pub fn export() -> Value {
    let cards:Vec<Value>=Card::catalogue().map(|card| {
        let rules=authored::rules(card).expect("catalogue rule");
        let spells=match rules.effect {
            Effect::Spell(plan,ops) => vec![spell(plan,ops,None)],
            Effect::Modal(modes) => modes.iter().enumerate().map(|(i,m)|spell(m.targets,m.program,Some(i))).collect(),
            _ => Vec::new(),
        };
        json!({"index":card.index(),"identity":card.identity(),"entry":entry(rules.entry),"spells":spells})
    }).collect();
    json!({"catalogue_version":CATALOGUE_VERSION,"cards":cards})
}
