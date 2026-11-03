//! Bounded legal-choice prefixes from typed, actor-private domains.
use rustmage::{Action, Mana, Observation};

pub struct Choice { pub action: Action, pub finite_index: Option<usize> }
const LIMIT: usize = 256;

fn add(out: &mut Vec<Choice>, action: Action, finite_index: Option<usize>) -> bool {
    if out.len() >= LIMIT { return true; }
    if !out.iter().any(|choice| choice.action == action) {
        out.push(Choice { action, finite_index });
    }
    out.len() >= LIMIT
}
fn payments(out: &mut Vec<Choice>, required: Mana, available: Mana, left: u32,
            index: usize, mut current: Mana) -> Result<bool,String> {
    if out.len() >= LIMIT { return Ok(true); }
    if index == 6 {
        if left == 0 { return Ok(add(out, Action::PayCosts { payment: current }, None)); }
        return Ok(false);
    }
    if available[index] < required[index] { return Err("invalid payment bounds".into()); }
    let capacity = (index+1..6).try_fold(0u64, |sum,i| {
        available[i].checked_sub(required[i]).map(|v| sum + u64::from(v)).ok_or("invalid payment bounds")
    })?;
    let low = u64::from(left).saturating_sub(capacity) as u32;
    let high = left.min(available[index]-required[index]);
    for extra in low..=high {
        current[index] = required[index] + extra;
        if payments(out,required,available,left-extra,index+1,current)? { return Ok(true); }
    }
    Ok(false)
}

pub fn choices(obs: &Observation, allow_concede: bool) -> Result<Vec<Choice>,String> {
    if obs.actor != Some(obs.viewer) { return Err("viewer is not the decision actor".into()); }
    let mut out=Vec::new();
    for (index,action) in obs.actions.iter().enumerate() {
        if allow_concede || !matches!(action, Action::Concede) { if add(&mut out,action.clone(),Some(index)) { return Ok(out); } }
    }
    if let Some(domain) = &obs.x_choice {
        for interval in &domain.intervals {
            for value in interval.minimum..=interval.maximum {
                if add(&mut out,Action::ChooseX { value },None) { return Ok(out); }
            }
        }
    }
    if let Some(domain)=obs.damage_assignment {
        for amount in domain.minimum..=domain.maximum {
            if add(&mut out,Action::AssignDamage { target: domain.target, amount },None) { return Ok(out); }
        }
    }
    if let Some(domain)=obs.target_amount {
        for amount in domain.minimum..=domain.maximum {
            if add(&mut out,Action::AllocateTargetAmount { amount },None) { return Ok(out); }
        }
    }
    if let Some(domain)=obs.source_counter_choice {
        for amount in 0..=domain.maximum { if add(&mut out,Action::ChooseSourceCounterAmount { amount },None) { return Ok(out); } }
    }
    if let Some(domain)=&obs.mana_lot_spend {
        if let Some(values)=&domain.values {
            for &amount in values { if add(&mut out,Action::ChooseManaLotSpend { amount },None) { return Ok(out); } }
        } else {
            for amount in domain.minimum..=domain.maximum { if add(&mut out,Action::ChooseManaLotSpend { amount },None) { return Ok(out); } }
        }
    }
    if let Some(domain)=obs.mana_payment {
        if payments(&mut out,domain.required,domain.available,domain.generic,0,domain.required)? { return Ok(out); }
    }
    if out.is_empty() { return Err("No playable Standard action".into()); }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_numeric_prefix_stops_without_walking_the_domain() {
        let mut out = Vec::new();
        for value in 0..u32::MAX {
            if add(&mut out, Action::ChooseX { value }, None) { break; }
        }
        assert_eq!(out.len(), LIMIT);
        for (value, choice) in out.iter().enumerate() {
            assert_eq!(choice.action, Action::ChooseX { value: value as u32 });
        }
    }

    #[test]
    fn payment_recursion_stops_at_legal_deterministic_prefix() {
        let mut out = Vec::new();
        let available = [1_000_000_000; 6];
        assert!(payments(&mut out, [0; 6], available, 1_000_000_000, 0, [0; 6]).unwrap());
        assert_eq!(out.len(), LIMIT);
        for (index, choice) in out.iter().enumerate() {
            match choice.action {
                Action::PayCosts { payment } => {
                    assert_eq!(payment.iter().map(|v|u64::from(*v)).sum::<u64>(), 1_000_000_000);
                    assert_eq!(payment, [0, 0, 0, 0, index as u32, 1_000_000_000-index as u32]);
                },
                _ => panic!("Expected a legal payment"),
            }
        }
    }

    #[test]
    fn duplicates_keep_the_first_finite_index() {
        let mut out=Vec::new();
        assert!(!add(&mut out, Action::ChooseX { value: 7 }, Some(3)));
        assert!(!add(&mut out, Action::ChooseX { value: 7 }, Some(4)));
        assert_eq!(out.len(),1);
        assert_eq!(out[0].finite_index,Some(3));
    }
}
