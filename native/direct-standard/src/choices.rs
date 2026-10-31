//! Complete Standard legal-choice expansion from typed, actor-private domains.
use rustmage::{Action, Mana, Observation};

pub struct Choice { pub action: Action, pub finite_index: Option<usize> }
const LIMIT: usize = 4096;

fn add(out: &mut Vec<Choice>, action: Action, finite_index: Option<usize>) -> Result<(),String> {
    if !out.iter().any(|choice| choice.action == action) {
        out.push(Choice { action, finite_index });
        if out.len() > LIMIT { return Err("Standard candidate domain exceeds enumeration bound".into()); }
    }
    Ok(())
}
fn payments(out: &mut Vec<Choice>, required: Mana, available: Mana, left: u32,
            index: usize, mut current: Mana) -> Result<(),String> {
    if index == 6 {
        if left == 0 { add(out, Action::PayCosts { payment: current }, None)?; }
        return Ok(());
    }
    if available[index] < required[index] { return Err("invalid payment bounds".into()); }
    let capacity = (index+1..6).try_fold(0u64, |sum,i| {
        available[i].checked_sub(required[i]).map(|v| sum + u64::from(v)).ok_or("invalid payment bounds")
    })?;
    let low = u64::from(left).saturating_sub(capacity) as u32;
    let high = left.min(available[index]-required[index]);
    for extra in low..=high {
        current[index] = required[index] + extra;
        payments(out,required,available,left-extra,index+1,current)?;
    }
    Ok(())
}

pub fn choices(obs: &Observation, allow_concede: bool) -> Result<Vec<Choice>,String> {
    if obs.actor != Some(obs.viewer) { return Err("viewer is not the decision actor".into()); }
    let mut out=Vec::new();
    for (index,action) in obs.actions.iter().enumerate() {
        if allow_concede || !matches!(action, Action::Concede) { add(&mut out,action.clone(),Some(index))?; }
    }
    if let Some(domain) = &obs.x_choice {
        for interval in &domain.intervals {
            for value in interval.minimum..=interval.maximum {
                add(&mut out,Action::ChooseX { value },None)?;
            }
        }
    }
    if let Some(domain)=obs.damage_assignment {
        for amount in domain.minimum..=domain.maximum {
            add(&mut out,Action::AssignDamage { target: domain.target, amount },None)?;
        }
    }
    if let Some(domain)=obs.target_amount {
        for amount in domain.minimum..=domain.maximum {
            add(&mut out,Action::AllocateTargetAmount { amount },None)?;
        }
    }
    if let Some(domain)=obs.source_counter_choice {
        for amount in 0..=domain.maximum { add(&mut out,Action::ChooseSourceCounterAmount { amount },None)?; }
    }
    if let Some(domain)=&obs.mana_lot_spend {
        if let Some(values)=&domain.values {
            for &amount in values { add(&mut out,Action::ChooseManaLotSpend { amount },None)?; }
        } else {
            for amount in domain.minimum..=domain.maximum { add(&mut out,Action::ChooseManaLotSpend { amount },None)?; }
        }
    }
    if let Some(domain)=obs.mana_payment {
        payments(&mut out,domain.required,domain.available,domain.generic,0,domain.required)?;
    }
    if out.is_empty() { return Err("No playable Standard action".into()); }
    Ok(out)
}
