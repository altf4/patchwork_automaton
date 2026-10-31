//! Numerical Standard v5 tensors constructed directly from typed private views.
use crate::{Encoder, Result, arrays::{Array,Arrays}, choices::Choice, entity};
use crate::actions::{self, ActionFields, Value};
use rustmage::{Action, Observation, Player, SpellView, Target, Zone};
use std::collections::HashMap;

const SEM: usize=252;
const ENT: usize=154;
const CAND: usize=1423;
const CTX: usize=379;
const HIST: usize=292;
const STEPS: [&str;13]=["Pregame","Upkeep","Draw","Main1","BeginCombat","Attackers","Blockers","FirstStrikeDamage","CombatDamage","EndCombat","Main2","End","Cleanup"];
const DECISIONS: [&str;65]=["mulligan","bottom","opening_hand","priority","pay_costs","kicker","targets","attackers","blockers","damage","discard","order_triggers","optional_effect","mana_lot_spend",
"entry_value","entry_life","entry_timestamps","entry_replacement","aura_entry","mana_choice","mana_rollback","spell_mode","spell_cost","activation_cost","additional_cost","cast_costs","offspring","casualty","casualty_payment","bargain","gift","x","resolution_x","source_counter_amount","target_amount","library_search","library_search_destination","library_choice","library_selection","hand_library_search","library_graveyard_search","hand_card_choice","revealed_discard","effect_discard","explore","manifest","legend_rule","role_rule","linked_exile","resolution_mana","effect_mana","resolution_life","attack_cost","plot_payment","resolution_cast","ordered_costs","graveyard_symbol_cost","blight_intent","blight_payment","teamwork_intent","teamwork_payment","effect_choice","retarget","resolution_casts","face_up_payment"];
const BASICS: [&str;5]=["Plains","Island","Swamp","Mountain","Forest"];
const DESTS: [&str;5]=["Top","Bottom","Graveyard","Shuffle","SecondFromTop"];
const FACES: [&str;4]=["Normal","Adventure","Omen","Back"];
const PERMANENTS: [&str;6]=["Creature","Artifact","Enchantment","Planeswalker","Land","Battle"];

fn hot(row:&mut Vec<f32>, value: Option<&str>, names:&[&str]) {
    row.extend(names.iter().map(|n| (Some(*n)==value) as u8 as f32));
}
fn numeric(row:&mut Vec<f32>, value:Option<&Value>) {
    entity::number(row,match value {Some(Value::Integer(v))=>Some(*v as f64),_=>None});
}
fn scalar_u64(value:Option<&Value>)->Option<u64> {
    match value {Some(Value::Integer(v))=>u64::try_from(*v).ok(),_=>None}
}
fn reference(value:Option<&Value>)->Option<u32> {
    match value {Some(Value::Object(o))=>Some(o.card),_=>None}
}
fn name(value:Option<&Value>)->Option<&str> {
    match value {Some(Value::Name(s))=>Some(s),_=>None}
}
fn traits<'a>(encoder:&'a Encoder,name:Option<&str>)->&'a [f32] {
    match name {None=>&[],Some(n)=>encoder.catalogue.traits.get(n).map(Vec::as_slice)
        .unwrap_or(encoder.catalogue.unknown_traits.as_slice())}
}
fn target_index(target:Target, viewer:Player, lookup:&HashMap<u32,usize>, stack:&HashMap<u64,&SpellView>)->Option<usize> {
    match target {
        Target::Player(p)=>Some((p!=viewer) as usize),
        Target::Card(i)|Target::Spell(i)=>lookup.get(&i).copied(),
        Target::Stack(id)=>stack.get(&id).and_then(|s| lookup.get(&s.card)).copied(),
    }
}
fn role(row:&mut Vec<f32>,index:Option<usize>, source_name:Option<&str>,
        all_traits:&[Vec<f32>], rows:&[Vec<f32>],encoder:&Encoder) {
    if let Some(i)=index { row.push(1.);row.extend(&all_traits[i]);row.extend(&rows[i]); }
    else {
        row.push(source_name.is_some() as u8 as f32);
        if source_name.is_some() {row.extend(traits(encoder,source_name));} else {row.extend([0.;SEM]);}
        row.extend([0.;ENT]);
    }
}
fn action_roles(action:&ActionFields,pending:Option<&SpellView>,viewer:Player,
                lookup:&HashMap<u32,usize>, stack:&HashMap<u64,&SpellView>) -> (Option<usize>,Option<usize>,Option<usize>,Option<&'static str>) {
    let source_item=if action.kind=="PutTrigger" {scalar_u64(action.get("id")).and_then(|i|stack.get(&i).copied())} else {pending};
    let source_name=source_item.map(|v|v.source_card.identity());
    let obj=reference(action.get("object")).or_else(||reference(action.get("sacrifice"))).or_else(||reference(action.get("source")));
    let card=obj.or_else(||scalar_u64(action.get("card")).and_then(|v|u32::try_from(v).ok()))
        .or_else(||scalar_u64(action.get("blocker")).and_then(|v|u32::try_from(v).ok()))
        .or_else(||source_item.map(|s|s.card));
    let target=match action.get("beneficiary") {Some(Value::Player(p))=>Some(Target::Player(*p)),_
        =>match action.get("target"){Some(Value::Target(t))=>Some(*t),_=>None}};
    let other=reference(action.get("permission")).or_else(||scalar_u64(action.get("attacker")).and_then(|v|u32::try_from(v).ok()));
    (card.and_then(|v|lookup.get(&v).copied()),target.and_then(|t|target_index(t,viewer,lookup,stack)),
     other.and_then(|v|lookup.get(&v).copied()),source_name)
}
fn extras(row:&mut Vec<f32>,a:&ActionFields)->Result<()> {
    match a.get("production") { Some(Value::Mana(m))=>for x in m {entity::number(row,Some(*x as f64))},
        _=>for _ in 0..6 {entity::number(row,None)} }
    for key in ["mode","cost","clause"] {numeric(row,a.get(key));}
    numeric(row,a.get("value"));
    let basic=match a.get("value") {
        Some(Value::Entry(rustmage::EntryChoiceValue::BasicLandType(t)))=>Some(format!("{t:?}")),
        Some(Value::Entry(other))=>return Err(format!("Unsupported entry choice: {other:?}")),_=>None};
    hot(row,basic.as_deref(),&BASICS);hot(row,name(a.get("destination")),&DESTS);
    for key in ["keep","graveyard","intrinsic","hand","library","flash_surcharge"] {
        match a.get(key) {Some(Value::Bool(b))=>entity::number(row,Some(*b as u8 as f64)),
            Some(Value::Null)|None=>entity::number(row,None),_=>return Err(format!("invalid boolean field {key}"))}
    }
    let face=name(a.get("face"));
    if face.is_some_and(|v|!FACES.contains(&v)) {return Err("unsupported cast face".into());}
    hot(row,face,&FACES);
    match a.get("cost") {Some(Value::Cost(c))=>{
        for x in c.colored {entity::number(row,Some(x as f64));}entity::number(row,Some(c.generic as f64));},
        _=>for _ in 0..7 {entity::number(row,None)}}
    hot(row,name(a.get("permanent_type")),&PERMANENTS);
    Ok(())
}

pub fn encode(obs:&Observation,choices:&[Choice],encoder:&Encoder)->Result<Arrays> {
    if obs.actor!=Some(obs.viewer) || obs.outcome.is_some() {return Err("private decision required".into());}
    let decision=match obs.decision {"optional_search"=>"optional_effect","attack_payment"=>"attack_cost", "harmonize"=>"bargain",v=>v};
    if !DECISIONS.contains(&decision) {return Err(format!("unsupported decision {}",obs.decision));}
    let step=format!("{:?}",obs.step);
    if !STEPS.contains(&step.as_str()) {return Err("unsupported step".into());}
    let n=obs.objects.len()+2;let count=choices.len();
    let mut rows=Vec::with_capacity(n);let mut all_traits=Vec::with_capacity(n);
    rows.push(entity::player_row(&obs.players[obs.viewer.index()],0));
    rows.push(entity::player_row(&obs.players[obs.viewer.other().index()],1));
    all_traits.push(vec![0.;SEM]);all_traits.push(vec![0.;SEM]);
    let mut lookup=HashMap::with_capacity(n);
    for (i,o) in obs.objects.iter().enumerate() {
        lookup.insert(o.id,i+2);
        rows.push(entity::object_row(o,(o.controller!=obs.viewer) as usize,&encoder.catalogue));
        all_traits.push(traits(encoder,Some(o.card.identity())).to_vec());
    }
    let stack:HashMap<u64,&SpellView>=obs.stack.iter().chain(obs.waiting_triggers.iter()).map(|s|(s.id,s)).collect();
    let pending=obs.announcing.as_ref().or(obs.resolving.as_ref());
    let mut relations=vec![0.;n*n*4];
    for (i,o) in obs.objects.iter().enumerate() {
        relations[((i+2)*n+(o.controller!=obs.viewer) as usize)*4]=1.;
        if let Some(attached)=o.attached_to.and_then(|x|lookup.get(&x.card)) {relations[((i+2)*n+*attached)*4+1]=1.;}
    }
    for s in obs.stack.iter().chain(obs.waiting_triggers.iter()).chain(pending) {
        if let Some(&source)=lookup.get(&s.card) {
            for t in &s.targets {
                if let Some(target)=target_index(t.target,obs.viewer,&lookup,&stack) {relations[(source*n+target)*4+2]=1.;}
            }
        }
    }
    for pair in obs.stack.windows(2) {
        if let (Some(&i),Some(&j))=(lookup.get(&pair[0].card),lookup.get(&pair[1].card)) {relations[(i*n+j)*4+3]=1.;}
    }
    for (blocker,attacker) in &obs.blocks {
        if let (Some(&i),Some(&j))=(lookup.get(blocker),lookup.get(attacker)) {relations[(i*n+j)*4+1]=1.;}
    }
    let mut candidates=Vec::with_capacity(count*CAND);let mut refs=vec![0.;count*n];
    for (ci,choice) in choices.iter().enumerate() {
        if obs.decision=="harmonize" && !matches!(choice.action,Action::ChooseHarmonizer {..}) {
            return Err("unexpected Harmonize action".into());
        }
        let a=actions::project(&choice.action)?;
        let ki=encoder.actions.iter().position(|k|k==a.kind).ok_or_else(||format!("unsupported action {}",a.kind))?;
        entity::hot(&mut candidates,Some(ki),encoder.actions.len());
        let (source,target,other,source_name)=action_roles(&a,pending,obs.viewer,&lookup,&stack);
        role(&mut candidates,source,source_name,&all_traits,&rows,encoder);
        role(&mut candidates,target,None,&all_traits,&rows,encoder);
        role(&mut candidates,other,None,&all_traits,&rows,encoder);
        for k in ["ability","amount"] {numeric(&mut candidates,a.get(k));}
        match a.get("payment") {Some(Value::Mana(m))=>for x in m {entity::number(&mut candidates,Some(*x as f64))},
            _=>for _ in 0..6 {entity::number(&mut candidates,None)}}
        for k in ["pay","accept"] {
            match a.get(k) {Some(Value::Bool(b))=>entity::number(&mut candidates,Some(*b as u8 as f64)),
                Some(Value::Null)|None=>entity::number(&mut candidates,None),_=>return Err(format!("invalid {k}"))}
        }
        match a.get("counter_payment") {Some(Value::Counters(values))=>for x in values {entity::number(&mut candidates,Some(*x as f64))},
            _=>for _ in 0..5 {entity::number(&mut candidates,Some(0.))}}
        extras(&mut candidates,&a)?;
        let linked:Vec<_>=[source,target,other].into_iter().flatten().collect();
        for i in &linked {refs[ci*n+*i]+=1.;}
        for v in &mut refs[ci*n..(ci+1)*n] {*v/=linked.len().max(1) as f32;}
    }
    if candidates.len()!=count*CAND {return Err(format!("candidate width mismatch: {}",candidates.len()));}
    let hand:Vec<_>=obs.objects.iter().filter(|o|o.controller==obs.viewer && matches!(o.zone,Zone::Hand)).collect();
    let board:Vec<_>=obs.objects.iter().filter(|o|o.controller==obs.viewer && matches!(o.zone,Zone::Battlefield)).collect();
    let mut context=Vec::with_capacity(CTX);hot(&mut context,Some(&step),&STEPS);hot(&mut context,Some(decision),&DECISIONS);
    context.extend([(obs.active==obs.viewer) as u8 as f32,pending.is_some() as u8 as f32,obs.announcing_payment.is_some() as u8 as f32]);
    for value in [obs.turn as usize,obs.stack.len(),obs.waiting_triggers.len(),hand.len(),
        hand.iter().filter(|o|o.kinds&2!=0).count(),board.iter().filter(|o|o.kinds&2!=0 && !o.tapped).count(),
        board.iter().filter(|o|o.kinds&2==0).count(),hand.iter().filter(|o|o.kinds&1!=0).count()] {
        entity::number(&mut context,Some(value as f64));
    }
    if let Some(p)=pending {context.extend(traits(encoder,Some(p.source_card.identity())))} else {context.extend([0.;SEM])}
    entity::number(&mut context,obs.targeting_group.map(|x|x as f64));
    entity::number(&mut context,pending.and_then(|p|p.x).map(|x|x as f64));
    entity::number(&mut context,Some(pending.and_then(|p|p.cast).and_then(|c|c.kicker).unwrap_or(false) as u8 as f64));
    for vector in [obs.mana_payment.map(|x|x.required),obs.mana_payment.map(|x|x.available)] {
        for i in 0..6 {entity::number(&mut context,vector.map(|v|v[i] as f64));}
    }
    if context.len()!=CTX {return Err(format!("context width mismatch: {}",context.len()));}
    Ok(Arrays::new(vec![
        Array::floats(0,&[n,ENT],rows.into_iter().flatten().collect()),Array::ints(1,&[n],vec![0;n]),
        Array::bools(2,&[n],vec![1;n]),Array::floats(3,&[n,1,SEM],all_traits.into_iter().flatten().collect()),
        Array::bools(4,&[n,1],vec![1;n]),Array::floats(5,&[n,n,4],relations),Array::floats(6,&[CTX],context),
        Array::floats(7,&[count,CAND],candidates),Array::floats(8,&[count,n],refs),Array::bools(9,&[count],vec![1;count]),
        Array::floats(10,&[CAND],vec![0.;CAND]),Array::floats(11,&[1,HIST],vec![0.;HIST]),
        Array::ints(12,&[1],vec![0]),Array::bools(13,&[1],vec![0])]))
}
