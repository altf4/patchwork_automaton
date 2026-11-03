//! Numerical Standard v6 tensors constructed directly from typed private views.
use crate::{Encoder, Result, arrays::{Array,Arrays}, choices::Choice, entity};
use crate::actions::{self, ActionFields, Value};
use rustmage::{Observation, Player, SpellView, Target, Zone, EntryChoiceValue, CombatDefender,
    SearchDestination, ZoneReplacementKind};
use std::collections::{HashMap, HashSet};

const SEM: usize=252;
const ENT: usize=268;
const CAND: usize=2672;
const CTX: usize=409;
const HIST: usize=292;
const STEPS: [&str;13]=["Pregame","Upkeep","Draw","Main1","BeginCombat","Attackers","Blockers","FirstStrikeDamage","CombatDamage","EndCombat","Main2","End","Cleanup"];
const DECISIONS: [&str;95]=["attack_cost", "draw_replacement", "miracle_reveal", "resolution_casts", "mana_lot_spend", "zone_replacement", "replacement", "created_token", "resolution_mana", "resolution_life", "connive_order", "opening_hand", "mulligan", "bottom", "priority", "plot_payment", "face_up_payment", "role_rule", "legend_rule", "retarget", "target_amount", "targets", "offspring", "casualty", "mana_rollback", "cast_costs", "casualty_payment", "bargain", "kicker", "gift", "endure", "harmonize", "convoke", "improvise", "waterbend", "mobilize", "blight_intent", "blight_payment", "teamwork_intent", "teamwork_payment", "saddle", "crew", "explore", "manifest", "resolution_x", "source_counter_amount", "target_counter_removal", "x", "evidence_intent", "evidence", "spell_cost", "activation_cost", "spell_mode", "additional_cost", "pay_costs", "order_triggers", "optional_effect", "target_library", "mana_value_parity", "linked_exile", "forage_method", "forage", "mana_choice", "library_choice", "equipment_attachment", "library_selection", "hand_card_choice", "revealed_discard", "recover_milled", "milled_selection", "creature_type_choice", "hand_library_search", "library_graveyard_search", "library_search_destination", "library_search", "effect_discard", "outside_game_choice", "outside_entry_replacement", "hand_defender", "token_choice", "keyword_choice", "proliferate", "effect_choice", "entry_timestamps", "linked_choice", "entry_value", "land_card_name", "entry_life", "entry_behold", "entry_replacement", "aura_entry", "attackers", "blockers", "damage", "discard"];
const BASICS: [&str;5]=["Plains","Island","Swamp","Mountain","Forest"];
const DESTS: [&str;5]=["Top","Bottom","Graveyard","Shuffle","SecondFromTop"];
const FACES: [&str;4]=["Normal","Adventure","Omen","Back"];
const PERMANENTS: [&str;6]=["Creature","Artifact","Enchantment","Planeswalker","Land","Battle"];

const NUMBER_FIELDS: [&str;9]=["mode", "cost", "clause", "index", "effect", "type_mask", "destination_count", "replacement_ability", "grant_timestamp"];
const BOOL_FIELDS: [&str;8]=["keep", "graveyard", "intrinsic", "hand", "library", "flash_surcharge", "collect", "put_counters"];
const CATEGORIES: [(&str,&[&str]);13]=[
    ("pile", &["FaceDown", "FaceUp"]),
    ("effect_kind", &["RevealShuffle", "Prevent", "Exile", "ExileReflexive", "Battlefield"]),
    ("prevention", &["ShieldCounter", "CombatBySource", "CombatToRecipient", "ProtectionEverything", "ProtectionInstantSorcery", "SourceLinkedDamage"]),
    ("contribution", &["Generic", "White", "Blue", "Black", "Red", "Green"]),
    ("parity", &["Odd", "Even"]),
    ("route", &["ExileThree", "Food", "Black"]),
    ("counter", &["Lore", "PlusOne", "MinusOne", "Stun", "Loyalty", "Incubation", "Bait", "Page", "Divinity", "Revival", "Soul", "Fellowship", "Wish", "Stash", "Charge", "Growth", "Plan", "Invasion", "Haste", "Flying", "FirstStrike", "DoubleStrike", "Deathtouch", "Indestructible", "Lifelink", "Menace", "Reach", "Trample", "Vigilance", "Finality", "Shield", "Hone", "Quest", "Defense", "Hexproof", "Blight", "Bloodstain", "Bore", "Chorus", "Coin", "Conqueror", "Decayed", "Dread", "Dream", "Film", "Fire", "Flood", "Impostor", "Ingenuity", "Landmark", "Loot", "Nest", "Net", "Point", "Possession", "Rev", "Skewer", "Supply", "Time", "Unlock"]),
    ("creature_type", &["Advisor", "Aetherborn", "Alien", "Ally", "Angel", "Antelope", "Ape", "Archer", "Archon", "Armadillo", "Army", "Artificer", "Assassin", "Assembly-Worker", "Astartes", "Atog", "Aurochs", "Avatar", "Azra", "Badger", "Balloon", "Barbarian", "Bard", "Basilisk", "Bat", "Bear", "Beast", "Beaver", "Beeble", "Beholder", "Berserker", "Bird", "Bison", "Blinkmoth", "Boar", "Bringer", "Brushwagg", "C'tan", "Camarid", "Camel", "Capybara", "Caribou", "Carrier", "Cat", "Centaur", "Child", "Chimera", "Citizen", "Cleric", "Clown", "Cockatrice", "Construct", "Coward", "Coyote", "Crab", "Crocodile", "Custodes", "Cyberman", "Cyclops", "Dalek", "Dauthi", "Demigod", "Demon", "Deserter", "Detective", "Devil", "Dinosaur", "Djinn", "Doctor", "Dog", "Dragon", "Drake", "Dreadnought", "Drix", "Drone", "Druid", "Dryad", "Dwarf", "Echidna", "Efreet", "Egg", "Elder", "Eldrazi", "Elemental", "Elephant", "Elf", "Elk", "Employee", "Eternal", "Eye", "Faerie", "Ferret", "Fish", "Flagbearer", "Fox", "Fractal", "Frog", "Fungus", "Gamer", "Gamma", "Gargoyle", "Germ", "Giant", "Giraffe", "Gith", "Glimmer", "Gnoll", "Gnome", "Goat", "Goblin", "God", "Golem", "Gorgon", "Graveborn", "Gremlin", "Griffin", "Guest", "Hag", "Halfling", "Hamster", "Harpy", "Hedgehog", "Hellion", "Hero", "Hippo", "Hippogriff", "Homarid", "Homunculus", "Horror", "Horse", "Human", "Hydra", "Hyena", "Illusion", "Imp", "Incarnation", "Inhuman", "Inkling", "Inquisitor", "Insect", "Jackal", "Jellyfish", "Juggernaut", "Kangaroo", "Kavu", "Kirin", "Kithkin", "Knight", "Kobold", "Kor", "Kraken", "Kree", "Lamia", "Lammasu", "Leech", "Lemur", "Leviathan", "Lhurgoyf", "Licid", "Lizard", "Llama", "Lobster", "Manticore", "Masticore", "Mercenary", "Merfolk", "Metathran", "Minion", "Minotaur", "Mite", "Mole", "Monger", "Mongoose", "Monk", "Monkey", "Moogle", "Moonfolk", "Mount", "Mouse", "Mutant", "Myr", "Mystic", "Nautilus", "Necron", "Nephilim", "Nightmare", "Nightstalker", "Ninja", "Noble", "Noggle", "Nomad", "Nymph", "Octopus", "Ogre", "Ooze", "Orb", "Orc", "Orgg", "Otter", "Ouphe", "Ox", "Oyster", "Pangolin", "Peasant", "Pegasus", "Pentavite", "Performer", "Pest", "Phelddagrif", "Phoenix", "Phyrexian", "Pilot", "Pincher", "Pirate", "Plant", "Platypus", "Porcupine", "Possum", "Praetor", "Primarch", "Prism", "Processor", "Qu", "Rabbit", "Raccoon", "Ranger", "Rat", "Rebel", "Reflection", "Rhino", "Rigger", "Robot", "Rogue", "Sable", "Salamander", "Samurai", "Sand", "Saproling", "Satyr", "Scarecrow", "Scientist", "Scion", "Scorpion", "Scout", "Sculpture", "Seal", "Serf", "Serpent", "Servo", "Shade", "Shaman", "Shapeshifter", "Shark", "Sheep", "Shi'ar", "Siren", "Skeleton", "Skrull", "Skunk", "Slith", "Sliver", "Sloth", "Slug", "Snail", "Snake", "Soldier", "Soltari", "Sorcerer", "Spawn", "Specter", "Spellshaper", "Sphinx", "Spider", "Spike", "Spirit", "Splinter", "Sponge", "Spy", "Squid", "Squirrel", "Starfish", "Surrakar", "Survivor", "Symbiote", "Synth", "Tentacle", "Tetravite", "Thalakos", "Thopter", "Thrull", "Tiefling", "Time Lord", "Toy", "Treefolk", "Trilobite", "Triskelavite", "Troll", "Turtle", "Tyranid", "Unicorn", "Utrom", "Vampire", "Varmint", "Vedalken", "Villain", "Volver", "Wall", "Walrus", "Warlock", "Warrior", "Weasel", "Weird", "Werewolf", "Whale", "Wizard", "Wolf", "Wolverine", "Wombat", "Worm", "Wraith", "Wurm", "Yeti", "Zombie", "Zubera"]),
    ("defender_kind", &["Player", "Planeswalker", "Battle"]),
    ("entry_kind", &["BasicLandType", "Color", "CreatureType", "ManaValueParity", "CardName"]),
    ("entry_color", &["White", "Blue", "Black", "Red", "Green"]),
    ("entry_parity", &["Odd", "Even"]),
    ("destination_kind", &["Top", "Bottom", "Graveyard", "Shuffle", "SecondFromTop", "HandOrGraveyard", "LinkedExile", "ExileFreeThisTurn", "Hand", "Battlefield", "BattlefieldThenHand"]),
];

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
fn spell_card(item:&SpellView)->Option<u32> {item.source_identity().map(|(object,_)|object.card)}
fn target_index(target:Target, viewer:Player, lookup:&HashMap<u32,usize>, stack:&HashMap<u64,&SpellView>)->Option<usize> {
    match target {
        Target::Player(p)=>Some((p!=viewer) as usize),
        Target::Card(i)|Target::Spell(i)=>lookup.get(&i).copied(),
        Target::Stack(id)=>stack.get(&id).and_then(|s|spell_card(s)).and_then(|card|lookup.get(&card)).copied(),
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
    let source_name=source_item.and_then(|v|v.source_identity()).map(|(_,card)|card.identity());
    let nested=match action.get("effect") {
        Some(Value::Replacement(ZoneReplacementKind::ExileReflexive { source,.. }))=>Some(source.card),_=>None};
    let obj=reference(action.get("object")).or_else(||reference(action.get("sacrifice")))
        .or_else(||reference(action.get("source"))).or_else(||reference(action.get("equipment"))).or(nested)
        .or_else(||if action.get("card").is_none(){reference(action.get("grantor"))}else{None});
    let card=obj.or_else(||if action.get("card").is_some() {
        scalar_u64(action.get("card")).and_then(|v|u32::try_from(v).ok())
    }else{scalar_u64(action.get("blocker")).and_then(|v|u32::try_from(v).ok()).or_else(||source_item.and_then(spell_card))});
    let target=match action.get("defender") {
        Some(Value::Object(o))=>Some(Target::Card(o.card)),
        Some(Value::Defender(CombatDefender::Player(p)))=>Some(Target::Player(*p)),
        Some(Value::Defender(CombatDefender::Planeswalker(o)|CombatDefender::Battle(o)))=>Some(Target::Card(o.card)),
        _=>match action.get("player").or_else(||action.get("beneficiary")) {
            Some(Value::Player(p))=>Some(Target::Player(*p)),
            _=>match action.get("target"){Some(Value::Target(t))=>Some(*t),_=>None}}};
    let permission=match action.get("permission") {Some(Value::Permission(o,_))=>Some(o.card),v=>reference(v)};
    let other=reference(action.get("graveyard")).or(permission).or_else(||reference(action.get("exiled")))
        .or_else(||if action.get("card").is_some() || action.get("source").is_some(){reference(action.get("grantor"))}else{None})
        .or_else(||scalar_u64(action.get("attacker")).and_then(|v|u32::try_from(v).ok()));
    (card.and_then(|v|lookup.get(&v).copied()),target.and_then(|t|target_index(t,viewer,lookup,stack)),
     other.and_then(|v|lookup.get(&v).copied()),source_name)
}

fn extras(row:&mut Vec<f32>,a:&ActionFields,encoder:&Encoder)->Result<()> {
    match a.get("production") { Some(Value::Mana(m))=>for x in m {entity::number(row,Some(*x as f64))},
        _=>for _ in 0..6 {entity::number(row,None)} }
    for key in NUMBER_FIELDS {
        let value=match key {
            "destination_count"=>match a.get("destination") {
                Some(Value::Destination(SearchDestination::BattlefieldThenHand {battlefield}))=>Some(*battlefield as f64),_=>None},
            "replacement_ability"=>match a.get("effect") {
                Some(Value::Replacement(ZoneReplacementKind::ExileReflexive {ability,..}))=>Some(*ability as f64),_=>None},
            _=>match a.get(key){Some(Value::Integer(v))=>Some(*v as f64),_=>None}};
        entity::number(row,value);
    }
    numeric(row,a.get("value"));
    let mut categories:HashMap<&str,String>=HashMap::new();
    for (key,_) in CATEGORIES {if let Some(Value::Name(v))=a.get(key){categories.insert(key,v.clone());}}
    let mut basic=None;
    let mut choice_card=match a.get("token").or_else(||a.get("card")) {
        Some(Value::Card(c))=>Some(c.identity()),_=>None};
    if let Some(Value::Entry(value))=a.get("value") {
        let kind=match value {
            EntryChoiceValue::BasicLandType(v)=>{basic=Some(format!("{v:?}"));"BasicLandType"},
            EntryChoiceValue::Color(v)=>{categories.insert("entry_color",format!("{v:?}"));"Color"},
            EntryChoiceValue::CreatureType(v)=>{categories.insert("creature_type",v.name().to_string());"CreatureType"},
            EntryChoiceValue::ManaValueParity(v)=>{categories.insert("entry_parity",format!("{v:?}"));"ManaValueParity"},
            EntryChoiceValue::CardName(v)=>{choice_card=Some(v.as_str());"CardName"},
        };
        categories.insert("entry_kind",kind.to_string());
    }
    hot(row,basic.as_deref(),&BASICS);
    let destination=match a.get("destination") {
        Some(Value::Name(value))=>Some(value.clone()),
        Some(Value::Destination(SearchDestination::BattlefieldThenHand{..}))=>None,
        Some(Value::Destination(value))=>Some(format!("{value:?}")),_=>None};
    hot(row,destination.as_deref(),&DESTS);
    for key in BOOL_FIELDS {
        match a.get(key) {Some(Value::Bool(b))=>entity::number(row,Some(*b as u8 as f64)),
            _=>entity::number(row,None)}
    }
    let face=name(a.get("face"));
    if face.is_some_and(|v|!FACES.contains(&v)) {return Err("unsupported cast face".into());}
    hot(row,face,&FACES);
    match a.get("cost") {Some(Value::Cost(c))=>{
        for x in c.colored {entity::number(row,Some(x as f64));}entity::number(row,Some(c.generic as f64));},
        _=>for _ in 0..7 {entity::number(row,None)}}
    let permanent=match a.get("permission") {Some(Value::Permission(_,v))=>Some(format!("{v:?}")),_=>None};
    hot(row,permanent.as_deref().or_else(||name(a.get("permanent_type"))),&PERMANENTS);
    let defender=match a.get("defender") {
        Some(Value::Object(_))=>Some(if a.kind=="AttackBattle"{"Battle"}else{"Planeswalker"}),
        Some(Value::Defender(CombatDefender::Player(_)))=>Some("Player"),
        Some(Value::Defender(CombatDefender::Planeswalker(_)))=>Some("Planeswalker"),
        Some(Value::Defender(CombatDefender::Battle(_)))=>Some("Battle"),_=>None};
    if let Some(value)=defender{categories.insert("defender_kind",value.to_string());}
    if let Some(Value::Replacement(effect))=a.get("effect") {
        let kind=match effect {ZoneReplacementKind::RevealShuffle=>"RevealShuffle",ZoneReplacementKind::Prevent=>"Prevent",
            ZoneReplacementKind::Exile=>"Exile",ZoneReplacementKind::ExileReflexive{..}=>"ExileReflexive",ZoneReplacementKind::Battlefield=>"Battlefield"};
        categories.insert("effect_kind",kind.to_string());
    }
    match a.get("destination") {
        Some(Value::Name(value))=>{categories.insert("destination_kind",value.clone());},
        Some(Value::Destination(value))=>{
            let kind=match value {
                SearchDestination::HandOrGraveyard=>"HandOrGraveyard",SearchDestination::LinkedExile=>"LinkedExile",
                SearchDestination::ExileFreeThisTurn=>"ExileFreeThisTurn",SearchDestination::Graveyard=>"Graveyard",
                SearchDestination::Top=>"Top",SearchDestination::Hand=>"Hand",SearchDestination::Battlefield=>"Battlefield",
                SearchDestination::BattlefieldThenHand{..}=>"BattlefieldThenHand"};
            categories.insert("destination_kind",kind.to_string());
        },_=>{}}
    for (key,allowed) in CATEGORIES {
        let value=categories.get(key).map(String::as_str);
        if value.is_some_and(|v|!allowed.contains(&v)){return Err(format!("unsupported {key}: {value:?}"));}
        hot(row,value,allowed);
    }
    row.push(choice_card.is_some() as u8 as f32);
    if let Some(card)=choice_card{row.extend(traits(encoder,Some(card)));}else{row.extend([0.;SEM]);}
    match a.get("counter_payment") {
        Some(Value::Counters(values))=>for x in &values[5..] {entity::number(row,Some(*x as f64));},
        _=>for _ in entity::NAMED_COUNTERS {entity::number(row,Some(0.));}
    }
    Ok(())
}

pub fn encode(obs:&Observation,choices:&[Choice],encoder:&Encoder)->Result<Arrays> {
    if obs.actor!=Some(obs.viewer) || obs.outcome.is_some() {return Err("private decision required".into());}
    let decision=match obs.decision {"optional_search"=>"optional_effect","attack_payment"=>"attack_cost",v=>v};
    if !DECISIONS.contains(&decision) {return Err(format!("unsupported decision {}",obs.decision));}
    let step=format!("{:?}",obs.step);
    if !STEPS.contains(&step.as_str()) {return Err("unsupported step".into());}
    let legal_ids:HashSet<_>=choices.iter().filter_map(|choice| match choice.action {
        rustmage::Action::ChooseSearch{card}|rustmage::Action::ChooseSearchBattlefield{card}
        |rustmage::Action::SelectLibrary{card}|rustmage::Action::OrderLibraryBottom{card}
        |rustmage::Action::ArrangeLibrary{card,..}=>Some(card),_=>None
    }).collect();
    let mut objects:Vec<_>=obs.objects.iter().collect();
    let mut existing:HashSet<_>=objects.iter().map(|o|o.id).collect();
    // ObjectViews in these actor-private pools are already redacted by RustMage.
    // Admit only cards that this exact decision offers as a legal action.
    for object in obs.search_choice.iter().flat_map(|p|p.cards.iter())
        .chain(obs.library_choice.iter().flat_map(|p|p.remaining.iter()))
        .chain(obs.library_selection.iter().flat_map(|p|p.remaining.iter())) {
        if legal_ids.contains(&object.id) && existing.insert(object.id) {objects.push(object);}
    }
    let n=objects.len()+2;let count=choices.len();
    let mut rows=Vec::with_capacity(n);let mut all_traits=Vec::with_capacity(n);
    rows.push(entity::player_row(&obs.players[obs.viewer.index()],0));
    rows.push(entity::player_row(&obs.players[obs.viewer.other().index()],1));
    all_traits.push(vec![0.;SEM]);all_traits.push(vec![0.;SEM]);
    let mut lookup=HashMap::with_capacity(n);
    for (i,o) in objects.iter().enumerate() {
        lookup.insert(o.id,i+2);
        rows.push(entity::object_row(o,(o.controller!=obs.viewer) as usize,&encoder.catalogue));
        all_traits.push(traits(encoder,Some(o.card.identity())).to_vec());
    }
    let stack:HashMap<u64,&SpellView>=obs.stack.iter().chain(obs.waiting_triggers.iter()).map(|s|(s.id,s)).collect();
    let pending=obs.announcing.as_ref().or(obs.resolving.as_ref());
    let mut relations=vec![0.;n*n*4];
    for (i,o) in objects.iter().enumerate() {
        relations[((i+2)*n+(o.controller!=obs.viewer) as usize)*4]=1.;
        if let Some(attached)=o.attached_to.and_then(|x|lookup.get(&x.card)) {relations[((i+2)*n+*attached)*4+1]=1.;}
    }
    for s in obs.stack.iter().chain(obs.waiting_triggers.iter()).chain(pending) {
        if let Some(&source)=spell_card(s).and_then(|card|lookup.get(&card)) {
            for t in &s.targets {
                if let Some(target)=target_index(t.target,obs.viewer,&lookup,&stack) {relations[(source*n+target)*4+2]=1.;}
            }
        }
    }
    for pair in obs.stack.windows(2) {
        if let (Some(&i),Some(&j))=(spell_card(&pair[0]).and_then(|card|lookup.get(&card)),spell_card(&pair[1]).and_then(|card|lookup.get(&card))) {relations[(i*n+j)*4+3]=1.;}
    }
    for (blocker,attacker) in &obs.blocks {
        if let (Some(&i),Some(&j))=(lookup.get(blocker),lookup.get(attacker)) {relations[(i*n+j)*4+1]=1.;}
    }
    let mut candidates=Vec::with_capacity(count*CAND);let mut refs=vec![0.;count*n];
    for (ci,choice) in choices.iter().enumerate() {
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
        match a.get("counter_payment") {Some(Value::Counters(values))=>for x in &values[..5] {entity::number(&mut candidates,Some(*x as f64))},
            _=>for _ in 0..5 {entity::number(&mut candidates,Some(0.))}}
        extras(&mut candidates,&a,encoder)?;
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
    if let Some((_,card))=pending.and_then(|p|p.source_identity()) {context.extend(traits(encoder,Some(card.identity())))} else {context.extend([0.;SEM])}
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
