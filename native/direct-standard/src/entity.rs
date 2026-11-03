//! Standard v6 entity features from RustMage's already-redacted typed views.
use rustmage::{BasicLandType, EntryChoiceValue, Keywords, ObjectView, PlayerView, Zone, CounterKind};
use std::collections::HashMap;

pub const ENTITY_WIDTH: usize = 268;
pub const NAMED_COUNTERS: [CounterKind;55] = [CounterKind::Incubation, CounterKind::Bait, CounterKind::Page, CounterKind::Divinity, CounterKind::Revival, CounterKind::Soul, CounterKind::Fellowship, CounterKind::Wish, CounterKind::Stash, CounterKind::Charge, CounterKind::Growth, CounterKind::Plan, CounterKind::Invasion, CounterKind::Haste, CounterKind::Flying, CounterKind::FirstStrike, CounterKind::DoubleStrike, CounterKind::Deathtouch, CounterKind::Indestructible, CounterKind::Lifelink, CounterKind::Menace, CounterKind::Reach, CounterKind::Trample, CounterKind::Vigilance, CounterKind::Finality, CounterKind::Shield, CounterKind::Hone, CounterKind::Quest, CounterKind::Defense, CounterKind::Hexproof, CounterKind::Blight, CounterKind::Bloodstain, CounterKind::Bore, CounterKind::Chorus, CounterKind::Coin, CounterKind::Conqueror, CounterKind::Decayed, CounterKind::Dread, CounterKind::Dream, CounterKind::Film, CounterKind::Fire, CounterKind::Flood, CounterKind::Impostor, CounterKind::Ingenuity, CounterKind::Landmark, CounterKind::Loot, CounterKind::Nest, CounterKind::Net, CounterKind::Point, CounterKind::Possession, CounterKind::Rev, CounterKind::Skewer, CounterKind::Supply, CounterKind::Time, CounterKind::Unlock];
#[derive(Default, PartialEq)]
pub struct Catalogue { pub mana_values: HashMap<String, Option<f64>>, pub traits: HashMap<String, Vec<f32>>, pub identities: HashMap<String, String>, pub unknown_traits: Vec<f32> }

pub(crate) fn number(row: &mut Vec<f32>, value: Option<f64>) {
    match value.filter(|x| x.is_finite()) {
        Some(x) => row.extend([(x.abs().ln_1p().copysign(x) / 5.) as f32, 1.]),
        None => row.extend([0., 0.]),
    }
}
pub(crate) fn hot(row: &mut Vec<f32>, index: Option<usize>, width: usize) {
    row.extend((0..width).map(|i| (Some(i) == index) as u8 as f32));
}
fn zone_index(zone: Zone) -> Option<usize> {
    match zone { Zone::Hand => Some(1), Zone::Battlefield => Some(2), Zone::Graveyard => Some(3),
        Zone::Exile => Some(4), Zone::Stack => Some(5), Zone::Command => Some(6), Zone::Library => Some(7),
        Zone::Outside => None }
}
fn keywords(row: &mut Vec<f32>, value: Keywords) {
    let flags = [Keywords::FLYING, Keywords::REACH, Keywords::HASTE, Keywords::VIGILANCE,
        Keywords::DEFENDER, Keywords::FLASH, Keywords::LIFELINK, Keywords::DEATHTOUCH,
        Keywords::TRAMPLE, Keywords::FIRST_STRIKE, Keywords::DOUBLE_STRIKE, Keywords::CANT_BLOCK,
        Keywords::HEXPROOF, Keywords::INDESTRUCTIBLE, Keywords::CANT_BE_BLOCKED, Keywords::MENACE,
        Keywords::MAX_ONE_BLOCKER, Keywords::MIN_THREE_BLOCKERS, Keywords::NO_TOKEN_BLOCKERS,
        Keywords::CHANGELING, Keywords::SKULK, Keywords::NO_POWER_LE_TWO_BLOCKERS,
        Keywords::NO_POWER_GE_THREE_BLOCKERS, Keywords::NO_GREATER_POWER_BLOCKERS,
        Keywords::PREVENT_COMBAT_TO, Keywords::PREVENT_COMBAT_BY, Keywords::NO_HUMAN_BLOCKERS,
        Keywords::PROTECTION_EVERYTHING, Keywords::PROTECTION_INSTANT_SORCERY,
        Keywords::ONLY_HASTE_BLOCKERS, Keywords::CREW_SADDLE_PLUS_TWO, Keywords::SPLIT_SECOND];
    for flag in flags { row.push(value.contains(flag) as u8 as f32); }
    row.extend(std::iter::repeat_n(0., 64 - flags.len()));
}
fn counters(row: &mut Vec<f32>, values: [u64; 5]) {
    for x in values { number(row, Some(x as f64)); }
}
fn missing_stats(row: &mut Vec<f32>) { for _ in 0..8 { number(row, None); } }
fn mana(row: &mut Vec<f32>, values: [u32; 6]) { for x in values { number(row, Some(x as f64)); } }

pub fn player_row(player: &PlayerView, side: usize) -> Vec<f32> {
    let mut row = Vec::with_capacity(ENTITY_WIDTH);
    hot(&mut row, Some(0), 8); hot(&mut row, Some(side), 2); row.push(1.);
    row.extend([0.; 6 + 16 + 6 + 64]);
    for _ in 0..4 { number(&mut row, None); }
    counters(&mut row, [0; 5]);
    for x in [player.life as f64, player.hand_count as f64, player.library_count as f64,
        player.poison_counters as f64, player.spells_cast_this_turn as f64,
        player.cards_drawn_this_turn as f64, player.life_lost_this_turn as f64,
        player.extra_land_plays_remaining as f64] { number(&mut row, Some(x)); }
    mana(&mut row, player.mana);
    hot(&mut row, None, 5);
    for _ in NAMED_COUNTERS { number(&mut row, Some(0.)); }
    number(&mut row, Some(player.speed as f64));
    number(&mut row, Some(player.speed_triggered_this_turn as u8 as f64));
    assert_eq!(row.len(), ENTITY_WIDTH);
    row
}

pub fn object_row(object: &ObjectView, side: usize, catalogue: &Catalogue) -> Vec<f32> {
    let mut row = Vec::with_capacity(ENTITY_WIDTH);
    hot(&mut row, zone_index(object.zone), 8); hot(&mut row, Some(side), 2); row.push(0.);
    row.extend([object.tapped, object.summoning_sick, object.attacking, object.damaged_this_turn,
        object.dealt_damage_this_turn, object.modified].map(|v| v as u8 as f32));
    for i in 0..16 { row.push(((object.kinds & (1u16 << i)) != 0) as u8 as f32); }
    for i in 0..6 { row.push(((object.colors & (1u8 << i)) != 0) as u8 as f32); }
    keywords(&mut row, object.keywords);
    for x in [object.power, object.toughness, object.damage] { number(&mut row, Some(x as f64)); }
    number(&mut row, catalogue.mana_values.get(object.card.identity()).copied().flatten());
    counters(&mut row, [object.counters.plus_one, object.counters.minus_one, object.counters.stun,
        object.counters.loyalty, object.counters.lore]);
    missing_stats(&mut row); mana(&mut row, [0;6]);
    let basic = match object.entry_choice { Some(EntryChoiceValue::BasicLandType(x)) => Some(match x {
        BasicLandType::Plains=>0, BasicLandType::Island=>1, BasicLandType::Swamp=>2,
        BasicLandType::Mountain=>3, BasicLandType::Forest=>4 }), _=>None };
    hot(&mut row, basic, 5);
    for kind in NAMED_COUNTERS { number(&mut row, Some(object.counters.get(kind) as f64)); }
    number(&mut row, None);number(&mut row, None);
    assert_eq!(row.len(), ENTITY_WIDTH);
    row
}
