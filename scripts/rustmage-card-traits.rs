//! Project-owned offline visitor, compiled as a child of RustMage's cards module.
//! This is a lossy descriptor, never a second rules executor. No Debug/prose parsing.
use super::*;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Default, Clone, Serialize)]
struct Range {
    min: Option<f64>,
    max: Option<f64>,
    dynamic: bool,
}
impl Range {
    fn add(&mut self, amount: Option<f64>) {
        if let Some(v) = amount {
            self.min = Some(self.min.map_or(v, |x| x.min(v)));
            self.max = Some(self.max.map_or(v, |x| x.max(v)));
        } else {
            self.dynamic = true;
        }
    }
}

#[derive(Default, Clone, Serialize)]
struct Summary {
    flags: BTreeSet<String>,
    amounts: BTreeMap<String, Range>,
    produced_colors: u8,
    known_nodes: usize,
    unknown_nodes: usize,
}
impl Summary {
    fn flag(&mut self, flag: &str) {
        self.flags.insert(flag.into());
    }
    fn unknown(&mut self) {
        self.unknown_nodes += 1;
    }
    fn quantity(&mut self, key: &str, value: Option<f64>) {
        self.amounts.entry(key.into()).or_default().add(value);
        if value.is_none() {
            self.flag("dynamic_amount");
        }
    }
    fn effect(&mut self, key: &str, value: Option<f64>) {
        self.flag(&format!("effect.{key}"));
        self.quantity(key, value);
    }
    fn merge(&mut self, other: &Self) {
        self.flags.extend(other.flags.iter().cloned());
        self.produced_colors |= other.produced_colors;
        self.known_nodes += other.known_nodes;
        self.unknown_nodes += other.unknown_nodes;
        for (key, range) in &other.amounts {
            let dest = self.amounts.entry(key.clone()).or_default();
            if let Some(v) = range.min {
                dest.add(Some(v));
            }
            if let Some(v) = range.max {
                dest.add(Some(v));
            }
            dest.dynamic |= range.dynamic;
        }
    }
}

fn amount(value: AmountValue) -> Option<f64> {
    match value {
        AmountValue::Fixed(v) => Some(v.into()),
        AmountValue::Multiple { value, factor } => amount(*value).map(|x| x * f64::from(factor)),
        AmountValue::Sum(values) => values.iter().map(|v| amount(*v)).sum(),
        _ => None,
    }
}

fn subject(s: &mut Summary, value: Subject) {
    s.flag(match value {
        Subject::Source => "recipient.source",
        Subject::Attached => "recipient.attached",
        Subject::Targets => "recipient.target",
        Subject::Entering => "recipient.permanent",
    });
}
fn player(s: &mut Summary, value: PlayerSet) {
    s.flag("recipient.player");
    let flag = match value {
        PlayerSet::Controller | PlayerSet::SourceOwner => "recipient.self_player",
        PlayerSet::Opponent => "recipient.opponent",
        PlayerSet::Each => "recipient.each",
        PlayerSet::Targeted => "recipient.target",
        _ => {
            s.unknown();
            "recipient.player"
        }
    };
    s.flag(flag);
}
fn relation(s: &mut Summary, value: i8) {
    if value == 0 {
        s.flag("recipient.own");
    } else if value == 1 {
        s.flag("recipient.opposing");
    }
}
fn kinds(s: &mut Summary, mask: u16) {
    if mask & 1 != 0 {
        s.flag("recipient.creature");
    }
    if mask & (1 | 2 | 16 | 32 | 64 | 256) != 0 || mask == 0 {
        s.flag("recipient.permanent");
    }
}
fn target_rule(s: &mut Summary, rule: TargetRule) {
    s.known_nodes += 1;
    match rule {
        TargetRule::None => (),
        TargetRule::Any => {
            s.flag("recipient.player");
            s.flag("recipient.creature");
            s.flag("recipient.permanent");
        }
        TargetRule::Creature | TargetRule::DefendingCreature => s.flag("recipient.creature"),
        TargetRule::CreatureOrVehicle { relation: r } => {
            s.flag("recipient.creature");
            relation(s, r);
        }
        TargetRule::Player { opponent_only } => {
            s.flag("recipient.player");
            if opponent_only {
                s.flag("recipient.opponent");
            }
        }
        TargetRule::Object {
            zone,
            kinds: k,
            relation: r,
            other,
            nonland,
            tapped,
            min_mana,
            max_mana,
        } => {
            kinds(s, k.into());
            relation(s, r);
            if zone == TargetZone::Graveyard {
                s.flag("recipient.graveyard");
            }
            if other || nonland || tapped || min_mana > 0 || max_mana < u32::MAX {
                s.flag("conditional");
            }
        }
        TargetRule::Union(rules) => {
            for r in rules {
                target_rule(s, *r);
            }
        }
        TargetRule::Filtered { base, .. }
        | TargetRule::Grouped { base, .. }
        | TargetRule::ManaValueExactly { base, .. }
        | TargetRule::ManaValueAtMost { base, .. } => {
            target_rule(s, *base);
            s.flag("conditional");
            s.unknown();
        }
        TargetRule::Spell
        | TargetRule::Ability
        | TargetRule::CreatureOrLegendarySpell
        | TargetRule::SingleTargetStackObject
        | TargetRule::SpellKinds { .. }
        | TargetRule::YourSpellKinds { .. }
        | TargetRule::YourAbilityFromSourceKinds { .. }
        | TargetRule::SpellCharacteristics { .. } => s.flag("recipient.stack"),
        TargetRule::GraveCreatureEnteredThisTurn
        | TargetRule::GraveCreatureSourcePower { .. }
        | TargetRule::GraveCreaturePower { .. } => {
            s.flag("recipient.graveyard");
            s.flag("recipient.creature");
            s.flag("conditional");
        }
        TargetRule::DamagedPlayerPermanent { kinds: k } => {
            kinds(s, k.into());
            s.flag("conditional");
        }
    }
}
fn targets(s: &mut Summary, plan: TargetPlan) {
    let mut index = 0;
    while let Some(t) = plan.group(index) {
        if t.max > 0 {
            s.flag("recipient.target");
            target_rule(s, t.rule);
        }
        index += 1;
    }
}
fn cast_filter(s: &mut Summary, filter: CastFilter) {
    match filter {
        CastFilter::Noncreature | CastFilter::NonCreatureOrdinal(_) => {
            s.flag("trigger.noncreature_cast")
        }
        CastFilter::InstantSorcery | CastFilter::InstantSorceryOrdinal(_) => {
            s.flag("trigger.instant_sorcery_cast")
        }
        CastFilter::AnyOf(filters) => {
            for f in filters {
                cast_filter(s, *f);
            }
        }
        CastFilter::Colors { filter, .. }
        | CastFilter::ManaSymbols { filter, .. }
        | CastFilter::Targeted { filter, .. }
        | CastFilter::ManaValueMaximum { filter, .. }
        | CastFilter::ManaValueMinimum { filter, .. } => {
            cast_filter(s, *filter);
            s.flag("conditional");
        }
        CastFilter::Any
        | CastFilter::Creature
        | CastFilter::Sorcery
        | CastFilter::Artifact
        | CastFilter::Enchantment
        | CastFilter::Subtype(_) => (),
        _ => s.unknown(),
    }
}
fn trigger(s: &mut Summary, event: TriggerCondition) {
    s.known_nodes += 1;
    match event {
        TriggerCondition::OncePerTurn(event) => {
            s.flag("once_per_turn");
            trigger(s, *event);
        }
        TriggerCondition::InZone { event, .. }
        | TriggerCondition::Granted(event)
        | TriggerCondition::When { event, .. }
        | TriggerCondition::InterveningIf { event, .. }
        | TriggerCondition::DuringOpponentsTurn(event)
        | TriggerCondition::CastFromHand(event)
        | TriggerCondition::TreasureManaSpent(event)
        | TriggerCondition::ManaSpent { event, .. } => {
            s.flag("conditional");
            trigger(s, *event);
        }
        TriggerCondition::SelfEnters
        | TriggerCondition::SelfNontokenEnters
        | TriggerCondition::SelfEntersIfCast
        | TriggerCondition::SelfOffspring
        | TriggerCondition::SelfKicked
        | TriggerCondition::SelfGiftEnters
        | TriggerCondition::OtherCreatureEntersYou
        | TriggerCondition::CreatureEntersYou
        | TriggerCondition::CreatureEnters { .. }
        | TriggerCondition::PermanentEntersYou { .. }
        | TriggerCondition::ArtifactEntersYou
        | TriggerCondition::EnchantmentEntersYou
        | TriggerCondition::OtherSubtypeEntersYou(_)
        | TriggerCondition::SubtypeEntersYou(_)
        | TriggerCondition::TokensEnterYou => s.flag("trigger.entry"),
        TriggerCondition::SelfDies
        | TriggerCondition::SelfNontokenDies
        | TriggerCondition::SelfCreatureDies
        | TriggerCondition::SelfDiesWithCounters
        | TriggerCondition::CreatureDies { .. }
        | TriggerCondition::CreatureDiesBatch { .. }
        | TriggerCondition::CreatureDiesFiltered { .. }
        | TriggerCondition::AttachedCreatureDies
        | TriggerCondition::PermanentToGraveyard { .. } => s.flag("trigger.death"),
        TriggerCondition::SelfEntersOrDies => {
            s.flag("trigger.entry");
            s.flag("trigger.death");
        }
        TriggerCondition::SelfEntersOrAttacks => {
            s.flag("trigger.entry");
            s.flag("trigger.attack");
        }
        TriggerCondition::SelfAttacks
        | TriggerCondition::SelfFirstAttack
        | TriggerCondition::SelfAttacksAlone
        | TriggerCondition::EquippedAttacks
        | TriggerCondition::GrantedAttacks
        | TriggerCondition::AttackDeclared { .. }
        | TriggerCondition::AttackPlayer(_)
        | TriggerCondition::ControlledCreatureAttacksAlone
        | TriggerCondition::ControlledEquippedCreatureAttacks => s.flag("trigger.attack"),
        TriggerCondition::SpellCast { filter, .. } => {
            s.flag("trigger.cast");
            cast_filter(s, filter);
        }
        TriggerCondition::SelfCast => s.flag("trigger.cast"),
        TriggerCondition::LandEntersYou => s.flag("trigger.landfall"),
        TriggerCondition::SelfTargeted { .. }
        | TriggerCondition::OpponentTargetsControlledCreatureOrSpell
        | TriggerCondition::ControlledAbilityTargetsPlayerOrPermanent => s.flag("trigger.targeted"),
        TriggerCondition::SourceDealtCombatDamage
        | TriggerCondition::AttachedDealtCombatDamage
        | TriggerCondition::PlayerDamaged { .. }
        | TriggerCondition::SelfDamaged
        | TriggerCondition::CreatureDamaged { .. } => s.flag("trigger.damage"),
        TriggerCondition::LifeGained { .. } => s.flag("trigger.lifegain"),
        TriggerCondition::CardDrawn { .. } => s.flag("trigger.draw"),
        TriggerCondition::SelfDiscarded | TriggerCondition::Discarded { .. } => {
            s.flag("trigger.discard")
        }
        TriggerCondition::AtStep { .. } | TriggerCondition::Chapter { .. } => {
            s.flag("trigger.step")
        }
        TriggerCondition::CounterThresholdCrossed { .. }
        | TriggerCondition::StateCounterAtLeast { .. }
        | TriggerCondition::SelfCountersAdded { .. }
        | TriggerCondition::CounterPlacedByYou { .. }
        | TriggerCondition::OtherCreatureCounterPlacedByYou { .. } => s.flag("trigger.counter"),
        TriggerCondition::Ward => s.flag("trigger.ward"),
        _ => s.unknown(),
    }
}
fn mana(s: &mut Summary, value: ManaProduction) {
    match value {
        ManaProduction::Fixed(v) => {
            for (i, amount) in v.iter().enumerate() {
                if *amount > 0 {
                    s.produced_colors |= 1 << i;
                }
            }
            s.effect("mana", Some(v.iter().map(|x| f64::from(*x)).sum()));
        }
        ManaProduction::Choice { mask, amount } | ManaProduction::Combination { mask, amount } => {
            s.produced_colors |= mask;
            s.effect("mana", Some(amount.into()));
        }
        ManaProduction::Tagged { production, .. } => {
            mana(s, *production);
            s.flag("conditional");
        }
        ManaProduction::SourcePower { color } | ManaProduction::CountSubtype { color, .. } => {
            s.produced_colors |= 1 << color;
            s.effect("mana", None);
        }
        ManaProduction::TapSourcePowerChoice { mask } => {
            s.produced_colors |= mask;
            s.effect("mana", None);
        }
        ManaProduction::ChosenColor
        | ManaProduction::MonocoloredPermanentColors
        | ManaProduction::DistinctCreaturePowers => {
            // Possible colors, not a guarantee that the board permits all five.
            s.produced_colors |= 31;
            s.flag("conditional");
            s.effect("mana", None);
        }
    }
}
fn instruction(s: &mut Summary, op: Instruction) -> Option<Value> {
    use Instruction::*;
    s.known_nodes += 1;
    match op {
        Damage(n) => {
            s.effect("damage", Some(n.into()));
            s.flag("recipient.target");
        }
        DamageScaled(a) => {
            s.effect("damage", amount(a));
            s.flag("recipient.target");
        }
        DamagePlayers(p, n) | EventObjectDamagePlayers(p, n) => {
            s.effect("damage", Some(n.into()));
            player(s, p);
        }
        DamagePlayersScaled(p, a) => {
            s.effect("damage", amount(a));
            player(s, p);
        }
        DamageDivided { total } => s.effect("damage", Some(total.into())),
        DamageWithController {
            damage,
            controller_damage,
        } => {
            s.effect("damage", Some(damage.into()));
            s.effect("damage", Some(controller_damage.into()));
        }
        DamageBattlefield { amount, .. } | DamageBattlefieldExceptSubtype { amount, .. } => {
            s.effect("damage", Some(amount.into()));
            s.flag("recipient.each");
        }
        PowerDamage { .. }
        | CreaturePowerDamageAny { .. }
        | ScaledPowerDamage { .. }
        | EventObjectPowerDamage => s.effect("damage", None),
        Draw(p, n) => {
            s.effect("draw", Some(n.into()));
            player(s, p);
        }
        DrawScaled(p, a) => {
            s.effect("draw", amount(a));
            player(s, p);
        }
        DrawMoved | DrawDamageAmount => s.effect("draw", None),
        GainLife(p, n) => {
            s.effect("gain_life", Some(n.into()));
            player(s, p);
        }
        GainLifeScaled(p, a) => {
            s.effect("gain_life", amount(a));
            player(s, p);
        }
        GainLifeMoved => s.effect("gain_life", None),
        LoseLife(p, n) => {
            s.effect("lose_life", Some(n.into()));
            player(s, p);
        }
        LoseLifeScaled(p, a) => {
            s.effect("lose_life", amount(a));
            player(s, p);
        }
        Discard(p, n) => {
            s.effect("discard", Some(n.into()));
            player(s, p);
        }
        DiscardScaled(p, a) => {
            s.effect("discard", amount(a));
            player(s, p);
        }
        DiscardHand(p) | DiscardHandRememberCount(p) => {
            s.effect("discard", None);
            player(s, p);
        }
        DiscardUpToOneDraw => {
            s.effect("discard", Some(1.));
            s.effect("draw", Some(1.));
            s.flag("optional");
        }
        DrawThenDiscardIfDrew { amount: a, discard } => {
            s.effect("draw", amount(a));
            s.effect("discard", Some(discard.into()));
            s.flag("conditional");
        }
        DrainOpponents(n) => {
            s.effect("lose_life", Some(n.into()));
            s.effect("gain_life", Some(n.into()));
        }
        DrainLifeScaled(a) | LoseOpponentGainSame(a) => {
            s.effect("lose_life", amount(a));
            s.effect("gain_life", amount(a));
        }
        CreateTokens { amount, .. }
        | CreateTokensFor { amount, .. }
        | CreateTokensReflexive { amount, .. }
        | CreateTokensWithSourceCounters { amount, .. }
        | CreateEquipmentForSource { amount, .. } => s.effect("token", Some(amount.into())),
        CreateTokensScaled { amount: a, .. }
        | CreateTokensWithTemporaryKeywords { amount: a, .. }
        | Mobilize { amount: a, .. } => s.effect("token", amount(a)),
        CreateTokensWithCounters {
            amount: n,
            counters,
            ..
        } => {
            s.effect("token", Some(n.into()));
            s.effect("counter", amount(counters));
        }
        CreateRole { .. } | CreateRoleReflexive { .. } | JobSelect { .. } => {
            s.effect("token", Some(1.))
        }
        PutCounters {
            subject: sub,
            amount,
            ..
        } => {
            subject(s, sub);
            s.effect("counter", Some(amount.into()));
        }
        PutCountersScaled {
            subject: sub,
            amount: a,
            ..
        } => {
            subject(s, sub);
            s.effect("counter", amount(a));
        }
        PutCreatureCounters { amount, .. }
        | PutPlayerCreatureCounters { amount, .. }
        | PutOtherControlledCreatureCounters { amount, .. } => {
            s.effect("counter", Some(amount.into()))
        }
        PutFilteredCounters { amount, .. } | DistributeCounters { total: amount, .. } => {
            s.effect("counter", Some(amount.into()))
        }
        DistributeCountersX { .. } | DoubleCounters | DoubleCounterKind { .. } => {
            s.effect("counter", None)
        }
        Grow {
            subject: sub,
            power,
            toughness,
        } => {
            subject(s, sub);
            grow(s, power.into(), toughness.into());
        }
        GrowCreatures {
            power, toughness, ..
        }
        | GrowPlayerCreatures {
            power, toughness, ..
        }
        | GrowPermanentGroup {
            power, toughness, ..
        } => {
            s.flag("recipient.each");
            grow(s, power.into(), toughness.into());
        }
        GrowScaled { subject: sub, .. } => {
            subject(s, sub);
            s.effect("buff", None);
        }
        DoubleTargetPower
        | MultiplyTargetStats { .. }
        | GrowCreaturesScaled { .. }
        | GrowPermanentGroupScaled { .. } => s.effect("buff", None),
        GrantKeywords { subject: sub, .. } => {
            subject(s, sub);
            s.flag("effect.grant_keyword");
        }
        GrantCreatures { .. }
        | GrantPermanentsThisTurn(_)
        | GrantPermanentGroupKeywords { .. }
        | GrantCastCreatureKeywords(_)
        | GrantEquippedAttackers(_) => s.flag("effect.grant_keyword"),
        AddMana(value) => mana(s, value),
        AddColorlessScaled(a) => {
            s.produced_colors |= 1 << 5;
            s.effect("mana", amount(a));
        }
        AddManaScaled { color, amount: a } => {
            s.produced_colors |= 1 << color.index();
            s.effect("mana", amount(a));
        }
        AddCombatRedScaled(a) => {
            s.produced_colors |= 1 << 3;
            s.effect("mana", amount(a));
        }
        AddCombatRed(n) => {
            s.produced_colors |= 1 << 3;
            s.effect("mana", Some(n.into()));
        }
        Destroy | DestroyTargetNameGroup | DestroyManaValue { .. } => s.flag("effect.destroy"),
        Exile
        | ExileUntilSourceLeaves
        | ExileGraveTargets
        | ExileGraveTargetsLinked
        | ExileGraveyard(_)
        | ExileTargetsForModifiedCast => s.flag("effect.exile"),
        ReturnToHand
        | ReturnSourceToHand(_)
        | ReturnAllTargetsToHand
        | ReturnEventCardToHand
        | ReturnSourceFromFirstPublicZoneToHand
        | ReturnSpellToHand => s.flag("effect.bounce"),
        ReturnToBattlefield
        | ReturnToBattlefieldTapped
        | ReturnToBattlefieldUnderControl
        | ReturnSourceToBattlefield { .. }
        | ReturnGraveyardCreatures { .. } => s.flag("effect.reanimate"),
        Counter
        | CounterRelated
        | CounterUnlessPay { .. }
        | CounterUnlessPayScaled { .. }
        | CounterPermanentExileCast => s.flag("effect.counter_spell"),
        Tap | TapSource | TapAttached | TapCreatures(_) => s.flag("effect.tap"),
        Untap
        | UntapSource
        | UntapEventObject
        | UntapCreatures(_)
        | UntapPlayerCreatures(_)
        | UntapLands(_) => s.flag("effect.untap"),
        Search(_)
        | SearchRequired(_)
        | SearchTargetController(_)
        | SearchTargetPlayer(_)
        | SearchHandAndOrLibrary(_)
        | SearchDifferentNames(_) => s.flag("effect.search"),
        Scry(n) => s.effect("scry", Some(n.into())),
        Surveil(n) => s.effect("surveil", Some(n.into())),
        Mill(_, _) | MayMill(_) | MillLifeEvent => s.flag("effect.mill"),
        ExileTopPlay { amount, .. }
        | ExileTopChoosePlay { amount, .. }
        | ExileTargetLibraryPlay { amount } => s.effect("exile_play", Some(amount.into())),
        ExileTopPlayScaled { amount: a, .. } => s.effect("exile_play", amount(a)),
        GainControlUntilEndTurn | GainControl { .. } | GainControlTargetsUntilSourceLeaves => {
            s.flag("effect.control")
        }
        CreateCopyToken { .. }
        | CreateCopyTokenWithExceptions { .. }
        | BecomeCopy(_)
        | CopyTargetSpell { .. }
        | CopyTargetAbility { .. } => s.flag("effect.copy"),
        IfKicked { skip } => {
            s.flag("conditional");
            return Some(json!({"kind":"if_kicked","skip":skip}));
        }
        IfGift { promised, skip } => {
            s.flag("conditional");
            return Some(json!({"kind":"if_gift","promised":promised,"skip":skip}));
        }
        IfAmountAtLeast { skip, .. }
        | IfSubject { skip, .. }
        | IfManaSpent { skip, .. }
        | IfAnotherInstantSorceryCast { skip }
        | IfCastFromGraveyard { skip }
        | IfCastFromNonHand { skip } => {
            s.flag("conditional");
            s.unknown();
            return Some(json!({"kind":"conditional","skip":skip}));
        }
        May { skip }
        | MayOncePerTurn { skip }
        | MayPlayer { skip, .. }
        | MayOpponent { skip }
        | MayTargetController { skip }
        | MayPayMana { skip, .. }
        | MayPayManaAlternatives { skip, .. } => {
            s.flag("optional");
            return Some(json!({"kind":"optional","skip":skip}));
        }
        Skip(skip) => return Some(json!({"kind":"skip","skip":skip})),
        SelectGroup(group) => return Some(json!({"kind":"target_group","group":group})),
        _ => {
            s.unknown();
        }
    }
    None
}
fn grow(s: &mut Summary, power: f64, toughness: f64) {
    s.flag("effect.buff");
    s.quantity("power_bonus", Some(power));
    s.quantity("toughness_bonus", Some(toughness));
}
fn program(s: &mut Summary, ops: &[Instruction]) -> Vec<Value> {
    ops.iter()
        .enumerate()
        .map(|(pc, op)| {
            let mut node = Summary::default();
            let control = instruction(&mut node, *op);
            s.merge(&node);
            json!({"pc":pc,"control":control,"summary":node})
        })
        .collect()
}
fn activation_cost(s: &mut Summary, cost: ActivationCost) {
    for (flag, present) in [
        ("tap", cost.tap),
        ("sacrifice", cost.sacrifice),
        ("discard", cost.discard),
        ("exile", cost.exile_self),
        ("life", cost.life > 0),
        ("remove_counter", cost.remove.is_some()),
        ("loyalty", cost.loyalty.is_some()),
        (
            "mana",
            cost.generic > 0 || cost.colored.iter().any(|v| *v > 0),
        ),
    ] {
        if present {
            s.flag(&format!("cost.{flag}"));
        }
    }
    if cost.additional.is_some() {
        s.flag("nonmana_additional_cost");
        s.unknown();
    }
}
fn static_fragment(s: &mut Summary, fragment: StaticAbilityFragment) {
    use StaticAbilityFragment::*;
    s.known_nodes += 1;
    match fragment {
        IntrinsicKeywords(_) | Changeling => (),
        AttachmentStats(stats) | AttachmentTurnStats(stats) => {
            s.flag("recipient.attached");
            grow(s, stats.power as f64, stats.toughness as f64);
        }
        StaticModifier(_, modifier) | ConditionalCreatureModifier { modifier, .. } => {
            grow(
                s,
                modifier.stats.power as f64,
                modifier.stats.toughness as f64,
            );
            s.flag("conditional");
        }
        ScalingModifier(_, modifier) => {
            grow(
                s,
                modifier.stats.power as f64,
                modifier.stats.toughness as f64,
            );
            s.flag("conditional");
            s.flag("dynamic_amount");
        }
        KeywordGrant(_, _)
        | StaticRuleGrant(_, _)
        | AttachmentKeywords(_)
        | AttachedTurnKeywords(_)
        | AttachmentRule(_)
        | SourceConditionalKeywords { .. }
        | SourceCounterKeywords { .. }
        | SourcePlusOnePlacedKeywords(_)
        | CounterKeyword(_, _, _) => {
            s.flag("effect.grant_keyword");
            s.flag("conditional");
        }
        DynamicStats(_, _) | BaseStatSetter(_, _) | LiveBase(_, _) | LiveBonus(_, _) => {
            s.flag("power_dynamic");
            s.flag("toughness_dynamic");
            s.flag("dynamic_amount");
        }
        AttachmentMana(value)
        | AttachmentSacrificeMana(value)
        | LandMana {
            production: value, ..
        } => mana(s, value),
        CostReduction(_, _) | TargetCostReduction(_, _) | EquipTargetReduction(_) => {
            s.flag("cost_reduction")
        }
        AttachmentTrigger(_) | AttachmentActivation(_) => s.flag("recipient.attached"),
        _ => s.unknown(),
    }
}

fn alternative(kind: &str, cost: ManaCost) -> Value {
    json!({"kind":kind,"cost":cost,"mana_value":u64::from(cost.generic) + cost.colored.iter().map(|v| u64::from(*v)).sum::<u64>()})
}

pub fn export() -> Value {
    let cards: Vec<Value> = Card::catalogue().map(|card| {
        let rules = authored::rules(card).expect("catalogue rule"); let extra = rules.extras;
        let mut all = Summary::default(); let mut abilities = Vec::new(); let mut alternatives = Vec::new();
        let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
        let mut add = |kind: &str, slot: usize, mode: Option<usize>, mut summary: Summary, plan: TargetPlan, ops: &[Instruction], cost: Option<Value>| {
            targets(&mut summary, plan); let steps = program(&mut summary, ops);
            all.merge(&summary);
            abilities.push(json!({"kind":kind,"slot":slot,"mode":mode,"cost":cost,"summary":summary,"steps":steps}));
        };
        match rules.effect {
            Effect::Spell(plan, ops) => { add("spell", 0, None, Summary::default(), plan, ops, None); counts.insert("spell_abilities", 1); }
            Effect::Modal(modes) => {
                for (i, mode) in modes.iter().enumerate() { let mut s = Summary::default(); s.flag("modal"); add("spell", 0, Some(i), s, mode.targets, mode.program, None); }
                counts.insert("spell_abilities", modes.len());
            }
            Effect::Aura(plan) => { add("spell", 0, None, Summary::default(), plan, &[], None); counts.insert("spell_abilities", 1); }
            _ => (),
        }
        for (i, ability) in rules.triggers.iter().enumerate() {
            let mut s = Summary::default(); trigger(&mut s, ability.condition);
            add("triggered", i, None, s, ability.targets, ability.program, None);
        }
        // Basics and some mana creatures use an intrinsic engine activation
        // instead of an authored entry in rules.activated.
        let intrinsic = card.mana_color().map(|_| card.activation(u16::MAX));
        let activated = rules.activated.iter().enumerate()
            .chain(intrinsic.into_iter().map(|ability| (usize::from(u16::MAX), ability)));
        for (i, ability) in activated {
            let mut s = Summary::default(); activation_cost(&mut s, ability.cost);
            if ability.sorcery { s.flag("sorcery_activation"); }
            if ability.once_per_turn || ability.exhaust { s.flag("once_per_turn"); }
            if ability.guard != ActivationGuard::Always { s.flag("conditional"); }
            let cost = json!({"colored":ability.cost.colored,"generic":ability.cost.generic,"life":ability.cost.life,
                "tap":ability.cost.tap,"sacrifice":ability.cost.sacrifice,"discard":ability.cost.discard,
                "exile":ability.cost.exile_self,"loyalty":ability.cost.loyalty});
            add("activated", i, None, s, ability.targets, ability.program, Some(cost));
        }
        drop(add);
        // These two arrays duplicate source identities in many lifted bodies.
        let groups = if rules.static_groups.is_empty() { extra.static_ability_groups } else { rules.static_groups };
        for (i, group) in groups.iter().enumerate() {
            let mut s = Summary::default(); for fragment in group.fragments { static_fragment(&mut s, *fragment); }
            all.merge(&s); abilities.push(json!({"kind":"static","slot":i,"summary":s}));
        }
        counts.insert("triggered_abilities", rules.triggers.len());
        counts.insert("activated_abilities", rules.activated.len() + usize::from(intrinsic.is_some()));
        counts.insert("static_abilities", groups.len());
        for (kind, cost) in [("kicker", extra.kicker_cost), ("offspring", extra.offspring_cost),
            ("flashback", extra.flashback_cost), ("plot", extra.plot_cost), ("warp", extra.warp_cost),
            ("disguise", extra.disguise_cost), ("sneak", extra.sneak_cost), ("harmonize", extra.harmonize_cost),
            ("additional", extra.additional_mana_alternative)] { if let Some(cost) = cost { alternatives.push(alternative(kind, cost)); } }
        if let Some(costs) = extra.tiered_costs { for cost in costs { alternatives.push(alternative("tiered", *cost)); } }
        for cost in extra.cost_alternatives { alternatives.push(alternative("hybrid_alternative", *cost)); }
        for (kind, face) in [("adventure", extra.adventure_spell), ("omen", extra.omen_spell),
            ("prepared", extra.prepared_spell), ("back_face", card.back_face())] {
            if let Some(face) = face { let (colored, generic) = face.cost(); let mut value = alternative(kind, ManaCost { colored, generic }); value["identity"] = json!(face.identity()); alternatives.push(value); }
        }
        if extra.additional_cost.is_some() || extra.flashback_additional.is_some() { all.flag("nonmana_additional_cost"); all.unknown(); }
        if extra.target_cost_reduction.is_some() || !extra.cost_reductions.is_empty() || extra.has_convoke || extra.has_improvise || extra.bargain_reduction.is_some() { all.flag("cost_reduction"); }
        if card.is_legendary() { all.flag("legendary"); } if card.is_basic_land() { all.flag("basic"); }
        if card.is_token() { all.flag("token"); }
        if card.inset_parent().is_some() || card.face_parent().is_some() { all.flag("alternate_face"); }
        match rules.entry { EntryTapped::Never => (), EntryTapped::Always => all.flag("enters_tapped"), _ => all.flag("conditional_entry") }
        let (mut power, mut toughness) = match rules.effect { Effect::Creature(p, t) => (Some(p), Some(t)), _ => (None, None) };
        if extra.dynamic_stats.is_some() { power = None; toughness = None; all.flag("power_dynamic"); all.flag("toughness_dynamic"); }
        if !matches!(rules.effect, Effect::Unsupported) && abilities.is_empty() && !extra.replacement_effects.is_empty() { all.unknown(); }
        // Unvisited extra rule families remain explicit, rather than looking vanilla.
        if !extra.replacement_effects.is_empty() || !extra.live_static_effects.is_empty()
            || extra.combat_restriction.is_some() || extra.zone_replacement.is_some()
            || !extra.entry_counter_rules.is_empty() || extra.gift.is_some() { all.unknown(); }
        let mut numbers = serde_json::Map::new();
        for (name, count) in counts { numbers.insert(name.into(), json!(count)); }
        for (name, range) in &all.amounts { numbers.insert(format!("{name}_min"), json!(range.min)); numbers.insert(format!("{name}_max"), json!(range.max)); }
        for kind in ["spell", "triggered", "activated"] {
            for effect in ["damage", "draw"] {
                let maximum = abilities.iter().filter(|a| a["kind"] == kind)
                    .filter_map(|a| a["summary"]["amounts"][effect]["max"].as_f64()).reduce(f64::max);
                numbers.insert(format!("{kind}_{effect}_max"), json!(maximum));
            }
        }
        let (colored, generic) = card.cost();
        json!({"index":card.index(),"identity":card.identity(),"name":card.name(),"implemented":card.is_supported(),
            "rules_known":!matches!(rules.effect, Effect::Unsupported),"mana_cost":card.info().mana_cost,
            "mana_value":card.fixed_mana_value(),"cost":{"colored":colored,"generic":generic},"x_symbols":card.x_symbols(),
            "kinds":card.kind_mask(),"colors":card.colors(),"keywords":card.keywords().0,"produced_colors":all.produced_colors,
            "power":power,"toughness":toughness,"starting_loyalty":card.starting_loyalty(),
            "alternative_costs":alternatives,"abilities":abilities,"flags":all.flags,"numbers":numbers,
            "known_nodes":all.known_nodes,"unknown_nodes":all.unknown_nodes})
    }).collect();
    json!({"catalogue_version":CATALOGUE_VERSION,"cards":cards})
}
