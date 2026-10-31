//! Extract only model-relevant fields from typed actions, without JSON.
use rustmage::{Action, Counters, Target, ObjectRef, Player, EntryChoiceValue, ManaCost,
    CastFace, LibraryDestination, PermanentType};

#[derive(Clone, Debug)]
pub enum Value {
    Null, Bool(bool), Integer(i128), Mana([u32; 6]), Counters([u64; 5]), Target(Target),
    Object(ObjectRef), Player(Player), Name(String), Entry(EntryChoiceValue), Cost(ManaCost),
}
pub trait Field { fn field(&self) -> Value; }
impl Field for bool { fn field(&self)->Value { Value::Bool(*self) } }
macro_rules! ints { ($($t:ty),*) => {$(impl Field for $t { fn field(&self)->Value { Value::Integer(*self as i128) } })*}; }
ints!(u16,u32,u64,i128);
impl<T:Field> Field for Option<T> { fn field(&self)->Value { self.as_ref().map_or(Value::Null,Field::field) } }
impl Field for [u32;6] { fn field(&self)->Value {Value::Mana(*self)} }
impl Field for Counters { fn field(&self)->Value { Value::Counters([self.plus_one,self.minus_one,self.stun,self.loyalty,self.lore]) } }
impl Field for Target { fn field(&self)->Value { Value::Target(*self) } }
impl Field for ObjectRef { fn field(&self)->Value { Value::Object(*self) } }
impl Field for Player { fn field(&self)->Value { Value::Player(*self) } }
impl Field for EntryChoiceValue { fn field(&self)->Value { Value::Entry(*self) } }
impl Field for ManaCost { fn field(&self)->Value { Value::Cost(*self) } }
macro_rules! names { ($($t:ty),*) => {$(impl Field for $t { fn field(&self)->Value { Value::Name(format!("{self:?}")) } })*}; }
names!(CastFace,LibraryDestination,PermanentType);

pub struct ActionFields { pub kind: &'static str, pub fields: Vec<(&'static str,Value)> }
impl ActionFields {
    pub fn get(&self,key:&str)->Option<&Value> { self.fields.iter().find(|(k,_)|*k==key).map(|(_,v)|v) }
}
macro_rules! fields {($name:literal $(,$key:ident)* $(,)?) => { ActionFields { kind:$name, fields:vec![$((stringify!($key),$key.field())),*] } };}

pub fn project(action:&Action)->Result<ActionFields,String> {
    Ok(match action {
        Action::Concede => fields!("Concede"),
        Action::Keep => fields!("Keep"),
        Action::Mulligan => fields!("Mulligan"),
        Action::Bottom { card } => fields!("Bottom", card),
        Action::FinishOpeningHand => fields!("FinishOpeningHand"),
        Action::Pass => fields!("Pass"),
        Action::PlayLand { card } => fields!("PlayLand", card),
        Action::TapMana { card } => fields!("TapMana", card),
        Action::AnnounceCast { card } => fields!("AnnounceCast", card),
        Action::AnnounceActivate { card, ability, counter_payment } => fields!("AnnounceActivate", card, ability, counter_payment),
        Action::PayCosts { payment } => fields!("PayCosts", payment),
        Action::Cast { card, target, payment } => fields!("Cast", card, target, payment),
        Action::BeginCast { card, payment } => fields!("BeginCast", card, payment),
        Action::Activate { card, ability, target, payment, counter_payment } => fields!("Activate", card, ability, target, payment, counter_payment),
        Action::BeginActivate { card, ability, payment, counter_payment } => fields!("BeginActivate", card, ability, payment, counter_payment),
        Action::ChooseKicker { pay } => fields!("ChooseKicker", pay),
        Action::ChooseTarget { target } => fields!("ChooseTarget", target),
        Action::FinishTargets => fields!("FinishTargets"),
        Action::Attack { card } => fields!("Attack", card),
        Action::FinishAttackers => fields!("FinishAttackers"),
        Action::Block { blocker, attacker } => fields!("Block", blocker, attacker),
        Action::FinishBlockers => fields!("FinishBlockers"),
        Action::AssignDamage { target, amount } => fields!("AssignDamage", target, amount),
        Action::Discard { card } => fields!("Discard", card),
        Action::PutTrigger { id } => fields!("PutTrigger", id),
        Action::ChooseOptional { accept } => fields!("ChooseOptional", accept),
        Action::ChooseManaLotSpend { amount } => fields!("ChooseManaLotSpend", amount),
        Action::ActivateMana { card, ability, payment, counter_payment, production } => fields!("ActivateMana", card, ability, payment, counter_payment, production),
        Action::AnnounceMana { card, ability, counter_payment, production } => fields!("AnnounceMana", card, ability, counter_payment, production),
        Action::ChooseMana { production } => fields!("ChooseMana", production),
        Action::TapGrantedMana { card, grantor, production } => fields!("TapGrantedMana", card, grantor, production),
        Action::AnnounceExile { card } => fields!("AnnounceExile", card),
        Action::AnnounceAdventure { card } => fields!("AnnounceAdventure", card),
        Action::AnnounceBack { card } => fields!("AnnounceBack", card),
        Action::AnnounceOmen { card } => fields!("AnnounceOmen", card),
        Action::AnnouncePlot { card } => fields!("AnnouncePlot", card),
        Action::PayPlot { payment } => fields!("PayPlot", payment),
        Action::AnnouncePlotted { card } => fields!("AnnouncePlotted", card),
        Action::AnnounceFlashback { card } => fields!("AnnounceFlashback", card),
        Action::AnnounceFlashSurcharge { card } => fields!("AnnounceFlashSurcharge", card),
        Action::AnnounceWarp { card } => fields!("AnnounceWarp", card),
        Action::AnnounceDisguise { card } => fields!("AnnounceDisguise", card),
        Action::AnnounceHarmonize { card } => fields!("AnnounceHarmonize", card),
        Action::AnnounceFree { card, face, flash_surcharge } => fields!("AnnounceFree", card, face, flash_surcharge),
        Action::AnnounceExileFree { card, face, flash_surcharge } => fields!("AnnounceExileFree", card, face, flash_surcharge),
        Action::AnnounceExileLife { card, face, flash_surcharge } => fields!("AnnounceExileLife", card, face, flash_surcharge),
        Action::BeginFaceUp { object, cost } => fields!("BeginFaceUp", object, cost),
        Action::PayFaceUp { payment } => fields!("PayFaceUp", payment),
        Action::AbortFaceUpPayment => fields!("AbortFaceUpPayment"),
        Action::PlayGraveyardPermanent { card, permission, permanent_type } => fields!("PlayGraveyardPermanent", card, permission, permanent_type),
        Action::AnnounceFreeGraveyardPermanent { card, face, flash_surcharge, permission, permanent_type } => fields!("AnnounceFreeGraveyardPermanent", card, face, flash_surcharge, permission, permanent_type),
        Action::AnnounceDisguiseGraveyard { card, permission, permanent_type } => fields!("AnnounceDisguiseGraveyard", card, permission, permanent_type),
        Action::AnnounceFreeFromPermission { card, face, flash_surcharge, permission } => fields!("AnnounceFreeFromPermission", card, face, flash_surcharge, permission),
        Action::AnnounceGrantedFlashback { card, flash_surcharge } => fields!("AnnounceGrantedFlashback", card, flash_surcharge),
        Action::ChooseEntryValue { value } => fields!("ChooseEntryValue", value),
        Action::ChooseEntryLife { pay } => fields!("ChooseEntryLife", pay),
        Action::ChooseEntryTimestamp { object } => fields!("ChooseEntryTimestamp", object),
        Action::ChooseEntryReplacement { source, intrinsic } => fields!("ChooseEntryReplacement", source, intrinsic),
        Action::ChooseManaRollback { keep } => fields!("ChooseManaRollback", keep),
        Action::ChooseOffspring { pay } => fields!("ChooseOffspring", pay),
        Action::ChooseCasualty { pay } => fields!("ChooseCasualty", pay),
        Action::ChooseBargain { sacrifice } => fields!("ChooseBargain", sacrifice),
        Action::ChooseGift { beneficiary } => fields!("ChooseGift", beneficiary),
        Action::ChooseBlight { pay } => fields!("ChooseBlight", pay),
        Action::ChooseTeamwork { pay } => fields!("ChooseTeamwork", pay),
        Action::ChooseAdditionalLife => fields!("ChooseAdditionalLife"),
        Action::ChooseAdditionalCost { object } => fields!("ChooseAdditionalCost", object),
        Action::ChooseCastCost { clause } => fields!("ChooseCastCost", clause),
        Action::ChooseSpellCost { cost } => fields!("ChooseSpellCost", cost),
        Action::ChooseActivationCost { cost } => fields!("ChooseActivationCost", cost),
        Action::ChooseMode { mode } => fields!("ChooseMode", mode),
        Action::ChooseX { value } => fields!("ChooseX", value),
        Action::ChooseSourceCounterAmount { amount } => fields!("ChooseSourceCounterAmount", amount),
        Action::AllocateTargetAmount { amount } => fields!("AllocateTargetAmount", amount),
        Action::ChooseSearch { card } => fields!("ChooseSearch", card),
        Action::ChooseSearchBattlefield { card } => fields!("ChooseSearchBattlefield", card),
        Action::FinishSearch => fields!("FinishSearch"),
        Action::ChooseSearchZones { library, graveyard } => fields!("ChooseSearchZones", library, graveyard),
        Action::ChooseHandLibrarySearchZones { hand, library } => fields!("ChooseHandLibrarySearchZones", hand, library),
        Action::SelectLibrary { card } => fields!("SelectLibrary", card),
        Action::FinishLibrarySelection => fields!("FinishLibrarySelection"),
        Action::OrderLibraryBottom { card } => fields!("OrderLibraryBottom", card),
        Action::ArrangeLibrary { card, destination } => fields!("ArrangeLibrary", card, destination),
        Action::ChooseLibraryDestination { object, destination } => fields!("ChooseLibraryDestination", object, destination),
        Action::ChooseHandCard { object } => fields!("ChooseHandCard", object),
        Action::ChooseRevealedDiscard { object } => fields!("ChooseRevealedDiscard", object),
        Action::ChooseExplore { object, graveyard } => fields!("ChooseExplore", object, graveyard),
        Action::ChooseManifest { object } => fields!("ChooseManifest", object),
        Action::Sacrifice { object } => fields!("Sacrifice", object),
        Action::KeepLegend { object } => fields!("KeepLegend", object),
        Action::KeepRole { object } => fields!("KeepRole", object),
        Action::ChooseAuraAttachment { object } => fields!("ChooseAuraAttachment", object),
        Action::ChooseLinkedExile { object } => fields!("ChooseLinkedExile", object),
        Action::BeginOnBattlefield { object } => fields!("BeginOnBattlefield", object),
        Action::BeginOrderedCosts => fields!("BeginOrderedCosts"),
        Action::PayResolutionMana { payment } => fields!("PayResolutionMana", payment),
        Action::DeclineResolutionMana => fields!("DeclineResolutionMana"),
        Action::PayResolutionLife => fields!("PayResolutionLife"),
        Action::DeclineResolutionLife => fields!("DeclineResolutionLife"),
        Action::PayAttackCost { payment } => fields!("PayAttackCost", payment),
        Action::AbortAttackPayment => fields!("AbortAttackPayment"),
        Action::CancelAttackDeclaration => fields!("CancelAttackDeclaration"),
        Action::DeclineEffectPayment => fields!("DeclineEffectPayment"),
        Action::CastExiledDuringResolution { object, face } => fields!("CastExiledDuringResolution", object, face),
        Action::FinishResolutionCasts => fields!("FinishResolutionCasts"),
        Action::FinishGraveyardSymbolCost => fields!("FinishGraveyardSymbolCost"),
        // The frozen 250K encoder maps Harmonize decisions onto the
        // existing Bargain feature columns, while preserving actual actions.
        Action::ChooseHarmonizer { object } => ActionFields {
            kind: "ChooseBargain", fields: vec![("sacrifice", object.field())] },
        other => return Err(format!("unsupported Standard action: {other:?}")),
    })
}
