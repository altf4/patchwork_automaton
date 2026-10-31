//! Experimental in-process RustMage bridge. Game observations stay typed and
//! player-redacted. JSON is used only for one-time configuration and the
//! explicitly diagnostic reference-view export, never for the observation ABI.
mod entity;
mod arrays;
mod actions;
mod encode;
mod history;
mod choices;
use entity::{Catalogue, ENTITY_WIDTH};
use arrays::{Arrays, View as ArrayView};
use rustmage::{Config, Game, Observation, Outcome, Player};
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::{OnceLock, atomic::{AtomicU64, Ordering}};
use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;

thread_local! { static ERROR: RefCell<CString> = RefCell::new(CString::new("").unwrap()); }
fn error(msg: impl ToString) {
    ERROR.with(|e| *e.borrow_mut() = CString::new(msg.to_string().replace('\0', " ")).unwrap());
}
type Result<T> = std::result::Result<T, String>;
fn guard<T: Default>(f: impl FnOnce() -> Result<T>) -> T {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(value)) => value,
        Ok(Err(e)) => { error(e); T::default() }
        Err(_) => { error("direct bridge panic; discard affected game"); T::default() }
    }
}
fn player(v: u32) -> Result<Player> {
    match v {0 => Ok(Player::P0), 1 => Ok(Player::P1), _ => Err("invalid player".into())}
}

#[derive(Deserialize)]
struct EncoderConfig { fingerprint: String, mana_values: HashMap<String, Option<f64>>, traits: HashMap<String, Vec<f32>>, identities: HashMap<String, String>, unknown_traits: Vec<f32>, actions: Vec<String> }
struct Encoder { fingerprint: String, catalogue: Catalogue, actions: Vec<String> }
static ENCODER: OnceLock<Encoder> = OnceLock::new();
pub struct EntityRows { values: Vec<f32>, rows: usize }
static NEXT_SESSION: AtomicU64 = AtomicU64::new(1);
pub struct Session { game: Game, id: u64, histories: [history::History; 2] }
pub struct View { observation: Observation, session_id: u64, choices: Option<Vec<choices::Choice>> }
#[repr(C)]
#[derive(Default)]
pub struct Metadata {
    revision: u64,
    turn: u32,
    viewer: u32,
    actor: i32,
    active: u32,
    outcome: i32,
    actions: usize,
    objects: usize,
    life0: i64,
    life1: i64,
}

/// Immutable catalogue data is configured once before numerical encoding.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ds_configure(data: *const c_char) -> bool {
    guard(|| {
        if data.is_null() { return Err("null encoder configuration".into()); }
        let c: EncoderConfig = serde_json::from_slice(unsafe { CStr::from_ptr(data) }.to_bytes())
            .map_err(|e| e.to_string())?;
        if c.fingerprint.is_empty() { return Err("missing schema fingerprint".into()); }
        if c.unknown_traits.len() != 252 || c.traits.values().any(|v| v.len() != 252) || c.actions.len() != 106 {
            return Err("unsupported Standard feature schema".into());
        }
        let new = Encoder { fingerprint: c.fingerprint, catalogue: Catalogue {
            mana_values: c.mana_values, traits: c.traits, identities: c.identities, unknown_traits: c.unknown_traits }, actions: c.actions };
        if ENCODER.get().is_none() { let _ = ENCODER.set(new); }
        let old = ENCODER.get().ok_or("encoder configuration failed")?;
        // Check the full input when called again; the caller supplies the schema
        // fingerprint and its vocabulary, which must describe the same encoder.
        let c: EncoderConfig = serde_json::from_slice(unsafe { CStr::from_ptr(data) }.to_bytes()).map_err(|e| e.to_string())?;
        if old.fingerprint != c.fingerprint || old.actions != c.actions || old.catalogue.mana_values != c.mana_values
            || old.catalogue.traits != c.traits || old.catalogue.identities != c.identities || old.catalogue.unknown_traits != c.unknown_traits {
            return Err("encoder already has a different schema or vocabulary".into());
        }
        Ok(true)
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn ds_version() -> u32 { 1 }
#[unsafe(no_mangle)]
pub extern "C" fn ds_error() -> *const c_char { ERROR.with(|e| e.borrow().as_ptr()) }

/// Configuration is loaded once per game; subsequent operations use typed data.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ds_new(config: *const c_char) -> *mut Session {
    guard(|| {
        if config.is_null() { return Err("null game configuration".into()); }
        let config: Config = serde_json::from_slice(unsafe { CStr::from_ptr(config) }.to_bytes())
            .map_err(|e| e.to_string())?;
        let game = Game::new(config).map_err(|e| e.to_string())?;
        let id = NEXT_SESSION.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_add(1))
            .map_err(|_| "session identity exhausted")?;
        Ok(Box::into_raw(Box::new(Session { game, id, histories: [history::History::default(), history::History::default()] })))
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ds_free(session: *mut Session) {
    if !session.is_null() { drop(unsafe { Box::from_raw(session) }); }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ds_observe(session: *const Session, viewer: u32) -> *mut View {
    guard(|| {
        let session = unsafe { session.as_ref() }.ok_or("null game")?;
        Ok(Box::into_raw(Box::new(View { observation: session.game.observe(player(viewer)?),
            session_id: session.id, choices: None })))
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ds_view_free(view: *mut View) {
    if !view.is_null() { drop(unsafe { Box::from_raw(view) }); }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ds_metadata(view: *const View, output: *mut Metadata) -> bool {
    guard(|| {
        let v = &unsafe { view.as_ref() }.ok_or("null observation")?.observation;
        let out = unsafe { output.as_mut() }.ok_or("null output")?;
        *out = Metadata { revision: v.revision, turn: v.turn, viewer: v.viewer.index() as u32,
            actor: v.actor.map_or(-1, |p| p.index() as i32), active: v.active.index() as u32,
            outcome: match v.outcome { None => -1, Some(Outcome::Draw) => 2,
                Some(Outcome::Winner(p)) => p.index() as i32 },
            actions: v.actions.len(), objects: v.objects.len(),
            life0: i64::try_from(v.players[0].life).map_err(|_| "player life exceeds bridge range")?,
            life1: i64::try_from(v.players[1].life).map_err(|_| "player life exceeds bridge range")? };
        Ok(true)
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ds_step(session: *mut Session, actor: u32, revision: u64, index: usize) -> bool {
    guard(|| {
        let s = unsafe { session.as_mut() }.ok_or("null game")?;
        s.game.apply_index(player(actor)?, revision, index).map_err(|e| e.to_string())?;
        Ok(true)
    })
}
/// Enumerate all supported legal choices without serializing the observation.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ds_prepare(view: *mut View, allow_concede: bool, count: *mut usize) -> bool {
    guard(|| {
        let view = unsafe { view.as_mut() }.ok_or("null observation")?;
        let choices = choices::choices(&view.observation, allow_concede)?;
        *unsafe { count.as_mut() }.ok_or("null count")? = choices.len();
        view.choices = Some(choices);
        Ok(true)
    })
}
/// Apply the chosen typed action, enforcing game, actor and revision identity.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ds_apply_choice(session: *mut Session, view: *const View, index: usize) -> bool {
    guard(|| {
        let session = unsafe { session.as_mut() }.ok_or("null game")?;
        let view = unsafe { view.as_ref() }.ok_or("null observation")?;
        if view.session_id != session.id { return Err("observation belongs to another game".into()); }
        if view.observation.revision != session.game.revision() { return Err("stale observation".into()); }
        let actor = view.observation.actor.ok_or("no decision actor")?;
        if actor != view.observation.viewer { return Err("viewer is not the decision actor".into()); }
        let choice = view.choices.as_ref().ok_or("choices not prepared")?.get(index).ok_or("choice out of bounds")?;
        if let Some(finite) = choice.finite_index {
            session.game.apply_index(actor, view.observation.revision, finite).map_err(|e| e.to_string())?;
        } else { session.game.apply(actor, choice.action.clone()).map_err(|e| e.to_string())?; }
        Ok(true)
    })
}
/// Apply an action chosen by an existing Python scripted policy. Neural
/// policies use ds_apply_choice and never serialize actions. This compatibility
/// entry point retains full numeric-domain support for scripted policies.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ds_apply_script(session: *mut Session, view: *const View, data: *const c_char) -> bool {
    guard(|| {
        let session = unsafe { session.as_mut() }.ok_or("null game")?;
        let view = unsafe { view.as_ref() }.ok_or("null observation")?;
        if view.session_id != session.id || view.observation.revision != session.game.revision() {
            return Err("foreign or stale observation".into());
        }
        let actor = view.observation.actor.ok_or("no decision actor")?;
        if actor != view.observation.viewer { return Err("viewer is not the decision actor".into()); }
        if data.is_null() { return Err("null scripted action".into()); }
        let action: rustmage::Action = serde_json::from_slice(unsafe { CStr::from_ptr(data) }.to_bytes())
            .map_err(|e| e.to_string())?;
        session.game.apply(actor, action).map_err(|e| e.to_string())?;
        Ok(true)
    })
}
/// Diagnostic-only action export for parity testing.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ds_choice_debug_json(view: *const View, index: usize) -> *mut c_char {
    guard(|| {
        let view = unsafe { view.as_ref() }.ok_or("null observation")?;
        let choice = view.choices.as_ref().ok_or("choices not prepared")?.get(index).ok_or("choice out of bounds")?;
        let text = serde_json::to_string(&choice.action).map_err(|e| e.to_string())?;
        Ok(CString::new(text).map_err(|e| e.to_string())?.into_raw())
    })
}

/// Update only the observation viewer's private history for this game.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ds_history_observe(session: *mut Session, view: *const View) -> bool {
    guard(|| {
        let session = unsafe { session.as_mut() }.ok_or("null game")?;
        let view = unsafe { view.as_ref() }.ok_or("null observation")?;
        if view.session_id != session.id || view.observation.revision != session.game.revision() {
            return Err("foreign or stale observation".into());
        }
        let encoder = ENCODER.get().ok_or("encoder not configured")?;
        session.histories[view.observation.viewer.index()].observe(&view.observation, encoder)?;
        Ok(true)
    })
}
/// Consume the actor's delta history only on a non-forced model decision.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ds_history_take(session: *mut Session, view: *const View) -> *mut Arrays {
    guard(|| {
        let session = unsafe { session.as_mut() }.ok_or("null game")?;
        let view = unsafe { view.as_ref() }.ok_or("null observation")?;
        if view.session_id != session.id || view.observation.revision != session.game.revision()
            || view.observation.actor != Some(view.observation.viewer) { return Err("foreign, stale or non-actor observation".into()); }
        Ok(Box::into_raw(Box::new(session.histories[view.observation.viewer.index()].take())))
    })
}
/// Action label for diagnostics and rollout summaries; action values remain Rust-owned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ds_choice_kind(view: *const View, index: usize) -> *mut c_char {
    guard(|| {
        let view = unsafe { view.as_ref() }.ok_or("null observation")?;
        let choice = view.choices.as_ref().ok_or("choices not prepared")?.get(index).ok_or("choice out of bounds")?;
        let kind = if matches!(choice.action, rustmage::Action::ChooseHarmonizer { .. }) {"ChooseHarmonizer"}
                   else {actions::project(&choice.action)?.kind};
        Ok(CString::new(kind).map_err(|e|e.to_string())?.into_raw())
    })
}

/// Directly build Standard numerical entity rows from the typed private view.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ds_entity_rows(view: *const View) -> *mut EntityRows {
    guard(|| {
        let o = &unsafe { view.as_ref() }.ok_or("null observation")?.observation;
        let encoder = ENCODER.get().ok_or("encoder not configured")?;
        let count = o.objects.len() + 2;
        let mut values = Vec::with_capacity(count * ENTITY_WIDTH);
        values.extend(entity::player_row(&o.players[o.viewer.index()], 0));
        values.extend(entity::player_row(&o.players[o.viewer.other().index()], 1));
        for object in &o.objects {
            values.extend(entity::object_row(object, (object.controller != o.viewer) as usize, &encoder.catalogue));
        }
        Ok(Box::into_raw(Box::new(EntityRows { values, rows: count })))
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ds_entity_data(rows: *const EntityRows, count: *mut usize) -> *const f32 {
    guard(|| {
        let rows = unsafe { rows.as_ref() }.ok_or("null entity rows")?;
        *unsafe { count.as_mut() }.ok_or("null count")? = rows.rows;
        Ok(rows.values.as_ptr())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ds_entity_free(rows: *mut EntityRows) {
    if !rows.is_null() { drop(unsafe { Box::from_raw(rows) }); }
}

/// Encode the complete Standard v5 model input directly from the typed view.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ds_encode(view: *const View) -> *mut Arrays {
    guard(|| {
        let view = unsafe { view.as_ref() }.ok_or("null observation")?;
        let encoder = ENCODER.get().ok_or("encoder not configured")?;
        let choices = view.choices.as_ref().ok_or("choices not prepared")?;
        Ok(Box::into_raw(Box::new(encode::encode(&view.observation, choices, encoder)?)))
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ds_arrays_views(arrays: *const Arrays, count: *mut usize) -> *const ArrayView {
    guard(|| {
        let arrays = unsafe { arrays.as_ref() }.ok_or("null arrays")?;
        *unsafe { count.as_mut() }.ok_or("null count")? = arrays.views.len();
        Ok(arrays.views.as_ptr())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ds_arrays_free(arrays: *mut Arrays) {
    if !arrays.is_null() { drop(unsafe { Box::from_raw(arrays) }); }
}

/// Diagnostic only. Production feature encoding must consume the typed view.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ds_debug_json(view: *const View) -> *mut c_char {
    guard(|| {
        let v = &unsafe { view.as_ref() }.ok_or("null observation")?.observation;
        let text = serde_json::to_string(v).map_err(|e| e.to_string())?;
        Ok(CString::new(text).map_err(|e| e.to_string())?.into_raw())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ds_string_free(value: *mut c_char) {
    if value != ptr::null_mut() { drop(unsafe { CString::from_raw(value) }); }
}
