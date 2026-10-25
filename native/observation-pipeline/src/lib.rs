//! Numeric observation boundary for the pinned mono-red encoder. No game rules.
//! All handles are owned. ctypes holds locks and keeps buffers alive across calls.
mod arrays;
mod encode;
mod history;
use arrays::{Arrays, View};
use encode::{Choice, Config};
use history::History;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    cell::RefCell,
    ffi::{CStr, CString, c_char},
    panic::{AssertUnwindSafe, catch_unwind},
    sync::OnceLock,
};
type Result<T> = std::result::Result<T, String>;
static CONFIG: OnceLock<Config> = OnceLock::new();
fn config() -> &'static Config {
    CONFIG
        .get()
        .expect("Configure the native encoder before use")
}
thread_local! {static ERROR:RefCell<CString>=RefCell::new(CString::new("").unwrap());}
fn guard<T: Default>(f: impl FnOnce() -> Result<T>) -> T {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(v)) => v,
        other => {
            let error = match other {
                Ok(Err(e)) => e,
                _ => "Native pipeline panic (invalid input or schema)".into(),
            };
            ERROR.with(|s| *s.borrow_mut() = CString::new(error.replace('\0', " ")).unwrap());
            T::default()
        }
    }
}
fn json_string(v: &impl serde::Serialize) -> CString {
    CString::new(serde_json::to_vec(v).unwrap()).unwrap()
}
pub struct Observation {
    raw: Value,
    metadata: CString,
    full: OnceLock<CString>,
    fingerprint: OnceLock<[u8; 32]>,
    choices: Option<(bool, Vec<Choice>, CString)>,
}
impl Observation {
    fn new(raw: Value) -> Result<Self> {
        if !raw.is_object() {
            return Err("Expected observation object".into());
        }
        let mut meta = serde_json::Map::new();
        for k in [
            "schema",
            "viewer",
            "actor",
            "revision",
            "decision",
            "turn",
            "step",
            "active",
            "outcome",
            "players",
            "actions",
            "x_choice",
            "damage_assignment",
            "mana_lot_spend",
            "mana_payment",
        ] {
            meta.insert(k.into(), raw[k].clone());
        }
        Ok(Self {
            raw,
            metadata: json_string(&meta),
            full: OnceLock::new(),
            fingerprint: OnceLock::new(),
            choices: None,
        })
    }
    fn choices(&mut self, allow: bool) -> Result<&(bool, Vec<Choice>, CString)> {
        if self.choices.as_ref().is_none_or(|v| v.0 != allow) {
            let values = encode::choices(&self.raw, allow)?;
            let text = json_string(&values);
            self.choices = Some((allow, values, text));
        }
        Ok(self.choices.as_ref().unwrap())
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn pp_abi_version() -> u32 {
    1
}
#[unsafe(no_mangle)]
pub extern "C" fn pp_error() -> *const c_char {
    ERROR.with(|s| s.borrow().as_ptr())
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pp_configure(text: *const c_char) -> bool {
    guard(|| {
        let c: Config = serde_json::from_slice(unsafe { CStr::from_ptr(text) }.to_bytes())
            .map_err(|e| e.to_string())?;
        if c.cards.len() != 11
            || c.actions.len() != 27
            || c.semantics.len() != 11
            || c.semantics.iter().any(|s| s.len() != 25)
            || c.mana_values.len() != 11
        {
            return Err("Unsupported numeric encoder configuration".into());
        }
        if let Some(old) = CONFIG.get() {
            if old.fingerprint != c.fingerprint {
                return Err("Native encoder already configured with a different vocabulary".into());
            }
        } else {
            let _ = CONFIG.set(c);
        }
        Ok(true)
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pp_parse(text: *const c_char, response: bool) -> *mut Observation {
    guard(|| {
        let mut v: Value = serde_json::from_slice(unsafe { CStr::from_ptr(text) }.to_bytes())
            .map_err(|e| e.to_string())?;
        if response {
            v = v
                .get_mut("observation")
                .ok_or("Response has no observation")?
                .take();
        }
        Ok(Box::into_raw(Box::new(Observation::new(v)?)))
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pp_observation_free(p: *mut Observation) {
    if !p.is_null() {
        drop(unsafe { Box::from_raw(p) });
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pp_metadata(p: *const Observation) -> *const c_char {
    unsafe { &*p }.metadata.as_ptr()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pp_json(p: *const Observation) -> *const c_char {
    let o = unsafe { &*p };
    o.full.get_or_init(|| json_string(&o.raw)).as_ptr()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pp_fingerprint(p: *const Observation) -> *const u8 {
    let o = unsafe { &*p };
    o.fingerprint
        .get_or_init(|| {
            // Borrow the large object/stack trees. Only the top-level map needs
            // filtering; serde_json's maps already preserve canonical key order.
            let v: std::collections::BTreeMap<_, _> = o
                .raw
                .as_object()
                .unwrap()
                .iter()
                .filter(|(key, _)| key.as_str() != "revision")
                .collect();
            Sha256::digest(serde_json::to_vec(&v).unwrap()).into()
        })
        .as_ptr()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pp_choices(p: *mut Observation, allow: bool) -> *const c_char {
    guard(|| Ok(unsafe { &mut *p }.choices(allow)?.2.as_ptr()))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pp_encode(p: *mut Observation, allow: bool) -> *mut Arrays {
    guard(|| {
        let o = unsafe { &mut *p };
        o.choices(allow)?;
        Ok(Box::into_raw(Box::new(encode::encode(
            &o.raw,
            &o.choices.as_ref().unwrap().1,
        )?)))
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pp_arrays_free(p: *mut Arrays) {
    if !p.is_null() {
        drop(unsafe { Box::from_raw(p) });
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pp_arrays_views(p: *const Arrays, n: *mut usize) -> *const View {
    let a = unsafe { &*p };
    unsafe {
        *n = a.views.len();
    }
    a.views.as_ptr()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pp_arrays_bytes(p: *const Arrays) -> usize {
    use arrays::{Array, Data};
    use std::mem::size_of;
    let a = unsafe { &*p };
    size_of::<Arrays>()
        + a.arrays.capacity() * size_of::<Array>()
        + a.views.capacity() * size_of::<View>()
        + a.arrays
            .iter()
            .map(|v| {
                v.shape.capacity() * size_of::<usize>()
                    + match &v.data {
                        Data::Float(x) => x.capacity() * size_of::<f32>(),
                        Data::Int(x) => x.capacity() * size_of::<i32>(),
                        Data::Bool(x) => x.capacity(),
                    }
            })
            .sum::<usize>()
        + 64 * (2 * a.arrays.len() + 3) // Conservative allocator overhead.
}
#[unsafe(no_mangle)]
pub extern "C" fn pp_history_new(limit: usize) -> *mut History {
    Box::into_raw(Box::new(History::new(limit)))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pp_history_free(p: *mut History) {
    if !p.is_null() {
        drop(unsafe { Box::from_raw(p) });
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pp_history_observe(h: *mut History, o: *const Observation) -> bool {
    guard(|| {
        unsafe { &mut *h }.observe(&unsafe { &*o }.raw)?;
        Ok(true)
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pp_history_take(h: *mut History) -> *mut Arrays {
    guard(|| Ok(Box::into_raw(Box::new(unsafe { &mut *h }.take()))))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pp_collate(v: *const View, batch: usize) -> *mut Arrays {
    guard(|| {
        if v.is_null() || batch == 0 {
            return Err("Cannot collate an empty batch".into());
        }
        Ok(Box::into_raw(Box::new(unsafe {
            arrays::collate(std::slice::from_raw_parts(v, batch * 14), batch)
        }?)))
    })
}
