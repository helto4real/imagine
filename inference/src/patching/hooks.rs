//! Hook keyframe scheduling and clone identities from the locked hooks.py.
use super::options::{self, Options};
use crate::{execution_value::Value as V, native::Error};
use std::{cell::RefCell, rc::Rc};
use tch::Tensor;

#[derive(Clone)]
pub struct Keyframe {
    pub strength: f64,
    pub start_percent: f64,
    pub start_t: V,
    pub guarantee_steps: i64,
}
impl Keyframe {
    pub fn new(strength: f64, start_percent: f64, guarantee_steps: i64) -> Self {
        Self {
            strength,
            start_percent,
            start_t: V::Float(999999999.9),
            guarantee_steps,
        }
    }
    pub fn effective_guarantee(&self, max_sigma: &Tensor) -> Result<i64, Error> {
        let before = match &self.start_t {
            V::Float(value) => max_sigma.f_lt(*value)?.f_is_nonzero()?,
            V::Tensor(_) => self
                .start_t
                .as_tensor()?
                .f_gt_tensor(max_sigma)?
                .f_is_nonzero()?,
            _ => return Err(Error::Native("Invalid keyframe start sigma".into())),
        };
        Ok(if before { 0 } else { self.guarantee_steps })
    }
    fn reached(&self, time: f64) -> Result<bool, Error> {
        match &self.start_t {
            V::Float(value) => Ok(*value >= time),
            V::Tensor(_) => Ok(self.start_t.as_tensor()?.f_ge(time)?.f_is_nonzero()?),
            _ => Err(Error::Native("Invalid keyframe start sigma".into())),
        }
    }
}
type Frame = Rc<RefCell<Keyframe>>;
pub struct Keyframes {
    pub keyframes: Vec<Frame>,
    pub current: Option<Frame>,
    pub used_steps: i64,
    pub current_index: usize,
    pub current_strength: Option<f64>,
    pub last_t: f64,
    /// reset writes curr_t in the source, leaving _curr_t unchanged.
    pub reset_curr_t: Option<f64>,
}
impl Default for Keyframes {
    fn default() -> Self {
        Self {
            keyframes: Vec::new(),
            current: None,
            used_steps: 0,
            current_index: 0,
            current_strength: None,
            last_t: -1.,
            reset_curr_t: None,
        }
    }
}
impl Keyframes {
    pub fn strength(&self) -> f64 {
        self.current
            .as_ref()
            .map_or(1., |frame| frame.borrow().strength)
    }
    fn set_first(&mut self) {
        self.current = self.keyframes.first().cloned();
    }
    pub fn add(&mut self, keyframe: Frame) {
        self.keyframes.push(keyframe);
        // Python groups identical start_percent values before its stable sort.
        let mut groups: Vec<(f64, Vec<Frame>)> = Vec::new();
        for keyframe in self.keyframes.drain(..) {
            let start = keyframe.borrow().start_percent;
            if let Some((_, group)) = groups.iter_mut().find(|(value, _)| *value == start) {
                group.push(keyframe);
            } else {
                groups.push((start, vec![keyframe]));
            }
        }
        groups.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        self.keyframes = groups.into_iter().flat_map(|(_, group)| group).collect();
        self.set_first();
    }
    pub fn reset(&mut self) {
        self.current = None;
        self.used_steps = 0;
        self.current_index = 0;
        self.current_strength = None;
        self.reset_curr_t = Some(-1.);
        self.set_first();
    }
    pub fn has_guarantee_steps(&self) -> bool {
        self.keyframes
            .iter()
            .any(|f| f.borrow().guarantee_steps > 0)
    }
    pub fn initialize_timesteps(
        &mut self,
        percent_to_sigma: &mut dyn FnMut(f64) -> Result<V, Error>,
    ) -> Result<(), Error> {
        for frame in &self.keyframes {
            let start = frame.borrow().start_percent;
            frame.borrow_mut().start_t = percent_to_sigma(start)?;
        }
        Ok(())
    }
    pub fn prepare(&mut self, time: f64, max_sigma: &Tensor) -> Result<bool, Error> {
        if self.keyframes.is_empty() || time == self.last_t {
            return Ok(false);
        }
        let old_index = self.current_index;
        let old_strength = self.current_strength;
        if self.used_steps
            >= self
                .current
                .as_ref()
                .unwrap()
                .borrow()
                .effective_guarantee(max_sigma)?
        {
            for index in self.current_index + 1..self.keyframes.len() {
                let frame = &self.keyframes[index];
                if frame.borrow().reached(time)? {
                    self.current_index = index;
                    self.current_strength = Some(frame.borrow().strength);
                    self.current = Some(frame.clone());
                    self.used_steps = 0;
                    if frame.borrow().effective_guarantee(max_sigma)? > 0 {
                        break;
                    }
                } else {
                    break;
                }
            }
        }
        self.used_steps += 1;
        self.last_t = time;
        Ok(old_index != self.current_index && old_strength != self.current_strength)
    }
    pub fn clone_source(&self) -> Self {
        let mut cloned = Self {
            keyframes: self
                .keyframes
                .iter()
                .map(|frame| Rc::new(RefCell::new(frame.borrow().clone())))
                .collect(),
            ..Self::default()
        };
        cloned.set_first();
        cloned
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kind {
    Base,
    Weight,
    ObjectPatch,
    AdditionalModels,
    TransformerOptions,
    Injections,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Scope {
    AllConditioning,
    HookedOnly,
}
pub struct Hook {
    /// Concrete Python class, used by Hook.__eq__, independently of hook_type.
    pub kind: Kind,
    pub hook_type: Kind,
    pub reference: Rc<()>,
    pub id: Option<String>,
    pub keyframes: Rc<RefCell<Keyframes>>,
    pub scope: Scope,
    pub values: Options,
    pub weights_clip: Options,
    pub need_weight_init: bool,
    pub strength_model: f64,
    pub strength_clip: f64,
    pub skip_adding: bool,
    pub should_register: RegisterPredicate,
}
impl Hook {
    pub fn new(kind: Kind) -> Self {
        Self {
            kind,
            hook_type: kind,
            reference: Rc::new(()),
            id: None,
            keyframes: Rc::new(RefCell::new(Keyframes::default())),
            scope: if kind == Kind::Weight {
                Scope::HookedOnly
            } else {
                Scope::AllConditioning
            },
            values: Options::from_value(crate::execution_value::Value::None),
            weights_clip: Options::from_value(crate::execution_value::Value::None),
            need_weight_init: true,
            strength_model: 1.,
            strength_clip: 1.,
            skip_adding: false,
            should_register: Rc::new(|_, _, _, _, _| Ok(true)),
        }
    }
    pub fn same_hook(&self, other: &Self) -> bool {
        self.kind == other.kind && Rc::ptr_eq(&self.reference, &other.reference)
    }
    pub fn clone_source(&self) -> Self {
        let mut values = self.values.clone();
        // The source copies only truthy AdditionalModels/Injections lists.
        if matches!(self.kind, Kind::AdditionalModels | Kind::Injections)
            && let Options::List(list) = &self.values
            && !list.borrow().is_empty()
        {
            values = Options::List(Rc::new(RefCell::new(list.borrow().clone())));
        }
        Self {
            kind: self.kind,
            hook_type: self.hook_type,
            reference: self.reference.clone(),
            id: self.id.clone(),
            keyframes: self.keyframes.clone(),
            scope: self.scope,
            values,
            weights_clip: self.weights_clip.clone(),
            need_weight_init: self.need_weight_init,
            strength_model: self.strength_model,
            strength_clip: self.strength_clip,
            skip_adding: self.skip_adding,
            should_register: self.should_register.clone(),
        }
    }
    pub fn initialize_timesteps(
        &self,
        percent_to_sigma: &mut dyn FnMut(f64) -> Result<V, Error>,
    ) -> Result<(), Error> {
        self.keyframes.borrow_mut().reset();
        self.keyframes
            .borrow_mut()
            .initialize_timesteps(percent_to_sigma)
    }
    pub fn strength(&self, clip: bool) -> f64 {
        (if clip {
            self.strength_clip
        } else {
            self.strength_model
        }) * self.keyframes.borrow().strength()
    }
    pub fn apply_options(&self, transformer_options: &Options) -> Result<(), Error> {
        if self.kind == Kind::TransformerOptions && !self.skip_adding {
            options::merge_nested_dicts(transformer_options, &self.values, false)?;
        }
        Ok(())
    }
}
pub type HookHandle = Rc<RefCell<Hook>>;
pub type HookList = Rc<RefCell<Vec<HookHandle>>>;
#[derive(Default)]
pub struct HookGroup {
    pub hooks: Vec<HookHandle>,
    /// The source caches type lists when adding a hook. Later hook_type
    /// mutation does not rebuild this index; callers receive the actual list.
    type_lists: Vec<(Kind, HookList)>,
}
impl HookGroup {
    pub fn add(&mut self, hook: Rc<RefCell<Hook>>) {
        if !self.contains(&hook.borrow()) {
            self.hooks.push(hook.clone());
            let kind = hook.borrow().hook_type;
            if let Some((_, list)) = self.type_lists.iter().find(|(k, _)| *k == kind) {
                list.borrow_mut().push(hook);
            } else {
                self.type_lists
                    .push((kind, Rc::new(RefCell::new(vec![hook]))));
            }
        }
    }
    pub fn contains(&self, hook: &Hook) -> bool {
        self.hooks.iter().any(|h| h.borrow().same_hook(hook))
    }
    pub fn remove(&mut self, hook: &Hook) -> Result<(), Error> {
        if let Some(index) = self.hooks.iter().position(|h| h.borrow().same_hook(hook)) {
            self.hooks.remove(index);
            let list = self
                .type_lists
                .iter()
                .find(|(k, _)| *k == hook.hook_type)
                .ok_or_else(|| Error::ReferenceException {
                    class: "KeyError",
                    message: hook.hook_type.python_repr().into(),
                })?;
            let mut values = list.1.borrow_mut();
            let index = values
                .iter()
                .position(|h| h.borrow().same_hook(hook))
                .ok_or_else(|| Error::ReferenceException {
                    class: "ValueError",
                    message: "list.remove(x): x not in list".into(),
                })?;
            values.remove(index);
        }
        Ok(())
    }
    pub fn get_type(&self, kind: Kind) -> HookList {
        self.type_lists
            .iter()
            .find(|(k, _)| *k == kind)
            .map_or_else(|| Rc::new(RefCell::new(Vec::new())), |(_, v)| v.clone())
    }
    pub fn by_kind(&self, kind: Kind) -> Vec<HookHandle> {
        self.get_type(kind).borrow().clone()
    }
    pub fn is_subset(&self, other: &Self) -> bool {
        self.hooks.iter().all(|h| other.contains(&h.borrow()))
    }
    pub fn clone_source(&self) -> Self {
        let mut out = Self::default();
        for hook in &self.hooks {
            out.add(Rc::new(RefCell::new(hook.borrow().clone_source())));
        }
        out
    }
    pub fn common(&self, other: &Self) -> Self {
        let mut out = Self::default();
        for hook in &self.hooks {
            if other.contains(&hook.borrow()) {
                out.add(Rc::new(RefCell::new(hook.borrow().clone_source())));
            }
        }
        out
    }
    pub fn combine(&self, other: Option<&Self>) -> Self {
        let mut out = self.clone_source();
        if let Some(other) = other {
            for hook in &other.hooks {
                out.add(Rc::new(RefCell::new(hook.borrow().clone_source())));
            }
        }
        out
    }
    pub fn set_keyframes(&self, frames: Option<&Keyframes>) {
        let frames = Rc::new(RefCell::new(
            frames.map_or_else(Keyframes::default, Keyframes::clone_source),
        ));
        for hook in &self.hooks {
            hook.borrow_mut().keyframes = frames.clone();
        }
    }
}

impl Kind {
    fn python_repr(self) -> &'static str {
        match self {
            Self::Base => "None",
            Self::Weight => "<EnumHookType.Weight: 'weight'>",
            Self::ObjectPatch => "<EnumHookType.ObjectPatch: 'object_patch'>",
            Self::AdditionalModels => "<EnumHookType.AdditionalModels: 'add_models'>",
            Self::TransformerOptions => "<EnumHookType.TransformerOptions: 'transformer_options'>",
            Self::Injections => "<EnumHookType.Injections: 'add_injections'>",
        }
    }
}
pub struct ClipRange {
    pub range: (f64, f64),
    pub hooks: Vec<(HookHandle, Option<Frame>)>,
}
fn isclose(a: f64, b: f64) -> bool {
    a == b || (a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-9 * a.abs().max(b.abs()))
}
impl HookGroup {
    pub fn reset(&self) {
        for hook in &self.hooks {
            hook.borrow().keyframes.borrow_mut().reset();
        }
    }
    pub fn clip_schedule(&self) -> Vec<ClipRange> {
        let mut schedules = Vec::new();
        for hook in self.by_kind(Kind::Weight) {
            let frames = hook.borrow().keyframes.clone();
            let frames = frames.borrow();
            let mut schedule = Vec::new();
            if frames.keyframes.is_empty() {
                schedule.push(((0., 1.), None));
            } else {
                let mut prev = frames.keyframes[0].clone();
                for frame in &frames.keyframes {
                    let p = prev.borrow();
                    let f = frame.borrow();
                    let split =
                        f.start_percent > p.start_percent && !isclose(f.strength, p.strength);
                    let same = f.start_percent == p.start_percent;
                    if split {
                        schedule.push(((p.start_percent, f.start_percent), Some(prev.clone())));
                    }
                    drop(p);
                    drop(f);
                    if split || same {
                        prev = frame.clone();
                    }
                }
                let start = prev.borrow().start_percent;
                if !isclose(start, 1.) {
                    schedule.push(((start, 1.), Some(prev)));
                }
            }
            schedules.push((hook, schedule));
        }
        let mut boundaries = vec![0.];
        for (_, schedule) in &schedules {
            for ((a, b), _) in schedule {
                for value in [*a, *b] {
                    if !boundaries.contains(&value) {
                        boundaries.push(value);
                    }
                }
            }
        }
        boundaries.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        boundaries
            .windows(2)
            .map(|w| ClipRange {
                range: (w[0], w[1]),
                hooks: schedules
                    .iter()
                    .map(|(hook, schedule)| {
                        let frame = schedule
                            .iter()
                            .find(|((a, b), _)| *a < w[1] && *b > w[0])
                            .and_then(|(_, frame)| frame.clone());
                        (hook.clone(), frame)
                    })
                    .collect(),
            })
            .collect()
    }
    /// Exactly one non-None group is returned by identity; two or more are cloned.
    pub fn combine_all(
        groups: &[Option<Rc<RefCell<Self>>>],
        require_count: i64,
    ) -> Result<Option<Rc<RefCell<Self>>>, Error> {
        let actual = groups.iter().flatten().collect::<Vec<_>>();
        if (actual.len() as i64) < require_count {
            return Err(Error::ReferenceException {
                class: "Exception",
                message: format!(
                    "Need at least {require_count} hooks to combine, but only had {}.",
                    actual.len()
                ),
            });
        }
        if actual.is_empty() {
            return Ok(None);
        }
        if actual.len() == 1 {
            return Ok(Some(actual[0].clone()));
        }
        let mut combined = actual[0].borrow().clone_source();
        for group in actual.iter().skip(1) {
            combined = combined.combine(Some(&group.borrow()));
        }
        Ok(Some(Rc::new(RefCell::new(combined))))
    }
}

/// Source model dependencies are explicit Rust bindings; the hook implementation
/// never imports Python or changes a failed LoRA conversion into a fallback.
#[derive(Clone, Copy, Eq, PartialEq)]
pub enum WeightTarget {
    Model,
    Clip,
}
pub trait Registration {
    fn load_weight_patches(&mut self, weights: &Options, clip: bool) -> Result<Options, Error>;
    fn add_hook_patches(
        &mut self,
        hook: &HookHandle,
        patches: &Options,
        strength: f64,
    ) -> Result<(), Error>;
}
pub type RegisterPredicate = Rc<
    dyn Fn(
        &HookHandle,
        &mut dyn Registration,
        &Options,
        &Options,
        &mut HookGroup,
    ) -> Result<bool, Error>,
>;
pub fn register_hook(
    hook: &HookHandle,
    model: &mut dyn Registration,
    model_options: &Options,
    target: &Options,
    registered: &mut HookGroup,
) -> Result<bool, Error> {
    let kind = hook.borrow().kind;
    let message = match kind {
        Kind::Base => Some("add_hook_patches should be defined for Hook subclasses"),
        Kind::ObjectPatch => Some("ObjectPatchHook is not supported yet in ComfyUI."),
        Kind::Injections => Some("InjectionsHook is not supported yet in ComfyUI."),
        _ => None,
    };
    if let Some(message) = message {
        return Err(Error::ReferenceException {
            class: "NotImplementedError",
            message: message.into(),
        });
    }
    let should = hook.borrow().should_register.clone();
    if !should(hook, model, model_options, target, registered)? {
        return Ok(false);
    }
    match kind {
        Kind::Weight => {
            let clip = target
                .get("target")?
                .is_some_and(|t| matches!(t, Options::Object(ref value) if value.downcast_ref::<WeightTarget>() == Some(&WeightTarget::Clip)));
            let (initialize, strength, weights) = {
                let h = hook.borrow();
                (
                    h.need_weight_init,
                    if clip {
                        h.strength_clip
                    } else {
                        h.strength_model
                    },
                    if h.need_weight_init || !clip {
                        h.values.clone()
                    } else {
                        h.weights_clip.clone()
                    },
                )
            };
            let patches = if initialize {
                model.load_weight_patches(&weights, clip)?
            } else {
                weights
            };
            model.add_hook_patches(hook, &patches, strength)?;
            registered.add(hook.clone());
        }
        Kind::TransformerOptions => {
            let (scope, values) = {
                let mut h = hook.borrow_mut();
                h.skip_adding = false;
                (h.scope, h.values.clone())
            };
            let mut entries = vec![("to_load_options".into(), values.clone())];
            if scope == Scope::AllConditioning {
                entries.insert(0, ("transformer_options".into(), values));
                hook.borrow_mut().skip_adding = true;
            }
            registered.add(hook.clone());
            options::merge_nested_dicts(model_options, &Options::dictionary(entries), false)?;
        }
        Kind::AdditionalModels => registered.add(hook.clone()),
        _ => unreachable!(),
    }
    Ok(true)
}
pub fn transformer_options_from_hooks(
    hooks: Option<&HookGroup>,
    is_clip: bool,
    options: Option<Options>,
) -> Result<Options, Error> {
    let empty = || Options::dictionary(vec![]);
    if hooks.is_none() || is_clip {
        return Ok(empty());
    }
    let output = options.unwrap_or_else(empty);
    for hook in hooks.unwrap().by_kind(Kind::TransformerOptions) {
        hook.borrow().apply_options(&output)?;
    }
    Ok(output)
}
