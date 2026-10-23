//! Functions with custom transform rules ([`custom_function`]) and gradient
//! checkpointing ([`checkpoint`]), over trees and (the `*_flat` forms) positional arrays.

use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use crate::primitive::Primitive;
use crate::transforms::IntoResult;
use crate::tree::Tree;
use crate::{Array, Error, Result};

type Body = dyn Fn(&[Array]) -> Result<Vec<Array>> + Send + Sync;
type VjpRule = dyn Fn(&[Array], &[Array], &[Array]) -> Result<Vec<Array>> + Send + Sync;
type JvpRule = dyn Fn(&[Array], &[Array]) -> Result<Vec<Array>> + Send + Sync;
type VmapRule = dyn Fn(&[Array], &[Option<usize>]) -> Result<(Vec<Array>, usize)> + Send + Sync;

/// A function and its optional custom rules.
#[derive(Clone)]
pub(crate) struct Rules {
    pub(crate) body: Arc<Body>,
    pub(crate) vjp: Option<Arc<VjpRule>>,
    pub(crate) jvp: Option<Arc<JvpRule>>,
    pub(crate) vmap: Option<Arc<VmapRule>>,
}

/// One call of a custom function, as recorded in the graph: its rules and the number of
/// primal inputs. Equal (and hashed) by the identity of the function.
#[derive(Clone)]
pub struct CustomCall {
    pub(crate) rules: Arc<Rules>,
    pub(crate) primals: usize,
}

impl PartialEq for CustomCall {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.rules, &other.rules) && self.primals == other.primals
    }
}

impl Eq for CustomCall {}

impl Hash for CustomCall {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (Arc::as_ptr(&self.rules) as *const () as usize).hash(state);
        self.primals.hash(state);
    }
}

impl fmt::Debug for CustomCall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CustomCall({} primals)", self.primals)
    }
}

/// A function of positional arrays with custom transform rules; see
/// [`custom_function_flat`].
#[derive(Clone)]
pub struct CustomFunctionFlat {
    rules: Arc<Rules>,
}

/// Wraps `f` (a function of positional arrays) so that transforms use custom rules for it,
/// set with [`CustomFunctionFlat::with_vjp`], [`CustomFunctionFlat::with_jvp`] and
/// [`CustomFunctionFlat::with_vmap`]; the flat form of [`custom_function`].
///
/// Without a rule, a transform differentiates or batches `f` itself, re-running it. `f`'s
/// intermediates are never kept for a transform, so a custom function without rules is a
/// gradient checkpoint (see [`checkpoint_flat`]).
///
/// ```
/// use nuvec_core::{Array, Result, array, custom_function_flat, grad};
///
/// # fn main() -> Result<()> {
/// // A straight-through estimator: forward x², gradient as if the identity.
/// let square = custom_function_flat(|xs| Ok(vec![xs[0].multiply(&xs[0])]))
///     .with_vjp(|_primals, _outputs, cotangents| Ok(cotangents.to_vec()));
/// let g = grad(move |x: &Array| Ok(square.call(std::slice::from_ref(x))?[0].sum(..)));
/// let x = array![3.0, -1.0];
/// assert_eq!(g(&x)?.to_vec::<f32>()?, [1.0, 1.0]);
/// # Ok(())
/// # }
/// ```
pub fn custom_function_flat(
    f: impl Fn(&[Array]) -> Result<Vec<Array>> + Send + Sync + 'static,
) -> CustomFunctionFlat {
    CustomFunctionFlat {
        rules: Arc::new(Rules {
            body: Arc::new(f),
            vjp: None,
            jvp: None,
            vmap: None,
        }),
    }
}

impl CustomFunctionFlat {
    fn with(self, set: impl FnOnce(&mut Rules)) -> Self {
        let mut rules = (*self.rules).clone();
        set(&mut rules);
        Self {
            rules: Arc::new(rules),
        }
    }

    /// Sets the reverse-mode rule: `(primals, outputs, cotangents)` → one cotangent per
    /// primal.
    pub fn with_vjp(
        self,
        rule: impl Fn(&[Array], &[Array], &[Array]) -> Result<Vec<Array>> + Send + Sync + 'static,
    ) -> Self {
        self.with(|r| r.vjp = Some(Arc::new(rule)))
    }

    /// Sets the forward-mode rule: `(primals, tangents)` → one tangent per output (missing
    /// input tangents are passed as zeros).
    pub fn with_jvp(
        self,
        rule: impl Fn(&[Array], &[Array]) -> Result<Vec<Array>> + Send + Sync + 'static,
    ) -> Self {
        self.with(|r| r.jvp = Some(Arc::new(rule)))
    }

    /// Sets the batching rule: `(inputs, batch axes)` (`None` for unbatched inputs) →
    /// the batched outputs and their common batch axis.
    pub fn with_vmap(
        self,
        rule: impl Fn(&[Array], &[Option<i32>]) -> Result<(Vec<Array>, i32)> + Send + Sync + 'static,
    ) -> Self {
        let rule = move |xs: &[Array], axes: &[Option<usize>]| {
            let axes: Vec<Option<i32>> = axes
                .iter()
                .map(|a| a.map(|a| i32::try_from(a).unwrap_or(i32::MAX)))
                .collect();
            let (out, axis) = rule(xs, &axes)?;
            let axis = usize::try_from(axis).map_err(|_| {
                Error::InvalidArgument(format!(
                    "custom_function: the vmap rule returned the negative output axis {axis}"
                ))
            })?;
            Ok((out, axis))
        };
        self.with(|r| r.vmap = Some(Arc::new(rule)))
    }

    /// Calls the function on `inputs`. Values equal `f(inputs)`; transforms of the
    /// results use the custom rules.
    pub fn call(&self, inputs: &[Array]) -> Result<Vec<Array>> {
        // `f` runs on untraced stop-gradient views: its graph detaches after evaluation
        // even inside a transform, and transforms never walk into it.
        let views = inputs
            .iter()
            .map(|x| Ok(Array::record(Primitive::StopGradient, vec![x.clone()], None)?.remove(0)))
            .collect::<Result<Vec<_>>>()?;
        let outputs = (self.rules.body)(&views)?;
        let trace = crate::transforms::inherited(inputs);
        let call = CustomCall {
            rules: self.rules.clone(),
            primals: inputs.len(),
        };
        let all = inputs.iter().chain(&outputs).cloned().collect();
        Array::record(Primitive::Custom(call), all, trace)
    }
}

/// Returns `f` (a function of positional arrays) with gradient checkpointing; the flat form
/// of [`checkpoint`].
///
/// ```
/// use nuvec_core::{Array, Result, checkpoint_flat, grad};
///
/// # fn main() -> Result<()> {
/// let block = checkpoint_flat(|xs| Ok(vec![xs[0].sin().sin()]));
/// let g = grad(move |x: &Array| Ok(block(std::slice::from_ref(x))?[0].sum(..)));
/// let x = Array::from(0.5f32);
/// let expect = 0.5f32.sin().cos() * 0.5f32.cos();
/// assert!((g(&x)?.item::<f32>()? - expect).abs() < 1e-6);
/// # Ok(())
/// # }
/// ```
pub fn checkpoint_flat(
    f: impl Fn(&[Array]) -> Result<Vec<Array>> + Send + Sync + 'static,
) -> impl Fn(&[Array]) -> Result<Vec<Array>> + Send + Sync {
    let function = custom_function_flat(f);
    move |inputs| function.call(inputs)
}

// ── Over trees ──────────────────────────────────────────────────────────────

type TreeBody<I, O> = dyn Fn(&I) -> Result<O> + Send + Sync;
type TreeVjp<I, O> = dyn Fn(&I, &O, &O) -> Result<I> + Send + Sync;
type TreeJvp<I, O> = dyn Fn(&I, &I) -> Result<O> + Send + Sync;
type TreeVmap<I, O> = dyn Fn(&I, &[Option<i32>]) -> Result<(O, i32)> + Send + Sync;

/// A function of a tree with custom transform rules; see [`custom_function`].
pub struct CustomFunction<I, O> {
    body: Arc<TreeBody<I, O>>,
    vjp: Option<Arc<TreeVjp<I, O>>>,
    jvp: Option<Arc<TreeJvp<I, O>>>,
    vmap: Option<Arc<TreeVmap<I, O>>>,
    /// Restore the thread's random key around recomputations (checkpoints).
    replay_random: bool,
}

impl<I, O> Clone for CustomFunction<I, O> {
    fn clone(&self) -> Self {
        Self {
            body: self.body.clone(),
            vjp: self.vjp.clone(),
            jvp: self.jvp.clone(),
            vmap: self.vmap.clone(),
            replay_random: self.replay_random,
        }
    }
}

/// Wraps `f` (a function of a tree, returning a tree) so that transforms use custom rules
/// for it, set with [`CustomFunction::with_vjp`], [`CustomFunction::with_jvp`] and
/// [`CustomFunction::with_vmap`].
///
/// Without a rule, a transform differentiates or batches `f` itself, re-running it; `f`'s
/// intermediates are never kept for a transform. The rules are stored in the graph (and
/// may run later, on another thread), so `f` and the rules are `Send + Sync + 'static`.
///
/// ```
/// use nuvec_core::{Array, Result, array, custom_function, grad};
///
/// # fn main() -> Result<()> {
/// // A straight-through estimator: forward x², gradient as if the identity.
/// let square = custom_function(|x: &Array| x.multiply(x))
///     .with_vjp(|_x: &Array, _y: &Array, cotangent: &Array| Ok(cotangent.clone()));
/// let g = grad(move |x: &Array| Ok(square.call(x)?.sum(..)));
/// assert_eq!(g(&array![3.0, -1.0])?.to_vec::<f32>()?, [1.0, 1.0]);
/// # Ok(())
/// # }
/// ```
pub fn custom_function<I, O, F, R>(f: F) -> CustomFunction<I, O>
where
    I: Tree + Send + Sync,
    O: Tree + Send + Sync,
    F: Fn(&I) -> R + Send + Sync + 'static,
    R: IntoResult<Ok = O>,
{
    CustomFunction {
        body: Arc::new(move |i: &I| f(i).into_result()),
        vjp: None,
        jvp: None,
        vmap: None,
        replay_random: false,
    }
}

impl<I, O> CustomFunction<I, O>
where
    I: Tree + Send + Sync,
    O: Tree + Send + Sync,
{
    /// Sets the reverse-mode rule: `(primals, outputs, cotangents)` → the cotangent of the
    /// primals (a tree like them).
    pub fn with_vjp(
        mut self,
        rule: impl Fn(&I, &O, &O) -> Result<I> + Send + Sync + 'static,
    ) -> Self {
        self.vjp = Some(Arc::new(rule));
        self
    }

    /// Sets the forward-mode rule: `(primals, tangents)` → the outputs' tangents (missing
    /// input tangents are passed as zeros).
    pub fn with_jvp(mut self, rule: impl Fn(&I, &I) -> Result<O> + Send + Sync + 'static) -> Self {
        self.jvp = Some(Arc::new(rule));
        self
    }

    /// Sets the batching rule: `(inputs, batch axis of each input leaf)` (`None` for
    /// unbatched leaves) → the batched outputs and their common batch axis.
    pub fn with_vmap(
        mut self,
        rule: impl Fn(&I, &[Option<i32>]) -> Result<(O, i32)> + Send + Sync + 'static,
    ) -> Self {
        self.vmap = Some(Arc::new(rule));
        self
    }

    /// Calls the function on `inputs`. Values equal `f(inputs)`; transforms of the results
    /// use the custom rules.
    pub fn call(&self, inputs: &I) -> Result<O> {
        let template = Arc::new(inputs.clone());
        let output: Arc<Mutex<Option<O>>> = Arc::new(Mutex::new(None));
        let out_template = {
            let output = output.clone();
            move || {
                output
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .clone()
                    .ok_or_else(|| Error::InvalidArgument("custom_function: no output yet".into()))
            }
        };
        let body = {
            let (f, template, output) = (self.body.clone(), template.clone(), output.clone());
            // A checkpoint's recomputation draws the same random numbers as its forward
            // pass: the thread's key is restored around every run after the first.
            let key = self.replay_random.then(crate::random::sequence_key);
            let first = AtomicBool::new(true);
            move |xs: &[Array]| -> Result<Vec<Array>> {
                let restore = match &key {
                    Some(k) => {
                        let current = crate::random::sequence_key();
                        crate::random::set_sequence_key(k.clone());
                        (!first.swap(false, Ordering::AcqRel)).then_some(current)
                    }
                    None => None,
                };
                let out = template.unflatten(xs.iter().cloned()).and_then(|i| f(&i));
                if let Some(current) = restore {
                    crate::random::set_sequence_key(current);
                }
                let out = out?;
                let leaves = out.leaves();
                *output.lock().unwrap_or_else(PoisonError::into_inner) = Some(out);
                Ok(leaves)
            }
        };
        let vjp: Option<Arc<VjpRule>> = self.vjp.clone().map(|rule| {
            let (template, out_template) = (template.clone(), out_template.clone());
            Arc::new(move |p: &[Array], o: &[Array], c: &[Array]| {
                let tpl = out_template()?;
                let i = template.unflatten(p.iter().cloned())?;
                let o = tpl.unflatten(o.iter().cloned())?;
                let c = tpl.unflatten(c.iter().cloned())?;
                Ok(rule(&i, &o, &c)?.leaves())
            }) as Arc<VjpRule>
        });
        let jvp: Option<Arc<JvpRule>> = self.jvp.clone().map(|rule| {
            let template = template.clone();
            Arc::new(move |p: &[Array], t: &[Array]| {
                let i = template.unflatten(p.iter().cloned())?;
                let t = template.unflatten(t.iter().cloned())?;
                Ok(rule(&i, &t)?.leaves())
            }) as Arc<JvpRule>
        });
        let vmap: Option<Arc<VmapRule>> = self.vmap.clone().map(|rule| {
            let (template, out_template) = (template.clone(), out_template.clone());
            Arc::new(move |xs: &[Array], axes: &[Option<usize>]| {
                let axes: Vec<Option<i32>> = axes
                    .iter()
                    .map(|a| a.map(|a| i32::try_from(a).unwrap_or(i32::MAX)))
                    .collect();
                let i = template.unflatten(xs.iter().cloned())?;
                let (o, axis) = rule(&i, &axes)?;
                let axis = usize::try_from(axis).map_err(|_| {
                    Error::InvalidArgument(format!(
                        "custom_function: the vmap rule returned the negative output axis {axis}"
                    ))
                })?;
                let _ = out_template()?;
                Ok((o.leaves(), axis))
            }) as Arc<VmapRule>
        });
        let flat = CustomFunctionFlat {
            rules: Arc::new(Rules {
                body: Arc::new(body),
                vjp,
                jvp,
                vmap,
            }),
        };
        let leaves = flat.call(&inputs.leaves())?;
        out_template()?.unflatten(leaves)
    }
}

/// Returns `f` with gradient checkpointing: values and derivatives equal `f`'s, but `f`'s
/// intermediates are not kept for the backward pass; `f` runs again to differentiate.
/// Random numbers drawn inside `f` (dropout) are drawn again with the same key, so the
/// recomputation sees the same masks.
///
/// `f` is stored in the graph for the recomputation, so it is `Send + Sync + 'static`
/// (it cannot borrow locals: move them in).
///
/// ```
/// use nuvec_core::{Array, Result, checkpoint, grad};
///
/// # fn main() -> Result<()> {
/// let block = checkpoint(|x: &Array| x.sin().sin());
/// let g = grad(move |x: &Array| Ok(block(x)?.sum(..)));
/// let x = Array::from(0.5f32);
/// let expect = 0.5f32.sin().cos() * 0.5f32.cos();
/// assert!((g(&x)?.item::<f32>()? - expect).abs() < 1e-6);
/// # Ok(())
/// # }
/// ```
pub fn checkpoint<I, O, F, R>(f: F) -> impl Fn(&I) -> Result<O> + Send + Sync
where
    I: Tree + Send + Sync,
    O: Tree + Send + Sync,
    F: Fn(&I) -> R + Send + Sync + 'static,
    R: IntoResult<Ok = O>,
{
    let mut function = custom_function(f);
    function.replay_random = true;
    move |inputs| function.call(inputs)
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::array::State;

    /// Whether `f`'s forward intermediate keeps its graph after `grad` evaluates the
    /// forward pass inside the transform.
    fn intermediate_retained(checkpointed: bool) -> bool {
        let first = Arc::new(Mutex::new(None::<Array>));
        let stored = first.clone();
        let body = move |xs: &[Array]| {
            let h = crate::ops::imp::sin(&xs[0])?;
            stored.lock().unwrap().get_or_insert_with(|| h.clone());
            Ok(vec![crate::ops::imp::sin(&h)?])
        };
        let f: Box<Body> = if checkpointed {
            Box::new(checkpoint_flat(body))
        } else {
            Box::new(body)
        };
        // Checked inside the transform: retained nodes detach when it exits.
        let retained = Arc::new(Mutex::new(false));
        let seen = retained.clone();
        let g = crate::grad(move |x: &Array| {
            let y = f(std::slice::from_ref(x))?.remove(0);
            y.eval()?;
            let h = first.lock().unwrap().take().unwrap();
            *seen.lock().unwrap() = matches!(&*h.node.lock(), State::Retained { .. });
            crate::ops::imp::sum(&y, &[], false)
        });
        let _ = g(&Array::from(0.5f32)).unwrap();
        *retained.lock().unwrap()
    }

    #[test]
    fn checkpoint_does_not_keep_intermediates() {
        assert!(
            intermediate_retained(false),
            "without checkpoint the graph is kept"
        );
        assert!(
            !intermediate_retained(true),
            "with checkpoint it is released"
        );
    }
}
