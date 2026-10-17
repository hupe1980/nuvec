//! Functions with custom transform rules ([`custom_function`]) and gradient
//! checkpointing ([`checkpoint`]).

use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::primitive::Primitive;
use crate::{Array, Result};

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

/// A function with custom transform rules; see [`custom_function`].
#[derive(Clone)]
pub struct CustomFunction {
    rules: Arc<Rules>,
}

/// Wraps `f` so that transforms use custom rules for it, set with
/// [`CustomFunction::with_vjp`], [`CustomFunction::with_jvp`] and
/// [`CustomFunction::with_vmap`].
///
/// Without a rule, a transform differentiates or batches `f` itself, re-running it. `f`'s
/// intermediates are never kept for a transform, so a custom function without rules is a
/// gradient checkpoint (see [`checkpoint`]).
///
/// ```
/// use nuvec_core::{Array, Result, custom_function, grad, multiply, sum};
///
/// # fn main() -> Result<()> {
/// // A straight-through estimator: forward x², gradient as if the identity.
/// let square = custom_function(|xs| Ok(vec![multiply(&xs[0], &xs[0])?]))
///     .with_vjp(|_primals, _outputs, cotangents| Ok(cotangents.to_vec()));
/// let g = grad(move |x: &Array| sum(&square.call(std::slice::from_ref(x))?[0], &[], false));
/// let x = Array::from_slice(&[3.0f32, -1.0], &[2])?;
/// assert_eq!(g(&x)?.to_vec::<f32>()?, [1.0, 1.0]);
/// # Ok(())
/// # }
/// ```
pub fn custom_function(
    f: impl Fn(&[Array]) -> Result<Vec<Array>> + Send + Sync + 'static,
) -> CustomFunction {
    CustomFunction {
        rules: Arc::new(Rules {
            body: Arc::new(f),
            vjp: None,
            jvp: None,
            vmap: None,
        }),
    }
}

impl CustomFunction {
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
        rule: impl Fn(&[Array], &[Option<usize>]) -> Result<(Vec<Array>, usize)> + Send + Sync + 'static,
    ) -> Self {
        self.with(|r| r.vmap = Some(Arc::new(rule)))
    }

    /// Calls the function on `inputs`. Values equal `f(inputs)`; transforms of the
    /// results use the custom rules.
    pub fn call(&self, inputs: &[Array]) -> Result<Vec<Array>> {
        // `f` runs on untraced stop-gradient views: its graph detaches after evaluation
        // even inside a transform, and transforms never walk into it.
        let views = inputs
            .iter()
            .map(|x| Ok(Array::record(Primitive::StopGradient, vec![x.clone()], false)?.remove(0)))
            .collect::<Result<Vec<_>>>()?;
        let outputs = (self.rules.body)(&views)?;
        let tracer = inputs.iter().any(|a| a.node.tracer);
        let call = CustomCall {
            rules: self.rules.clone(),
            primals: inputs.len(),
        };
        let all = inputs.iter().chain(&outputs).cloned().collect();
        Array::record(Primitive::Custom(call), all, tracer)
    }
}

/// Returns `f` with gradient checkpointing: values and derivatives equal `f`'s, but `f`'s
/// intermediates are not kept for the backward pass; `f` runs again to differentiate.
///
/// ```
/// use nuvec_core::{Array, Result, checkpoint, grad, sin, sum};
///
/// # fn main() -> Result<()> {
/// let block = checkpoint(|xs| Ok(vec![sin(&sin(&xs[0])?)?]));
/// let g = grad(move |x: &Array| sum(&block(std::slice::from_ref(x))?[0], &[], false));
/// let x = Array::scalar(0.5f32);
/// let expect = 0.5f32.sin().cos() * 0.5f32.cos();
/// assert!((g(&x)?.item::<f32>()? - expect).abs() < 1e-6);
/// # Ok(())
/// # }
/// ```
pub fn checkpoint(
    f: impl Fn(&[Array]) -> Result<Vec<Array>> + Send + Sync + 'static,
) -> impl Fn(&[Array]) -> Result<Vec<Array>> + Send + Sync {
    let function = custom_function(f);
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
            let h = crate::sin(&xs[0])?;
            stored.lock().unwrap().get_or_insert_with(|| h.clone());
            Ok(vec![crate::sin(&h)?])
        };
        let f: Box<Body> = if checkpointed {
            Box::new(checkpoint(body))
        } else {
            Box::new(body)
        };
        let g = crate::grad(move |x: &Array| {
            let y = f(std::slice::from_ref(x))?.remove(0);
            y.eval()?;
            crate::sum(&y, &[], false)
        });
        g(&Array::scalar(0.5f32)).unwrap();
        let h = first.lock().unwrap().take().unwrap();
        let retained = matches!(&*h.node.lock(), State::Retained { .. });
        retained
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
