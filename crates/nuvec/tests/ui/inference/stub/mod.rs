//! The contract signatures of the tree transforms over a stub `Array` and `Tree`, with
//! `unimplemented!()` bodies: the inference cases check which closure forms compile,
//! independently of the implementation.
#![allow(dead_code, clippy::all)]

#[derive(Clone, Debug)]
pub struct Array;

impl Array {
    pub fn sum(&self) -> Array {
        Array
    }
    pub fn matmul(&self, _: &Array) -> Array {
        Array
    }
    pub fn multiply(&self, _: &Array) -> Array {
        Array
    }
}

#[derive(Debug)]
pub struct Error;
pub type Result<T> = std::result::Result<T, Error>;

pub trait Tree: Clone + 'static {}
impl Tree for Array {}
impl<A: Tree, B: Tree> Tree for (A, B) {}
impl<T: Tree> Tree for Vec<T> {}

pub trait IntoResult {
    type Ok;
    fn into_result(self) -> Result<Self::Ok>;
}
impl<T: Tree> IntoResult for T {
    type Ok = T;
    fn into_result(self) -> Result<T> {
        Ok(self)
    }
}
impl<T> IntoResult for Result<T> {
    type Ok = T;
    fn into_result(self) -> Result<T> {
        self
    }
}

pub trait GradOutput {
    type Value;
}
impl GradOutput for Array {
    type Value = Array;
}
impl<A: Tree> GradOutput for (Array, A) {
    type Value = (Array, A);
}

pub fn grad<P: Tree, F, R>(_f: F) -> impl Fn(&P) -> Result<P>
where
    F: Fn(&P) -> R,
    R: IntoResult<Ok = Array>,
{
    |_| unimplemented!()
}

pub fn value_and_grad<P: Tree, F, R>(
    _f: F,
) -> impl Fn(&P) -> Result<(<R::Ok as GradOutput>::Value, P)>
where
    F: Fn(&P) -> R,
    R: IntoResult,
    R::Ok: GradOutput,
{
    |_| unimplemented!()
}

pub trait InAxis {}
impl InAxis for i32 {}
impl InAxis for Option<i32> {}
pub trait InAxes {}
impl InAxes for i32 {}
impl InAxes for Option<i32> {}
impl<A: InAxis, B: InAxis> InAxes for (A, B) {}

pub struct Vmap<F>(F);
pub fn vmap<I: Tree, O: Tree, F, R>(f: F) -> Vmap<F>
where
    F: Fn(&I) -> R,
    R: IntoResult<Ok = O>,
{
    Vmap(f)
}
impl<F> Vmap<F> {
    pub fn in_axes(self, _: impl InAxes) -> Self {
        self
    }
    pub fn out_axes(self, _: impl InAxes) -> Self {
        self
    }
    pub fn call<I: Tree, O: Tree, R>(&self, _: &I) -> Result<O>
    where
        F: Fn(&I) -> R,
        R: IntoResult<Ok = O>,
    {
        unimplemented!()
    }
}

pub fn compile_mut<S: Tree, I: Tree, O: Tree, F, R>(_f: F) -> impl Fn(&mut S, &I) -> Result<O>
where
    F: Fn(&mut S, &I) -> R,
    R: IntoResult<Ok = O>,
{
    |_, _| unimplemented!()
}

pub fn checkpoint<I: Tree, O: Tree, F, R>(_f: F) -> impl Fn(&I) -> Result<O> + Send + Sync
where
    F: Fn(&I) -> R + Send + Sync + 'static,
    R: IntoResult<Ok = O>,
{
    |_| unimplemented!()
}

#[derive(Clone)]
pub struct Model {
    pub w: Array,
}
impl Tree for Model {}
#[derive(Clone)]
pub struct Opt;
impl Tree for Opt {}
impl Opt {
    pub fn update(&mut self, _: &mut Model, _: &Model) -> Result<()> {
        std::result::Result::Ok(())
    }
}
