//! Indexing (`x.i(..)`, MLX's `x[...]`) and functional updates (`x.at(..).set(v)`, JAX's
//! `x.at[...].set(v)`).
//!
//! An index is an integer, a range (over `i32` or `usize`, optionally `.step(n)`), `..`,
//! [`Ellipsis`], [`NewAxis`], an integer array, or a tuple of up to eight of these. A rank
//! known only at run time takes a slice or `Vec` of ranges, one per leading axis. Basic
//! indices (no arrays) give views; integer arrays gather, following MLX's (NumPy's) rules for
//! broadcasting the index arrays and placing their dimensions.

use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};
use std::panic::Location;

use smallvec::SmallVec;

use crate::ops::imp;
use crate::scalar::ScalarOrArray;
use crate::shape::Shape;
use crate::view::SliceArg;
use crate::{Array, Category, DType, Error, Result};

mod sealed {
    pub trait Sealed {}
}

/// `...` in an index: as many full slices as the other entries leave dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Ellipsis;

/// `None` (`np.newaxis`) in an index: a new dimension of size 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NewAxis;

/// A range with a step, made by [`Step::step`]: the range selects positions and the step
/// walks them, backwards when negative (ndarray's rule). `(0..6).step(2)` is Python's
/// `0:6:2`, `(..).step(-1)` is `::-1`, `(1..4).step(-1)` is `3:0:-1` (positions 3, 2, 1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stepped<R> {
    range: R,
    step: i32,
}

/// One entry of an index after lowering.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub enum IndexItem {
    /// A single position (removes the dimension).
    Int(i64),
    /// A range `start..stop` (negative bounds count from the end; clamped) with a step: a
    /// negative step walks the range backwards from its end (ndarray's rule).
    Slice {
        start: Option<i64>,
        stop: Option<i64>,
        step: i64,
    },
    /// A new size-1 dimension.
    NewAxis,
    /// The remaining dimensions.
    Ellipsis,
    /// An integer index array (gather).
    Array(Array),
}

/// Bounds of a range, for [`Step`].
#[doc(hidden)]
pub trait Bounds: sealed::Sealed {
    #[doc(hidden)]
    fn bounds(&self) -> (Option<i64>, Option<i64>);
}

/// Adds a step to a range used as an index: `x.i((0..6).step(2))`, `x.i((..).step(-1))`
/// (see [`Stepped`] for negative steps).
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot take a step",
    label = "`.step(n)` applies to ranges such as `0..6`, `1..`, `..=4` or `..`"
)]
pub trait Step: Bounds + Sized {
    /// This range with every `n`-th position (negative: backwards, as Python's `::-1`).
    fn step(self, n: i32) -> Stepped<Self> {
        Stepped {
            range: self,
            step: n,
        }
    }
}

/// The exclusive end of an inclusive range ending at `end` (`..=-1` runs to the end).
fn inclusive_end(end: i64) -> Option<i64> {
    if end == -1 {
        None
    } else {
        Some(end.saturating_add(1))
    }
}

use crate::axes::wide;

macro_rules! bounds {
    ($($t:ty => |$r:ident| $e:expr),* $(,)?) => {$(
        impl sealed::Sealed for $t {}
        impl Bounds for $t {
            fn bounds(&self) -> (Option<i64>, Option<i64>) {
                let $r = self;
                $e
            }
        }
        impl Step for $t {}
    )*};
}

bounds! {
    Range<i32> => |r| (Some(r.start.into()), Some(r.end.into())),
    Range<usize> => |r| (Some(wide(r.start)), Some(wide(r.end))),
    RangeFrom<i32> => |r| (Some(r.start.into()), None),
    RangeFrom<usize> => |r| (Some(wide(r.start)), None),
    RangeTo<i32> => |r| (None, Some(r.end.into())),
    RangeTo<usize> => |r| (None, Some(wide(r.end))),
    RangeInclusive<i32> => |r| (Some((*r.start()).into()), inclusive_end((*r.end()).into())),
    RangeInclusive<usize> => |r| (Some(wide(*r.start())), inclusive_end(wide(*r.end()))),
    RangeToInclusive<i32> => |r| (None, inclusive_end(r.end.into())),
    RangeToInclusive<usize> => |r| (None, inclusive_end(wide(r.end))),
    RangeFull => |_r| (None, None),
}

/// An index of [`Array::i`] and [`Array::at`]: an `i32` or `usize`, a range over either
/// (`1..3`, `1..`, `..3`, `1..=2`, `..=2`, with an optional `.step(n)`), `..`, [`Ellipsis`],
/// [`NewAxis`], an integer index array (`&Array` or `Array`), or a tuple of up to eight of
/// these. For a rank known only at run time, a slice or `Vec` of ranges indexes the leading
/// axes, one range each (`&[1..3, 0..2]`); integers are not accepted there, so a list is
/// never mistaken for a gather.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not an index",
    label = "expected an integer, a range (`1..3`, `..=2`, `(0..6).step(2)`), `..`, `Ellipsis`, `NewAxis`, an index array (`&idx`), a tuple of these, or a slice of ranges"
)]
pub trait ArrayIndex: sealed::Sealed {
    #[doc(hidden)]
    fn index_items(self, out: &mut SmallVec<[IndexItem; 4]>);
}

fn slice_item(start: Option<i64>, stop: Option<i64>, step: i64) -> IndexItem {
    IndexItem::Slice { start, stop, step }
}

/// A range item as a Python slice of a dimension of size `n`.
fn python_slice(start: Option<i64>, stop: Option<i64>, step: i64, n: usize) -> SliceArg {
    let n = n as i64;
    let norm = |b: i64| (if b < 0 { b + n } else { b }).clamp(0, n);
    let s = start.map_or(0, norm);
    let e = stop.map_or(n, norm).max(s);
    let step = step.clamp(isize::MIN as i64, isize::MAX as i64) as isize;
    if step > 0 {
        SliceArg::new(Some(s as isize), Some(e as isize), step)
    } else if e == s {
        SliceArg::new(Some(0), Some(0), 1)
    } else {
        let stop = if s == 0 { None } else { Some(s as isize - 1) };
        SliceArg::new(Some(e as isize - 1), stop, step)
    }
}

macro_rules! range_index {
    ($($t:ty),*) => {$(
        impl ArrayIndex for $t {
            fn index_items(self, out: &mut SmallVec<[IndexItem; 4]>) {
                let (start, stop) = self.bounds();
                out.push(slice_item(start, stop, 1));
            }
        }
        impl sealed::Sealed for Stepped<$t> {}
        impl ArrayIndex for Stepped<$t> {
            fn index_items(self, out: &mut SmallVec<[IndexItem; 4]>) {
                let (start, stop) = self.range.bounds();
                out.push(slice_item(start, stop, self.step.into()));
            }
        }
    )*};
}

range_index!(
    Range<i32>,
    Range<usize>,
    RangeFrom<i32>,
    RangeFrom<usize>,
    RangeTo<i32>,
    RangeTo<usize>,
    RangeInclusive<i32>,
    RangeInclusive<usize>,
    RangeToInclusive<i32>,
    RangeToInclusive<usize>,
    RangeFull
);

impl sealed::Sealed for i32 {}
impl ArrayIndex for i32 {
    fn index_items(self, out: &mut SmallVec<[IndexItem; 4]>) {
        out.push(IndexItem::Int(self.into()));
    }
}

impl sealed::Sealed for usize {}
impl ArrayIndex for usize {
    fn index_items(self, out: &mut SmallVec<[IndexItem; 4]>) {
        out.push(IndexItem::Int(wide(self)));
    }
}

impl sealed::Sealed for Ellipsis {}
impl ArrayIndex for Ellipsis {
    fn index_items(self, out: &mut SmallVec<[IndexItem; 4]>) {
        out.push(IndexItem::Ellipsis);
    }
}

impl sealed::Sealed for NewAxis {}
impl ArrayIndex for NewAxis {
    fn index_items(self, out: &mut SmallVec<[IndexItem; 4]>) {
        out.push(IndexItem::NewAxis);
    }
}

impl sealed::Sealed for &Array {}
impl ArrayIndex for &Array {
    fn index_items(self, out: &mut SmallVec<[IndexItem; 4]>) {
        out.push(IndexItem::Array(self.clone()));
    }
}

impl sealed::Sealed for Array {}
impl ArrayIndex for Array {
    fn index_items(self, out: &mut SmallVec<[IndexItem; 4]>) {
        out.push(IndexItem::Array(self));
    }
}

macro_rules! tuple_index {
    ($($name:ident)+) => {
        impl<$($name: ArrayIndex),+> sealed::Sealed for ($($name,)+) {}
        #[diagnostic::do_not_recommend]
        impl<$($name: ArrayIndex),+> ArrayIndex for ($($name,)+) {
            #[allow(non_snake_case)]
            fn index_items(self, out: &mut SmallVec<[IndexItem; 4]>) {
                let ($($name,)+) = self;
                $($name.index_items(out);)+
            }
        }
    };
}

/// One range per leading axis: an index whose rank is known only at run time.
macro_rules! ranges_index {
    ($($t:ty),*) => {$(
        impl<R: Bounds> sealed::Sealed for $t {}
        impl<R: Bounds> ArrayIndex for $t {
            fn index_items(self, out: &mut SmallVec<[IndexItem; 4]>) {
                for r in self.iter() {
                    let (start, stop) = r.bounds();
                    out.push(slice_item(start, stop, 1));
                }
            }
        }
    )*};
}

ranges_index!(&[R], Vec<R>, &Vec<R>);

tuple_index!(A);
tuple_index!(A B);
tuple_index!(A B C);
tuple_index!(A B C D);
tuple_index!(A B C D E);
tuple_index!(A B C D E F);
tuple_index!(A B C D E F G);
tuple_index!(A B C D E F G H);

fn items_of(index: impl ArrayIndex) -> SmallVec<[IndexItem; 4]> {
    let mut items = SmallVec::new();
    index.index_items(&mut items);
    items
}

/// The arrays of an index, for failure reporting.
fn index_arrays(items: &[IndexItem]) -> impl Iterator<Item = &Array> {
    items.iter().filter_map(|i| match i {
        IndexItem::Array(a) => Some(a),
        _ => None,
    })
}

/// Where an output dimension of an index comes from.
#[derive(Debug, Clone)]
enum Slot {
    /// A new size-1 dimension.
    New,
    /// Input dimension `d`, sliced.
    Dim(usize),
    /// Input dimension `d`, indexed by an integer (`None`) or an index array.
    Adv(usize, Option<Array>, i64),
}

/// An index resolved against an array of `shape`: one slice per input dimension and the
/// output slots.
struct Resolved {
    args: Vec<SliceArg>,
    slots: Vec<Slot>,
}

impl Resolved {
    fn has_arrays(&self) -> bool {
        self.slots
            .iter()
            .any(|s| matches!(s, Slot::Adv(_, Some(_), _)))
    }
}

fn resolve(items: &[IndexItem], shape: &[usize]) -> Result<Resolved> {
    let ndim = shape.len();
    let ellipses = items
        .iter()
        .filter(|i| matches!(i, IndexItem::Ellipsis))
        .count();
    if ellipses > 1 {
        return Err(Error::invalid_index(
            "index",
            "an index can only have a single ellipsis",
        ));
    }
    let consuming = items
        .iter()
        .filter(|i| !matches!(i, IndexItem::Ellipsis | IndexItem::NewAxis))
        .count();
    if consuming > ndim {
        return Err(Error::invalid_index(
            "index",
            format!("too many indices ({consuming}) for an array of {ndim} dimensions"),
        ));
    }
    let mut args = Vec::with_capacity(ndim);
    let mut slots = Vec::with_capacity(items.len() + ndim);
    let mut dim = 0usize;
    let full = |dim: usize, args: &mut Vec<SliceArg>, slots: &mut Vec<Slot>| {
        args.push(SliceArg::full());
        slots.push(Slot::Dim(dim));
    };
    for item in items {
        match item {
            IndexItem::Ellipsis => {
                for _ in 0..ndim - consuming {
                    full(dim, &mut args, &mut slots);
                    dim += 1;
                }
            }
            IndexItem::NewAxis => slots.push(Slot::New),
            IndexItem::Slice { start, stop, step } => {
                if *step == 0 {
                    return Err(Error::invalid_index("index", "slice step must not be zero"));
                }
                args.push(python_slice(*start, *stop, *step, shape[dim]));
                slots.push(Slot::Dim(dim));
                dim += 1;
            }
            IndexItem::Int(i) => {
                let n = shape[dim] as i64;
                let k = if *i < 0 { i + n } else { *i };
                if !(0..n).contains(&k) {
                    return Err(Error::invalid_index(
                        "index",
                        format!("index {i} is out of bounds for axis {dim} with size {n}"),
                    ));
                }
                args.push(SliceArg::new(Some(k as isize), Some(k as isize + 1), 1));
                slots.push(Slot::Adv(dim, None, k));
                dim += 1;
            }
            IndexItem::Array(a) => {
                if !matches!(a.dtype().category(), Category::Signed | Category::Unsigned) {
                    return Err(Error::dtype_mismatch("index", a.dtype(), DType::I32));
                }
                args.push(SliceArg::full());
                slots.push(Slot::Adv(dim, Some(a.clone()), 0));
                dim += 1;
            }
        }
    }
    while dim < ndim {
        full(dim, &mut args, &mut slots);
        dim += 1;
    }
    Ok(Resolved { args, slots })
}

/// The shape of the basic index `r` applied to `sliced` (the array after slicing).
fn basic_shape(r: &Resolved, sliced: &[usize]) -> Shape {
    r.slots
        .iter()
        .filter_map(|s| match s {
            Slot::New => Some(1),
            Slot::Dim(d) => Some(sliced[*d]),
            Slot::Adv(..) => None,
        })
        .collect()
}

/// `a[index]` for a basic index: one slice view, reshaped to drop integer positions and
/// insert new axes (still a view).
fn basic(a: &Array, r: &Resolved) -> Result<Array> {
    let view = imp::slice(a, &r.args)?;
    let shape = basic_shape(r, view.dims());
    imp::reshape_to(&view, shape)
}

/// The index arrays of `r`'s advanced slots (integers as 0-D arrays), with their input
/// dimensions, wrapped (negative counts from the end) and clamped into range.
fn advanced_indices(a: &Array, r: &Resolved) -> Result<Vec<(usize, Array)>> {
    r.slots
        .iter()
        .filter_map(|s| match s {
            Slot::Adv(d, idx, k) => Some((*d, idx.clone(), *k)),
            _ => None,
        })
        .map(|(d, idx, k)| {
            let n = a.dims()[d] as i64;
            let idx = match idx {
                Some(idx) => {
                    let idx = imp::astype(&idx, DType::I64)?;
                    let wrapped = imp::where_(imp::less(&idx, 0)?, imp::add(&idx, n)?, &idx)?;
                    imp::minimum(&imp::maximum(&wrapped, 0)?, (n - 1).max(0))?
                }
                None => crate::creation::from_data(&[k], &[])?,
            };
            Ok((d, idx))
        })
        .collect()
}

/// One linear index over the indexed axes `adv` of `v` (row-major over them), broadcast to
/// the index arrays' common shape: `(index, that shape, the indexed axes' element count)`.
fn linear_index(v: &Array, adv: &[(usize, Array)]) -> Result<(Array, Shape, usize)> {
    let mut bshape: Shape = Shape::new();
    for (_, idx) in adv {
        bshape = crate::view::broadcast_shapes("index", &bshape, idx.dims())?;
    }
    let sizes: Vec<usize> = adv.iter().map(|&(d, _)| v.dims()[d]).collect();
    let total: usize = sizes.iter().product();
    let mut linear: Option<Array> = None;
    let mut stride = 1usize;
    for ((_, idx), &n) in adv.iter().zip(&sizes).rev() {
        let term = imp::multiply(idx, stride as i64)?;
        linear = Some(match linear {
            None => term,
            Some(l) => imp::add(&l, &term)?,
        });
        stride *= n;
    }
    let linear = imp::broadcast_to(
        &linear.unwrap_or_else(|| unreachable!("an advanced index has an index array")),
        &bshape,
    )?;
    let index_dtype = if total <= i32::MAX as usize {
        DType::I32
    } else {
        DType::I64
    };
    Ok((imp::astype(&linear, index_dtype)?, bshape, total))
}

/// `a[index]` with index arrays (MLX's advanced indexing): the index arrays (and integers)
/// broadcast together; their dimensions replace the indexed axes where those are adjacent,
/// else they come first. Lowered to one `take` over the indexed axes flattened.
fn advanced(a: &Array, r: &Resolved) -> Result<Array> {
    // Slices first (a view): every input dimension stays; integers index like 0-D arrays,
    // so their dimensions are not sliced.
    let mut args = r.args.clone();
    for slot in &r.slots {
        if let Slot::Adv(d, None, _) = slot {
            args[*d] = SliceArg::full();
        }
    }
    let v = imp::slice(a, &args)?;
    let adv = advanced_indices(&v, r)?;
    let adv_dims: Vec<usize> = adv.iter().map(|(d, _)| *d).collect();
    let rest: Vec<usize> = (0..v.ndim()).filter(|d| !adv_dims.contains(d)).collect();
    let (linear, bshape, total) = linear_index(&v, &adv)?;
    // Indexed axes first, flattened into one, gathered.
    let perm: Vec<usize> = adv_dims.iter().chain(&rest).copied().collect();
    let vt = imp::transpose(&v, &perm)?;
    let mut flat: Shape = Shape::from_elem(total, 1);
    flat.extend(rest.iter().map(|&d| v.dims()[d]));
    let vf = imp::reshape_to(&vt, flat)?;
    let out = crate::indexing::take(&vf, &linear, Some(0))?;
    // `out` is [bshape..., rest...]; arrange it as MLX places the dimensions.
    let nb = bshape.len();
    let adv_positions: Vec<usize> = r
        .slots
        .iter()
        .enumerate()
        .filter(|(_, s)| matches!(s, Slot::Adv(..)))
        .map(|(i, _)| i)
        .collect();
    let adjacent = adv_positions.windows(2).all(|w| w[1] == w[0] + 1);
    let mut order: Vec<usize> = Vec::with_capacity(out.ndim());
    let mut shape: Shape = Shape::new();
    let mut next_rest = nb;
    let mut placed = !adjacent;
    if !adjacent {
        order.extend(0..nb);
        shape.extend_from_slice(&bshape);
    }
    for slot in &r.slots {
        match slot {
            Slot::New => shape.push(1),
            Slot::Dim(_) => {
                order.push(next_rest);
                shape.push(out.dims()[next_rest]);
                next_rest += 1;
            }
            Slot::Adv(..) => {
                if !placed {
                    order.extend(0..nb);
                    shape.extend_from_slice(&bshape);
                    placed = true;
                }
            }
        }
    }
    let arranged = imp::transpose(&out, &order)?;
    imp::reshape_to(&arranged, shape)
}

fn index_with(a: &Array, items: &[IndexItem]) -> Result<Array> {
    let r = resolve(items, a.dims())?;
    if r.has_arrays() {
        advanced(a, &r)
    } else {
        basic(a, &r)
    }
}

impl Array {
    /// The array indexed as MLX's `x[...]`: integers, ranges (`1..3`, `..=2`,
    /// `(0..6).step(2)`, `(..).step(-1)`), `..`, [`Ellipsis`], [`NewAxis`], integer index
    /// arrays, and tuples of these (one entry per dimension, leading dimensions first).
    ///
    /// Basic indices give views (no copy); integer arrays gather. Integers out of bounds,
    /// two ellipses, too many entries or a zero step make the result a failed array
    /// ([`crate::OpErrorKind::InvalidIndex`]); slices clamp as in Python.
    ///
    /// ```
    /// use nuvec_core::{Array, Ellipsis, NewAxis, Step};
    ///
    /// let x = Array::from_slice(&(0..24).collect::<Vec<i32>>(), [2, 3, 4]);
    /// assert_eq!(x.i(0).shape(), [3, 4]);                          // x[0]
    /// assert_eq!(x.i((.., 1..3)).shape(), [2, 2, 4]);              // x[:, 1:3]
    /// assert_eq!(x.i((Ellipsis, -1)).to_vec::<i32>().unwrap()[0], 3); // x[..., -1]
    /// assert_eq!(x.i((.., NewAxis)).shape(), [2, 1, 3, 4]);        // x[:, None]
    /// assert_eq!(x.i((.., .., (0..4).step(2))).shape(), [2, 3, 2]); // x[:, :, 0:4:2]
    /// let rows = Array::from_slice(&[1i32, 0], [2]);
    /// assert_eq!(x.i((0, &rows)).shape(), [2, 4]);                 // x[0, rows]
    /// assert!(x.i(5).is_failed());
    /// ```
    #[track_caller]
    pub fn i(&self, index: impl ArrayIndex) -> Array {
        let items = items_of(index);
        let mut ins: SmallVec<[&Array; 4]> = SmallVec::new();
        ins.push(self);
        ins.extend(index_arrays(&items));
        Array::sticky("index", Location::caller(), &ins, || {
            index_with(self, &items)
        })
    }

    /// A functional update of this array at `index` (JAX's `x.at[...]`): finish it with
    /// [`At::set`] or [`At::add`]. It takes the array by value, so that
    /// `let x = x.at(..).set(&v);` leaves the update the only owner of `x`'s buffer and the
    /// write happens in place; `x.clone().at(..)` keeps `x`.
    ///
    /// ```
    /// use nuvec_core::{Array, Ellipsis};
    ///
    /// let x = Array::from_slice(&[0.0f32; 6], [2, 3]);
    /// let x = x.at((.., 1..)).set(1.0);
    /// let x = x.at((Ellipsis, 0)).add(&Array::from_slice(&[5.0f32, 6.0], [2]));
    /// assert_eq!(x.to_vec::<f32>().unwrap(), [5.0, 1.0, 1.0, 6.0, 1.0, 1.0]);
    /// ```
    pub fn at(self, index: impl ArrayIndex) -> At {
        At {
            base: self,
            items: items_of(index),
        }
    }

    /// A copy of this array with the region selected by `index` replaced by `update` (MLX's
    /// `slice_update`, `a[index] = update`): the borrowing form of [`Array::at`] with
    /// [`At::set`], which writes in place only when the base is not shared.
    #[track_caller]
    pub fn slice_update(&self, update: &Array, index: impl ArrayIndex) -> Array {
        let items = items_of(index);
        let mut ins: SmallVec<[&Array; 4]> = SmallVec::from_slice(&[self, update]);
        ins.extend(index_arrays(&items));
        Array::sticky("slice_update", Location::caller(), &ins, || {
            update_at(self, &items, ScalarOrArray::Array(update.clone()), false)
        })
    }
}

/// A pending functional update made by [`Array::at`].
#[must_use = "an update does nothing until `set` or `add` is called"]
#[derive(Debug, Clone)]
pub struct At {
    base: Array,
    items: SmallVec<[IndexItem; 4]>,
}

#[allow(clippy::should_implement_trait)] // `add` is MLX's name; `At` is not a number
impl At {
    /// The array with the indexed region replaced by `value` (broadcast to the region and
    /// cast to the array's dtype; a weak Rust number is allowed), as MLX's `x[...] = v`.
    /// With repeated index-array positions, one of the values wins.
    #[track_caller]
    pub fn set(self, value: impl Into<ScalarOrArray>) -> Array {
        self.apply("at.set", Location::caller(), value.into(), false)
    }

    /// The array with `value` added to the indexed region; repeated index-array positions
    /// accumulate (MLX's `x.at[...].add(v)`).
    #[track_caller]
    pub fn add(self, value: impl Into<ScalarOrArray>) -> Array {
        self.apply("at.add", Location::caller(), value.into(), true)
    }

    fn apply(
        self,
        op: &'static str,
        at: &'static Location<'static>,
        value: ScalarOrArray,
        add: bool,
    ) -> Array {
        let At { base, items } = self;
        let failed = std::iter::once(&base)
            .chain(index_arrays(&items))
            .chain(match &value {
                ScalarOrArray::Array(v) => Some(v),
                ScalarOrArray::Scalar(_) => None,
            })
            .find_map(Array::slot_error);
        if let Some(error) = failed {
            return Array::failed(error, base.dtype(), base.dims());
        }
        match update_at(&base, &items, value, add) {
            Ok(out) => out,
            Err(e) => Array::failed(e.locate(op, at), base.dtype(), base.dims()),
        }
    }
}

/// The common update of index arrays on adjacent axes `k..k + m` (an embedding gradient's
/// `h.at(&ids).add(&g)`), every other axis whole: the indexed axes merge into one (a view of
/// a contiguous base) and the update is one scatter along it, with the indices broadcast to
/// the update. `None` when the index has another form.
fn update_along_axis(
    base: &Array,
    r: &Resolved,
    value: &Array,
    add: bool,
) -> Result<Option<Array>> {
    let mut adv_dims = Vec::new();
    for (slot, arg) in r.slots.iter().zip(&r.args) {
        match slot {
            Slot::New => return Ok(None),
            Slot::Dim(d) => {
                let n = base.dims()[*d] as isize;
                let whole = arg.step == 1
                    && arg.start.is_none_or(|s| s == 0 || s <= -n)
                    && arg.stop.is_none_or(|e| e >= n);
                if !whole {
                    return Ok(None);
                }
            }
            Slot::Adv(d, ..) => adv_dims.push(*d),
        }
    }
    let k = adv_dims[0];
    if adv_dims.iter().enumerate().any(|(i, &d)| d != k + i) {
        return Ok(None);
    }
    let adv = advanced_indices(base, r)?;
    let (linear, bshape, total) = linear_index(base, &adv)?;
    let dims = base.dims();
    let (pre, post) = (&dims[..k], &dims[k + adv_dims.len()..]);
    // The base with the indexed axes merged, and the update in the indexed shape.
    let mut merged: Shape = pre.into();
    merged.push(total);
    merged.extend_from_slice(post);
    let flat_base = imp::reshape_to(base, merged)?;
    let mut region: Shape = pre.into();
    region.extend_from_slice(&bshape);
    region.extend_from_slice(post);
    let m: usize = bshape.iter().product();
    let mut along: Shape = pre.into();
    along.push(m);
    along.extend_from_slice(post);
    let update = imp::broadcast_to(&imp::astype(value, base.dtype())?, &region)?;
    let update = imp::reshape_to(&imp::contiguous(&update)?, along.clone())?;
    let mut index_shape: Shape = Shape::from_elem(1, pre.len());
    index_shape.push(m);
    index_shape.extend(std::iter::repeat_n(1, post.len()));
    let indices = imp::broadcast_to(&imp::reshape_to(&linear, index_shape)?, &along)?;
    let out = if add {
        crate::indexing::scatter_add_along_axis(&flat_base, &indices, &update, k as isize)?
    } else {
        crate::indexing::put_along_axis(&flat_base, &indices, &update, k as isize)?
    };
    Ok(Some(imp::reshape_to(&out, dims.into())?))
}

/// `base` with the region at `items` replaced by (or, with `add`, increased by) `value`.
fn update_at(base: &Array, items: &[IndexItem], value: ScalarOrArray, add: bool) -> Result<Array> {
    let r = resolve(items, base.dims())?;
    let value = value.into_array(Some(base.dtype()))?;
    if !r.has_arrays() {
        // A basic region: `slice_update` (in place when `base` is not shared).
        let region = imp::slice(base, &r.args)?;
        let indexed = basic_shape(&r, region.dims());
        let update = imp::broadcast_to(&imp::astype(&value, base.dtype())?, &indexed)?;
        let update = imp::reshape_to(&update, region.dims().into())?;
        let update = if add {
            imp::add(&region, &update)?
        } else {
            update
        };
        return imp::slice_update(base, &update, &r.args);
    }
    if let Some(out) = update_along_axis(base, &r, &value, add)? {
        return Ok(out);
    }
    // Index arrays on axes that are not adjacent, mixed with partial slices or new axes:
    // scatter into the flattened array at each element's position, found by indexing the
    // positions of `base` the same way.
    let positions_dtype = if base.size() <= i32::MAX as usize {
        DType::I32
    } else {
        DType::I64
    };
    let positions = imp::reshape_to(
        &imp::arange(0.0, base.size() as f64, 1.0, positions_dtype)?,
        base.dims().into(),
    )?;
    let selected = index_with(&positions, items)?;
    let update = imp::broadcast_to(&imp::astype(&value, base.dtype())?, selected.dims())?;
    let flat = imp::reshape_to(base, Shape::from_elem(base.size(), 1))?;
    let n = selected.size();
    let positions = imp::reshape_to(&selected, Shape::from_elem(n, 1))?;
    let update = imp::reshape_to(&imp::contiguous(&update)?, Shape::from_elem(n, 1))?;
    let out = if add {
        crate::indexing::scatter_add_along_axis(&flat, &positions, &update, 0)?
    } else {
        crate::indexing::put_along_axis(&flat, &positions, &update, 0)?
    };
    imp::reshape_to(&out, base.dims().into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arange(shape: &[usize]) -> Array {
        let n: usize = shape.iter().product();
        let data: Vec<i32> = (0..n as i32).collect();
        Array::from_slice(&data, shape)
    }

    #[test]
    fn basic_forms() {
        let x = arange(&[4, 5, 6]);
        assert_eq!(x.i(0).shape(), [5, 6]);
        assert_eq!(x.i(-1).to_vec::<i32>().unwrap()[0], 90);
        assert_eq!(x.i((.., 1..3)).shape(), [4, 2, 6]);
        assert_eq!(x.i((Ellipsis, -1)).shape(), [4, 5]);
        assert_eq!(x.i((.., NewAxis, ..)).shape(), [4, 1, 5, 6]);
        assert_eq!(x.i((..).step(-1)).to_vec::<i32>().unwrap()[0], 90);
        assert_eq!(x.i((1.., ..=2, (0..6).step(2))).shape(), [3, 3, 3]);
        assert_eq!(x.i(1usize..3usize).shape(), [2, 5, 6]);
        // Views: no gather, no copy.
        assert!(matches!(
            x.i((.., 1..3)).primitive_t(),
            Some(crate::graph::Primitive::Slice { .. })
        ));
    }

    #[test]
    fn invalid_indices() {
        let x = arange(&[4, 5]);
        for bad in [
            x.i(4),
            x.i(-5),
            x.i((Ellipsis, 0, Ellipsis)),
            x.i((0, 0, 0)),
            x.i((0..2).step(0)),
        ] {
            let e = bad.check().unwrap_err();
            assert!(
                matches!(e.kind(), Some(crate::OpErrorKind::InvalidIndex(_))),
                "{e}"
            );
        }
    }

    #[test]
    fn advanced_placement() {
        let x = arange(&[2, 3, 4]);
        let i2 = Array::from_slice(&[0i32, 1], [2]);
        let j2 = Array::from_slice(&[1i32, 2], [2]);
        assert_eq!(x.i((&i2, .., &j2)).shape(), [2, 3]);
        assert_eq!(x.i((.., &i2, &j2)).shape(), [2, 2]);
        assert_eq!(x.i((&i2, 1)).shape(), [2, 4]);
        assert_eq!(x.i((1, .., &i2)).shape(), [2, 3]);
        assert_eq!(x.i((NewAxis, &i2)).shape(), [1, 2, 3, 4]);
        assert_eq!(x.i((&i2, NewAxis)).shape(), [2, 1, 3, 4]);
        assert_eq!(x.i((Ellipsis, &i2)).shape(), [2, 3, 2]);
        assert_eq!(x.i((.., &i2, &j2)).to_vec::<i32>().unwrap(), [1, 6, 13, 18]);
    }

    #[test]
    fn updates() {
        let x = Array::from_slice(&[0i32; 6], [2, 3]);
        let y = x.clone().at((.., 1..)).set(7);
        assert_eq!(y.to_vec::<i32>().unwrap(), [0, 7, 7, 0, 7, 7]);
        let ids = Array::from_slice(&[0i32, 0, 1], [3]);
        let z = x.at(&ids).add(1);
        assert_eq!(z.to_vec::<i32>().unwrap(), [2, 2, 2, 1, 1, 1]);
    }
}
