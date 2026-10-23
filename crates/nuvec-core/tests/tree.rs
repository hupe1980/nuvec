//! Trees of arrays: leaf order and paths, views, `unflatten`, `load_flat`,
//! `TreeDef`, `Static`, leaf kinds, `Param` (hand-written impls: core does not use the
//! derive).

use std::collections::{BTreeMap, HashMap};
use std::hash::{DefaultHasher, Hasher};

use nuvec_core::tree::{
    LeafKind, Leaves, LoadMode, Param, Static, Tree, Visit, VisitMut, Visitor, VisitorMut,
};
use nuvec_core::{Array, DType, array, ops};

fn paths<T: Tree>(t: &T) -> Vec<String> {
    t.paths()
}

fn vals(a: &Array) -> Vec<f32> {
    a.astype(DType::F32).to_vec::<f32>().unwrap()
}

/// The leaves' paths of any `Leaves` value (views included).
fn view_paths<T: Leaves + ?Sized>(t: &T) -> Vec<String> {
    struct P(Vec<String>);
    impl Visitor for P {
        fn leaf(&mut self, path: &str, _: &Array, _: LeafKind, _: bool) {
            self.0.push(path.to_owned());
        }
    }
    let mut p = P(Vec::new());
    t.visit(&mut Visit::new(&mut p));
    p.0
}

fn a(v: f32) -> Array {
    Array::from(v)
}

#[test]
fn leaf_order_and_paths() {
    assert_eq!(paths(&a(1.0)), [""]);
    assert_eq!(paths(&Some(a(1.0))), [""]);
    assert_eq!(paths(&Box::new(a(1.0))), [""]);
    assert_eq!(paths(&vec![a(1.0), a(2.0)]), ["0", "1"]);
    assert_eq!(paths(&[a(1.0), a(2.0), a(3.0)]), ["0", "1", "2"]);
    assert!(paths(&()).is_empty());
    assert_eq!(paths(&(a(1.0),)), ["0"]);
    let eight = (
        a(0.0),
        a(1.0),
        a(2.0),
        a(3.0),
        a(4.0),
        a(5.0),
        a(6.0),
        a(7.0),
    );
    assert_eq!(paths(&eight), ["0", "1", "2", "3", "4", "5", "6", "7"]);
    let mut b = BTreeMap::new();
    b.insert("w".to_owned(), a(1.0));
    b.insert("b".to_owned(), a(2.0));
    b.insert("a".to_owned(), a(3.0));
    assert_eq!(paths(&b), ["a", "b", "w"]);
    let h: HashMap<String, Array> = b.clone().into_iter().collect();
    assert_eq!(
        paths(&h),
        paths(&b),
        "HashMap is visited in sorted key order"
    );
    let hv: Vec<f32> = h.leaves().iter().map(|x| vals(x)[0]).collect();
    let bv: Vec<f32> = b.leaves().iter().map(|x| vals(x)[0]).collect();
    assert_eq!(hv, bv);
    // MLX's `tree_flatten` of {"layers": [{"w": .., "b": ..}, None, {"w": ..}], "out": (x, y)}:
    // ['layers.0.b', 'layers.0.w', 'layers.2.w', 'out.0', 'out.1'].
    let mut l0 = BTreeMap::new();
    l0.insert("w".to_owned(), a(1.0));
    l0.insert("b".to_owned(), a(2.0));
    let mut l2 = BTreeMap::new();
    l2.insert("w".to_owned(), a(3.0));
    let nested = (vec![Some(l0), None, Some(l2)], (a(4.0), a(5.0)));
    assert_eq!(paths(&nested), ["0.0.b", "0.0.w", "0.2.w", "1.0", "1.1"]);
}

#[test]
fn unflatten_round_trip_and_wrong_count() {
    let t = (vec![a(1.0), a(2.0)], Some(a(3.0)));
    let doubled = t.unflatten(t.leaves().iter().map(|x| x * 2.0)).unwrap();
    assert_eq!(vals(&doubled.0[1]), [4.0]);
    assert_eq!(vals(doubled.1.as_ref().unwrap()), [6.0]);
    let err = t.unflatten([a(1.0)]).unwrap_err().to_string();
    assert!(err.contains('1') && err.contains('3'), "{err}");
    let mapped = t.map(|p, x| if p == "1" { x + 10.0 } else { x.clone() });
    assert_eq!(vals(mapped.1.as_ref().unwrap()), [13.0]);
}

#[test]
fn empty_trees_have_no_leaves() {
    assert!(().is_empty());
    assert!(None::<Array>.is_empty());
    assert!(Vec::<Array>::new().is_empty());
    assert!(BTreeMap::<String, Array>::new().is_empty());
    assert_eq!(().unflatten(Vec::new()).unwrap(), ());
}

#[test]
fn borrowed_views_visit_the_same_leaves() {
    let loss = a(1.0);
    let grads = vec![a(2.0), a(3.0)];
    let owned = (loss.clone(), grads.clone());
    assert_eq!(view_paths(&(&loss, &grads)), paths(&owned));
    assert_eq!(view_paths(&&owned), paths(&owned));
    let slice: &[Array] = &grads;
    assert_eq!(view_paths(slice), ["0", "1"]);
}

/// A hand-written enum with a leaf-less variant.
#[derive(Clone)]
enum Choice {
    Weight(Array),
    Nothing,
}

impl Leaves for Choice {
    fn visit(&self, v: &mut Visit<'_>) {
        match self {
            Choice::Weight(w) => {
                v.variant(0);
                v.child("", w);
            }
            Choice::Nothing => v.variant(1),
        }
    }
}

impl Tree for Choice {
    fn visit_mut(&mut self, v: &mut VisitMut<'_>) {
        if let Choice::Weight(w) = self {
            v.child("", w);
        }
    }
}

/// A hand-written layer with a parameter, a buffer, a mode flag and a static field.
#[derive(Clone)]
struct Layer {
    weight: Param,
    running: Array,
    training: bool,
    eps: f32,
    choice: Choice,
}

impl Layer {
    fn new() -> Self {
        Self {
            weight: Param::new(array![1.0, 2.0]),
            running: array![0.0],
            training: true,
            eps: 1e-8,
            choice: Choice::Nothing,
        }
    }
}

impl Leaves for Layer {
    fn visit(&self, v: &mut Visit<'_>) {
        v.child("weight", &self.weight);
        v.buffer("running", &self.running);
        v.mode(self.training);
        v.static_field("eps", &self.eps);
        v.child("choice", &self.choice);
    }
}

impl Tree for Layer {
    fn visit_mut(&mut self, v: &mut VisitMut<'_>) {
        v.child("weight", &mut self.weight);
        v.buffer("running", &mut self.running);
        v.training(&mut self.training);
        v.child("choice", &mut self.choice);
    }
}

#[test]
fn tree_def_captures_structure() {
    let t = (a(1.0), a(2.0));
    assert_eq!(t.def(), (a(5.0), a(6.0)).def(), "values do not matter");
    assert_ne!(t.def(), vec![a(1.0), a(2.0)].def(), "tuple vs Vec");
    let mut m1 = BTreeMap::new();
    m1.insert("x".to_owned(), a(1.0));
    let mut m2 = BTreeMap::new();
    m2.insert("y".to_owned(), a(1.0));
    assert_ne!(m1.def(), m2.def(), "paths");
    assert_ne!(
        Some(()).def(),
        None::<()>.def(),
        "leaf-less Option variants"
    );

    let base = Layer::new();
    assert_eq!(base.def(), Layer::new().def());
    let mut frozen = Layer::new();
    frozen.weight.set_frozen(true);
    assert_ne!(base.def(), frozen.def(), "frozen vs trainable");
    let mut eval = Layer::new();
    eval.training = false;
    assert_ne!(base.def(), eval.def(), "mode flag");
    let mut eps = Layer::new();
    eps.eps = 1e-7;
    assert_ne!(base.def(), eps.def(), "static field");
    let mut chosen = Layer::new();
    chosen.choice = Choice::Weight(a(1.0));
    assert_ne!(base.def(), chosen.def(), "variant");
    // A leaf-less variant differs from another leaf-less structure only by its index.
    let mut zero = Layer::new();
    zero.eps = 0.0;
    let mut neg = Layer::new();
    neg.eps = -0.0;
    assert_ne!(zero.def(), neg.def(), "floats hash by bits");
    assert_eq!(base.def().len(), 2);
    assert!(base.def().difference(&eps.def()).is_some());
    assert!(base.def().difference(&base.def()).is_none());
}

fn hash<T: Static>(v: &T) -> u64 {
    let mut h = DefaultHasher::new();
    v.hash_static(&mut h);
    h.finish()
}

#[test]
fn static_impls_hash_by_value() {
    assert_eq!(hash(&3u32), hash(&3u32));
    assert_ne!(hash(&3u32), hash(&4u32));
    assert_eq!(hash(&true), hash(&true));
    assert_eq!(hash(&"a".to_owned()), hash(&"a".to_owned()));
    assert_ne!(hash(&1.0f32), hash(&1.0001f32));
    assert_ne!(hash(&0.0f64), hash(&-0.0f64));
    assert_eq!(hash(&Some(2i64)), hash(&Some(2i64)));
    assert_ne!(hash(&Some(2i64)), hash(&None::<i64>));
    assert_eq!(hash(&vec![1u8, 2]), hash(&vec![1u8, 2]));
    assert_ne!(hash(&vec![1u8, 2]), hash(&vec![2u8, 1]));
    assert_eq!(hash(&(1u8, 2.0f32)), hash(&(1u8, 2.0f32)));
    assert_eq!(hash(&DType::F16), hash(&DType::F16));
    assert_ne!(hash(&DType::F16), hash(&DType::BF16));
}

#[test]
fn load_flat_reports_every_problem() {
    let mut t: BTreeMap<String, Array> = BTreeMap::new();
    t.insert("a".to_owned(), array![1.0, 2.0]);
    t.insert("b".to_owned(), array![3.0]);
    t.insert("c".to_owned(), array![4.0]);
    let before = t.clone();
    let named = vec![
        ("a".to_owned(), array![1.0, 2.0, 3.0]), // shape
        ("c".to_owned(), array![9.0]),
        ("z".to_owned(), array![0.0]), // unknown
    ]; // "b" missing
    let err = t
        .load_flat(named.clone(), LoadMode::Strict)
        .unwrap_err()
        .to_string();
    assert!(err.contains("3 problems"), "{err}");
    let missing = err.find("missing: b").expect("missing");
    let unknown = err.find("unknown: z").expect("unknown");
    let shape = err.find("shape: a [3], expected [2]").expect("shape");
    assert!(missing < unknown && unknown < shape, "{err}");
    assert_eq!(vals(&t["c"]), vals(&before["c"]), "unchanged on error");
    // Partial: assigns matches, ignores unknown names, still rejects shape mismatches.
    let err = t
        .load_flat(named, LoadMode::Partial)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("1 problem") && err.contains("shape: a"),
        "{err}"
    );
    t.load_flat(
        vec![("c".to_owned(), array![9.0]), ("z".to_owned(), array![0.0])],
        LoadMode::Partial,
    )
    .unwrap();
    assert_eq!(vals(&t["c"]), [9.0]);
    // Loaded dtypes are kept.
    t.load_flat(
        vec![("b".to_owned(), Array::from_slice(&[7i32], [1]))],
        LoadMode::Partial,
    )
    .unwrap();
    assert_eq!(t["b"].dtype(), DType::I32);
    // Duplicate names are an error.
    let dup = vec![("b".to_owned(), array![1.0]), ("b".to_owned(), array![2.0])];
    assert!(t.load_flat(dup, LoadMode::Partial).is_err());
    // A tree with a duplicate path: a map key containing `.` colliding with a nested one.
    let inner = |k: &str| -> BTreeMap<String, Array> { [(k.to_owned(), a(1.0))].into() };
    let mut clash: BTreeMap<String, BTreeMap<String, Array>> =
        [("a".to_owned(), inner("b")), ("a.b".to_owned(), inner(""))].into();
    assert_eq!(paths(&clash), ["a.b", "a.b"]);
    let err = clash.load_flat(Vec::new(), LoadMode::Partial).unwrap_err();
    assert!(err.to_string().contains("twice"), "{err}");
}

#[test]
fn leaf_kinds_and_params() {
    struct Kinds(Vec<(String, LeafKind, bool)>);
    impl Visitor for Kinds {
        fn leaf(&mut self, path: &str, _: &Array, kind: LeafKind, trainable: bool) {
            self.0.push((path.to_owned(), kind, trainable));
        }
    }
    let mut layer = Layer::new();
    let mut k = Kinds(Vec::new());
    layer.visit(&mut Visit::new(&mut k));
    assert_eq!(
        k.0,
        [
            ("weight".to_owned(), LeafKind::Param, true),
            ("running".to_owned(), LeafKind::Buffer, false),
        ]
    );
    layer.weight.set_frozen(true);
    let mapped = layer.map(|_, x| x.clone());
    assert!(mapped.weight.is_frozen(), "map keeps the flag");
    assert!(layer.clone().weight.is_frozen(), "clone keeps the flag");
    assert!(!Param::new(a(1.0)).is_frozen());
    assert!(Param::frozen(a(1.0)).is_frozen());
    assert_eq!((a(1.0), Param::new(a(2.0))).def().leaves().count(), 2);

    // VisitMut: mode flags and children as `&mut dyn Any`.
    struct Modes(Vec<String>, usize);
    impl VisitorMut for Modes {
        fn leaf(&mut self, _: &str, _: &mut Array, _: LeafKind, _: Option<&mut bool>) {}
        fn child(&mut self, path: &str, child: &mut dyn std::any::Any) -> bool {
            if child.downcast_mut::<Choice>().is_some() {
                self.0.push(path.to_owned());
            }
            true
        }
        fn training(&mut self, _: &str, flag: &mut bool) {
            *flag = false;
            self.1 += 1;
        }
    }
    let mut m = Modes(Vec::new(), 0);
    let mut two = vec![Layer::new(), Layer::new()];
    two.visit_mut(&mut VisitMut::new(&mut m));
    assert_eq!(m.0, ["0.choice", "1.choice"]);
    assert_eq!(m.1, 2);
    assert!(!two[1].training);
}

#[test]
fn params_read_as_arrays() {
    let p = Param::new(array![1.0, 2.0]);
    let x = array![3.0, 4.0];
    assert_eq!(vals(&(&x * &p)), [3.0, 8.0]);
    assert_eq!(vals(&(&p * &x)), [3.0, 8.0]);
    assert_eq!(vals(&(&p + 1.0)), [2.0, 3.0]);
    assert_eq!(vals(&(&p - &p)), [0.0, 0.0]);
    assert_eq!(vals(&(x.clone() / &p)), [3.0, 2.0]);
    assert_eq!(vals(&-&p), [-1.0, -2.0]);
    let m = array![[1.0, 0.0], [0.0, 1.0]];
    let pm = Param::new(m.clone());
    assert_eq!(vals(&x.reshape([1, 2]).matmul(&pm)), [3.0, 4.0]);
    assert_eq!(vals(&p.sum(..)), [3.0]);
    assert_eq!(vals(&ops::add(&p, &x)), [4.0, 6.0]);
    assert_eq!(vals(&x.add(&p)), [4.0, 6.0]);
    assert_eq!(p.shape(), [2]);
}
