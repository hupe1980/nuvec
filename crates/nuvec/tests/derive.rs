//! `#[derive(Tree)]` and `#[derive(Static)]` behaviour.

use std::marker::PhantomData;

use nuvec::{Array, LeafKind, Leaves, Param, Static, Tree, Visit, Visitor, array};

fn kinds<T: Leaves>(t: &T) -> Vec<(String, LeafKind, bool)> {
    struct K(Vec<(String, LeafKind, bool)>);
    impl Visitor for K {
        fn leaf(&mut self, path: &str, _: &Array, kind: LeafKind, trainable: bool) {
            self.0.push((path.to_owned(), kind, trainable));
        }
    }
    let mut k = K(Vec::new());
    t.visit(&mut Visit::new(&mut k));
    k.0
}

#[derive(Clone, Static, PartialEq, Debug)]
struct Config {
    heads: i32,
    eps: f32,
    name: String,
}

#[derive(Clone, Tree)]
struct Named {
    weight: Param,
    #[tree(skip)]
    heads: i32,
    #[tree(rename = "x")]
    renamed: Array,
    #[tree(buffer)]
    running: Array,
    #[tree(skip)]
    freqs: Array,
    #[tree(skip)]
    config: Config,
    #[tree(skip, static_unchecked)]
    note: std::rc::Rc<str>,
    #[tree(mode)]
    training: bool,
}

fn named() -> Named {
    Named {
        weight: Param::new(array![1.0]),
        heads: 4,
        renamed: array![2.0],
        running: array![3.0],
        freqs: array![4.0],
        config: Config {
            heads: 2,
            eps: 1e-5,
            name: "a".into(),
        },
        note: "n".into(),
        training: true,
    }
}

#[test]
fn named_structs() {
    let n = named();
    assert_eq!(
        kinds(&n),
        [
            ("weight".to_owned(), LeafKind::Param, true),
            ("x".to_owned(), LeafKind::Array, true),
            ("running".to_owned(), LeafKind::Buffer, false),
        ]
    );
    assert_eq!(n.def(), named().def());
    let mut heads = named();
    heads.heads = 8;
    assert_ne!(n.def(), heads.def(), "a skipped field is hashed");
    let mut eps = named();
    eps.config.eps = 1e-6;
    assert_ne!(n.def(), eps.def(), "a skipped f32 inside a derived Static");
    let mut freqs = named();
    freqs.freqs = array![9.0, 9.0];
    assert_eq!(
        n.def(),
        freqs.def(),
        "a skipped Array is a computed constant"
    );
    let mut note = named();
    note.note = "other".into();
    assert_eq!(n.def(), note.def(), "static_unchecked is not hashed");
    let mut eval = named();
    eval.training = false;
    assert_ne!(n.def(), eval.def(), "mode flags");
    let rebuilt = n
        .unflatten([array![5.0], array![6.0], array![7.0]])
        .unwrap();
    assert_eq!(rebuilt.renamed.to_vec::<f32>().unwrap(), [6.0]);
    assert_eq!(
        rebuilt.freqs.to_vec::<f32>().unwrap(),
        [4.0],
        "constants kept"
    );
}

#[derive(Clone, Tree)]
struct Pair(Array, Vec<Array>);

#[derive(Clone, Tree)]
struct Unit;

#[test]
fn tuple_and_unit_structs() {
    let p = Pair(array![1.0], vec![array![2.0], array![3.0]]);
    assert_eq!(p.paths(), ["0", "1.0", "1.1"]);
    assert!(Unit.paths().is_empty());
}

#[derive(Clone, Tree)]
enum Either {
    Plain(Array),
    #[tree(rename = "b")]
    Named(Vec<Array>),
    Fields {
        w: Array,
        #[tree(rename = "bias")]
        b: Array,
    },
    #[tree(rename = "r")]
    Renamed {
        w: Array,
    },
    Two(Array, Array),
    Empty,
    AlsoEmpty,
}

#[test]
fn enums() {
    assert_eq!(Either::Plain(array![1.0]).paths(), [""]);
    assert_eq!(Either::Named(vec![array![1.0]]).paths(), ["b.0"]);
    let f = Either::Fields {
        w: array![1.0],
        b: array![2.0],
    };
    assert_eq!(f.paths(), ["w", "bias"]);
    assert_eq!(Either::Renamed { w: array![1.0] }.paths(), ["r.w"]);
    assert_eq!(Either::Two(array![1.0], array![2.0]).paths(), ["0", "1"]);
    assert!(Either::Empty.paths().is_empty());
    assert_ne!(
        Either::Empty.def(),
        Either::AlsoEmpty.def(),
        "variant index"
    );
    let back = f.unflatten([array![3.0], array![4.0]]).unwrap();
    let Either::Fields { b, .. } = back else {
        panic!("variant kept")
    };
    assert_eq!(b.to_vec::<f32>().unwrap(), [4.0]);
}

#[derive(Clone, Tree)]
struct Inner {
    w: Param,
}

#[derive(Clone, Tree)]
struct Containers {
    boxed: Option<Box<Inner>>,
    list: Vec<Option<Inner>>,
}

#[test]
fn nested_containers() {
    let c = Containers {
        boxed: Some(Box::new(Inner {
            w: Param::new(array![1.0]),
        })),
        list: vec![
            None,
            Some(Inner {
                w: Param::frozen(array![2.0]),
            }),
        ],
    };
    assert_eq!(c.paths(), ["boxed.w", "list.1.w"]);
    assert!(!kinds(&c)[1].2, "frozen");
    let none = Containers {
        boxed: None,
        list: vec![],
    };
    assert_ne!(c.def(), none.def());
}

#[derive(Clone, Tree)]
struct Generic<T, C, M> {
    item: T,
    list: Vec<T>,
    #[tree(skip)]
    config: C,
    marker: PhantomData<M>,
}

/// Not a tree, not `Static`: usable as `M` (only in `PhantomData`).
#[derive(Clone)]
struct NotATree;

#[test]
fn generics_are_bounded_where_used() {
    let g: Generic<Array, u32, NotATree> = Generic {
        item: array![1.0],
        list: vec![array![2.0]],
        config: 3,
        marker: PhantomData,
    };
    assert_eq!(g.paths(), ["item", "list.0"]);
    let mut other = g.clone();
    other.config = 4;
    assert_ne!(g.def(), other.def());
}

mod hygiene {
    // No `use` of nuvec items: generated paths are `::nuvec::__derive::…`.
    #[derive(Clone, nuvec::Tree)]
    pub(super) struct Bare {
        pub(super) a: nuvec::Array,
        #[tree(skip)]
        pub(super) n: usize,
    }

    #[derive(Clone, nuvec::Tree)]
    #[tree(crate = "nuvec_core")]
    pub(super) struct ViaCore {
        pub(super) a: nuvec::Array,
    }
}

#[test]
fn hygiene_and_crate_attribute() {
    let b = hygiene::Bare {
        a: array![1.0],
        n: 1,
    };
    assert_eq!(b.paths(), ["a"]);
    let _ = b.n;
    let c = hygiene::ViaCore { a: array![1.0] };
    assert_eq!(nuvec_core::Tree::paths(&c), ["a"]);
}
