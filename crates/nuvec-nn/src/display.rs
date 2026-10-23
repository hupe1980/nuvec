//! Introspection: [`Summary`] (one row per leaf) and [`ModuleDisplay`] (the nested modules
//! with their parameter counts), built from a tree's paths and child boundaries.

use std::fmt;

use nuvec::tree::{LeafKind, Leaves, Visit, Visitor};
use nuvec::{Array, DType};

/// One leaf of a [`Summary`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SummaryRow {
    /// The leaf's path (`layers.0.weight`).
    pub path: String,
    /// Its shape.
    pub shape: Vec<i32>,
    /// Its dtype.
    pub dtype: DType,
    /// What it is: a parameter, a buffer or a plain array.
    pub kind: LeafKind,
    /// Whether it is trained.
    pub trainable: bool,
    /// Its size in bytes.
    pub bytes: usize,
}

/// A table of a model's leaves (path, shape, dtype, trainable, bytes) with totals,
/// printed as aligned columns ([`crate::Module::summary`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Summary {
    /// The leaves, in order.
    pub rows: Vec<SummaryRow>,
}

impl Summary {
    pub(crate) fn of<T: Leaves + ?Sized>(tree: &T) -> Self {
        struct Rows(Vec<SummaryRow>);
        impl Visitor for Rows {
            fn leaf(&mut self, path: &str, a: &Array, kind: LeafKind, trainable: bool) {
                self.0.push(SummaryRow {
                    path: path.to_owned(),
                    shape: a.shape().to_vec(),
                    dtype: a.dtype(),
                    kind,
                    trainable,
                    bytes: a.nbytes(),
                });
            }
        }
        let mut rows = Rows(Vec::new());
        tree.visit(&mut Visit::new(&mut rows));
        Self { rows: rows.0 }
    }

    /// Elements over every leaf.
    pub fn num_params(&self) -> usize {
        self.rows.iter().map(|r| elements(&r.shape)).sum()
    }

    /// Elements over the trainable leaves.
    pub fn num_trainable(&self) -> usize {
        self.rows
            .iter()
            .filter(|r| r.trainable)
            .map(|r| elements(&r.shape))
            .sum()
    }

    /// Bytes over every leaf.
    pub fn bytes(&self) -> usize {
        self.rows.iter().map(|r| r.bytes).sum()
    }
}

fn elements(shape: &[i32]) -> usize {
    shape.iter().map(|&d| d.max(0) as usize).product()
}

/// `n` with thousands separators: `25,818`.
pub(crate) fn grouped(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

impl fmt::Display for Summary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let shape = |s: &[i32]| {
            let dims: Vec<String> = s.iter().map(i32::to_string).collect();
            format!("[{}]", dims.join(", "))
        };
        let header = ["path", "shape", "dtype", "trainable", "bytes"];
        let cells: Vec<[String; 5]> = self
            .rows
            .iter()
            .map(|r| {
                [
                    r.path.clone(),
                    shape(&r.shape),
                    r.dtype.to_string(),
                    if r.trainable { "yes" } else { "no" }.to_owned(),
                    grouped(r.bytes),
                ]
            })
            .collect();
        let mut widths = header.map(str::len);
        for row in &cells {
            for (w, c) in widths.iter_mut().zip(row) {
                *w = (*w).max(c.chars().count());
            }
        }
        let line = |f: &mut fmt::Formatter<'_>, row: [&str; 5]| -> fmt::Result {
            let [p, s, d, t, b] = row;
            writeln!(
                f,
                "{p:<w0$}  {s:<w1$}  {d:<w2$}  {t:<w3$}  {b:>w4$}",
                w0 = widths[0],
                w1 = widths[1],
                w2 = widths[2],
                w3 = widths[3],
                w4 = widths[4]
            )
        };
        line(f, header)?;
        for row in &cells {
            line(f, [&row[0], &row[1], &row[2], &row[3], &row[4]])?;
        }
        write!(
            f,
            "{} params ({} trainable), {} bytes",
            grouped(self.num_params()),
            grouped(self.num_trainable()),
            grouped(self.bytes())
        )
    }
}

/// A model's nested modules with their parameter counts ([`crate::Module::display`]):
///
/// ```text
/// Mlp (25,818 params, 25,818 trainable)
///   layers.0: Linear (25,120)
///   layers.1: Linear (528)
///   layers.2: Linear (170)
/// ```
pub struct ModuleDisplay<'a, M: ?Sized> {
    module: &'a M,
}

impl<'a, M: ?Sized> ModuleDisplay<'a, M> {
    pub(crate) fn new(module: &'a M) -> Self {
        Self { module }
    }
}

/// One node of the module tree.
struct Node {
    path: String,
    type_name: &'static str,
    params: usize,
    trainable: usize,
    children: Vec<usize>,
    /// A container or leaf type (not printed itself; its modules are).
    hidden: bool,
}

/// The type's name without its module path and generic arguments.
fn short(type_name: &'static str) -> &'static str {
    let base = type_name.split('<').next().unwrap_or(type_name);
    base.rsplit("::").next().unwrap_or(base)
}

/// Containers and leaves, whose own node is not a module.
fn hidden(type_name: &str) -> bool {
    let base = type_name.split('<').next().unwrap_or(type_name);
    base.starts_with('(')
        || base.starts_with('[')
        || base.starts_with('&')
        || [
            "alloc::vec::Vec",
            "core::option::Option",
            "alloc::boxed::Box",
            "alloc::collections::btree::map::BTreeMap",
            "std::collections::hash::map::HashMap",
            "nuvec_core::array::Array",
            "nuvec_core::tree::Param",
            "core::marker::PhantomData",
        ]
        .contains(&base)
}

struct Builder {
    nodes: Vec<Node>,
    stack: Vec<usize>,
}

impl Visitor for Builder {
    fn leaf(&mut self, _: &str, a: &Array, _: LeafKind, trainable: bool) {
        let n = a.size();
        for &i in &self.stack {
            self.nodes[i].params += n;
            if trainable {
                self.nodes[i].trainable += n;
            }
        }
    }

    fn enter(&mut self, path: &str, name: &str, type_name: &'static str) {
        let parent = *self.stack.last().expect("the root is open");
        if name.is_empty() && !hidden(type_name) {
            // A transparent wrapper (`Option`, an enum's newtype variant): the inner type
            // names the module at this path.
            let mut at = parent;
            while self.nodes[at].hidden && self.nodes[at].path == path {
                match self.stack.iter().rev().nth(1) {
                    Some(&up) if self.nodes[up].path == path => at = up,
                    _ => break,
                }
            }
            if self.nodes[at].path == path {
                self.nodes[at].type_name = type_name;
                self.nodes[at].hidden = false;
            }
        }
        self.nodes.push(Node {
            path: path.to_owned(),
            type_name,
            params: 0,
            trainable: 0,
            children: Vec::new(),
            hidden: name.is_empty() || hidden(type_name),
        });
        let id = self.nodes.len() - 1;
        self.nodes[parent].children.push(id);
        self.stack.push(id);
    }

    fn exit(&mut self, _: &str) {
        self.stack.pop();
    }
}

impl<M: Leaves> fmt::Display for ModuleDisplay<'_, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut b = Builder {
            nodes: vec![Node {
                path: String::new(),
                type_name: std::any::type_name::<M>(),
                params: 0,
                trainable: 0,
                children: Vec::new(),
                hidden: false,
            }],
            stack: vec![0],
        };
        self.module.visit(&mut Visit::new(&mut b));
        let root = &b.nodes[0];
        write!(
            f,
            "{} ({} params, {} trainable)",
            short(root.type_name),
            grouped(root.params),
            grouped(root.trainable)
        )?;
        fn print(
            f: &mut fmt::Formatter<'_>,
            nodes: &[Node],
            at: usize,
            depth: usize,
        ) -> fmt::Result {
            for &c in &nodes[at].children {
                let node = &nodes[c];
                let mut next = depth;
                if !node.hidden && node.params > 0 {
                    let frozen = if node.trainable == 0 { ", frozen" } else { "" };
                    write!(
                        f,
                        "\n{:indent$}{}: {} ({}{frozen})",
                        "",
                        node.path,
                        short(node.type_name),
                        grouped(node.params),
                        indent = 2 * depth
                    )?;
                    next = depth + 1;
                }
                print(f, nodes, c, next)?;
            }
            Ok(())
        }
        print(f, &b.nodes, 0, 1)
    }
}
