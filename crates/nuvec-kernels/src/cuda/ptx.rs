//! PTX that nuvec generates (constitution I: kernels in Rust, no CUDA C): element-wise
//! kernels and row reductions, for sm_80 and newer. Half types load to f32, compute in f32
//! and round to nearest even on store, so an f16/bf16 add or multiply is the IEEE result.
//! Transcendentals use range reduction and polynomials, not the hardware approximations.

use std::fmt::Write;

/// An element type of the generated kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// 32-bit float.
    F32,
    /// IEEE half.
    F16,
    /// bfloat16.
    BF16,
}

impl DType {
    /// Bytes per element.
    pub fn size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
        }
    }
}

/// An element-wise operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Op {
    /// `a + b`.
    Add,
    /// `a · b`.
    Mul,
    /// `eᵃ`.
    Exp,
}

impl Op {
    /// Its operand count.
    pub fn arity(self) -> usize {
        match self {
            Self::Add | Self::Mul => 2,
            Self::Exp => 1,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Mul => "mul",
            Self::Exp => "exp",
        }
    }
}

/// The most dimensions a strided kernel indexes.
pub const MAX_DIMS: usize = 8;

/// A generated kernel: its entry name and its PTX module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Kernel {
    /// The entry point.
    pub name: String,
    /// The module's text.
    pub ptx: String,
}

/// Emits PTX with fresh virtual registers of each class.
#[derive(Default)]
struct Builder {
    body: String,
    r: u32,
    rd: u32,
    f: u32,
    h: u32,
    p: u32,
    label: u32,
}

impl Builder {
    fn r(&mut self) -> String {
        self.r += 1;
        format!("%r{}", self.r)
    }
    fn rd(&mut self) -> String {
        self.rd += 1;
        format!("%rd{}", self.rd)
    }
    fn f(&mut self) -> String {
        self.f += 1;
        format!("%f{}", self.f)
    }
    fn h(&mut self) -> String {
        self.h += 1;
        format!("%h{}", self.h)
    }
    fn p(&mut self) -> String {
        self.p += 1;
        format!("%p{}", self.p)
    }
    fn label(&mut self) -> String {
        self.label += 1;
        format!("L{}", self.label)
    }
    fn i(&mut self, line: impl AsRef<str>) {
        let _ = writeln!(self.body, "\t{};", line.as_ref());
    }
    fn at(&mut self, label: &str) {
        let _ = writeln!(self.body, "{label}:");
    }

    /// The module around one entry with these parameters.
    fn finish(self, name: &str, params: &[String], shared: Option<&str>) -> String {
        let mut s = String::from(".version 8.0\n.target sm_80\n.address_size 64\n\n");
        if let Some(shared) = shared {
            let _ = writeln!(s, "{shared}");
        }
        let _ = writeln!(
            s,
            ".visible .entry {name}(\n\t{}\n)\n{{",
            params.join(",\n\t")
        );
        for (class, ty, n) in [
            ("p", ".pred", self.p),
            ("r", ".b32", self.r),
            ("rd", ".b64", self.rd),
            ("f", ".f32", self.f),
            ("h", ".b16", self.h),
        ] {
            if n > 0 {
                let _ = writeln!(s, "\t.reg {ty} %{class}<{}>;", n + 1);
            }
        }
        s.push_str(&self.body);
        s.push_str("}\n");
        s
    }

    /// The global thread index (`ctaid.x · ntid.x + tid.x`), as u32.
    fn thread_index(&mut self) -> String {
        let (a, b, c, i) = (self.r(), self.r(), self.r(), self.r());
        self.i(format!("mov.u32 {a}, %ctaid.x"));
        self.i(format!("mov.u32 {b}, %ntid.x"));
        self.i(format!("mov.u32 {c}, %tid.x"));
        self.i(format!("mad.lo.u32 {i}, {a}, {b}, {c}"));
        i
    }

    /// An f32 register loaded from `addr` (a 64-bit global address) of `dtype`.
    fn load(&mut self, dtype: DType, addr: &str) -> String {
        let f = self.f();
        match dtype {
            DType::F32 => self.i(format!("ld.global.f32 {f}, [{addr}]")),
            DType::F16 => {
                let h = self.h();
                self.i(format!("ld.global.b16 {h}, [{addr}]"));
                self.i(format!("cvt.f32.f16 {f}, {h}"));
            }
            DType::BF16 => {
                // bf16 → f32 is exact: the bits move to the high half.
                let (h, r) = (self.h(), self.r());
                self.i(format!("ld.global.b16 {h}, [{addr}]"));
                self.i(format!("cvt.u32.u16 {r}, {h}"));
                self.i(format!("shl.b32 {r}, {r}, 16"));
                self.i(format!("mov.b32 {f}, {r}"));
            }
        }
        f
    }

    /// Stores f32 `f` at `addr` as `dtype`, rounded to nearest even.
    fn store(&mut self, dtype: DType, addr: &str, f: &str) {
        match dtype {
            DType::F32 => self.i(format!("st.global.f32 [{addr}], {f}")),
            DType::F16 => {
                let h = self.h();
                self.i(format!("cvt.rn.f16.f32 {h}, {f}"));
                self.i(format!("st.global.b16 [{addr}], {h}"));
            }
            DType::BF16 => {
                let h = self.h();
                self.i(format!("cvt.rn.bf16.f32 {h}, {f}"));
                self.i(format!("st.global.b16 [{addr}], {h}"));
            }
        }
    }

    /// `base + offset · size` for a 32-bit signed element offset.
    fn address(&mut self, base: &str, offset: &str, size: usize) -> String {
        let (wide, addr) = (self.rd(), self.rd());
        self.i(format!("mul.wide.s32 {wide}, {offset}, {size}"));
        self.i(format!("add.s64 {addr}, {base}, {wide}"));
        addr
    }

    /// `eˣ` of f32 `x`: x = n·ln2 + r (Cody–Waite, two-part ln 2), eʳ by its degree-7
    /// Taylor polynomial (|r| ≤ ln2/2: truncation 5·10⁻⁹ relative), then · 2ⁿ in two steps
    /// so that subnormal results and overflow to infinity come out right. NaN propagates.
    fn exp(&mut self, x: &str) -> String {
        let c = |v: f32| format!("0f{:08X}", v.to_bits());
        let (xc, t, n, r, p) = (self.f(), self.f(), self.f(), self.f(), self.f());
        self.i(format!("max.f32 {xc}, {x}, {}", c(-104.0)));
        self.i(format!("min.f32 {xc}, {xc}, {}", c(89.0)));
        self.i(format!(
            "mul.f32 {t}, {xc}, {}",
            c(std::f32::consts::LOG2_E)
        ));
        self.i(format!("cvt.rni.f32.f32 {n}, {t}"));
        // −ln 2 in two parts: the high part has 9 significant bits, so n · hi is exact.
        self.i(format!(
            "fma.rn.f32 {r}, {n}, {}, {xc}",
            c(f32::from_bits(0xBF31_7200))
        ));
        self.i(format!(
            "fma.rn.f32 {r}, {n}, {}, {r}",
            c(f32::from_bits(0xB5BF_BE8E))
        ));
        self.i(format!("mov.f32 {p}, {}", c(1.0 / 5040.0)));
        for k in [720.0f32, 120.0, 24.0, 6.0, 2.0, 1.0, 1.0] {
            self.i(format!("fma.rn.f32 {p}, {p}, {r}, {}", c(1.0 / k)));
        }
        // 2ⁿ = 2^(n1) · 2^(n2), each exponent in the normal range.
        let (ni, n1, n2, b1, b2, s1, s2) = (
            self.r(),
            self.r(),
            self.r(),
            self.r(),
            self.r(),
            self.f(),
            self.f(),
        );
        self.i(format!("cvt.rzi.s32.f32 {ni}, {n}"));
        self.i(format!("shr.s32 {n1}, {ni}, 1"));
        self.i(format!("sub.s32 {n2}, {ni}, {n1}"));
        self.i(format!("add.s32 {b1}, {n1}, 127"));
        self.i(format!("shl.b32 {b1}, {b1}, 23"));
        self.i(format!("add.s32 {b2}, {n2}, 127"));
        self.i(format!("shl.b32 {b2}, {b2}, 23"));
        self.i(format!("mov.b32 {s1}, {b1}"));
        self.i(format!("mov.b32 {s2}, {b2}"));
        let (y, nan) = (self.f(), self.p());
        self.i(format!("mul.f32 {y}, {p}, {s1}"));
        self.i(format!("mul.f32 {y}, {y}, {s2}"));
        self.i(format!("setp.nan.f32 {nan}, {x}, {x}"));
        self.i(format!("selp.f32 {y}, {x}, {y}, {nan}"));
        y
    }

    fn apply(&mut self, op: Op, args: &[String]) -> String {
        match op {
            Op::Add | Op::Mul => {
                let y = self.f();
                let m = if op == Op::Add { "add" } else { "mul" };
                self.i(format!("{m}.rn.f32 {y}, {}, {}", args[0], args[1]));
                y
            }
            Op::Exp => self.exp(&args[0]),
        }
    }
}

/// The element-wise kernel of `op` on `dtype`. With `dims = None` every operand is
/// contiguous and the output is too: parameters `(out, in…, n)`. With `Some(d)` the output
/// is contiguous and each input has its own element strides over `d ≤ 8` dimensions (0
/// broadcasts, negative walks backwards; the base pointer is the view's first element):
/// parameters `(out, in…, n, shape[8], strides₀[8], …)`, all 32-bit.
pub fn elementwise(op: Op, dtype: DType, dims: Option<usize>) -> Kernel {
    let arity = op.arity();
    let layout = dims.map_or_else(|| "c".to_string(), |d| format!("s{d}"));
    let name = format!("nuvec_{}_{}_{layout}", op.name(), dtype.name());
    let mut params = vec![".param .u64 out".to_string()];
    params.extend((0..arity).map(|k| format!(".param .u64 in{k}")));
    params.push(".param .u32 n".into());
    if dims.is_some() {
        params.push(format!(".param .align 4 .b8 shape[{}]", 4 * MAX_DIMS));
        params.extend(
            (0..arity).map(|k| format!(".param .align 4 .b8 strides{k}[{}]", 4 * MAX_DIMS)),
        );
    }
    let mut b = Builder::default();
    let i = b.thread_index();
    let (n, done, oob) = (b.r(), b.label(), b.p());
    b.i(format!("ld.param.u32 {n}, [n]"));
    b.i(format!("setp.ge.u32 {oob}, {i}, {n}"));
    b.i(format!("@{oob} bra {done}"));
    let out = b.rd();
    b.i(format!("ld.param.u64 {out}, [out]"));
    b.i(format!("cvta.to.global.u64 {out}, {out}"));
    let bases: Vec<String> = (0..arity)
        .map(|k| {
            let base = b.rd();
            b.i(format!("ld.param.u64 {base}, [in{k}]"));
            b.i(format!("cvta.to.global.u64 {base}, {base}"));
            base
        })
        .collect();
    let offsets: Vec<String> = match dims {
        None => vec![i.clone(); arity],
        Some(d) => {
            let offs: Vec<String> = (0..arity)
                .map(|_| {
                    let o = b.r();
                    b.i(format!("mov.u32 {o}, 0"));
                    o
                })
                .collect();
            // Peel the index from the innermost dimension outwards.
            let rest = b.r();
            b.i(format!("mov.u32 {rest}, {i}"));
            for axis in (0..d).rev() {
                let (size, q, rem) = (b.r(), b.r(), b.r());
                b.i(format!("ld.param.u32 {size}, [shape+{}]", 4 * axis));
                b.i(format!("div.u32 {q}, {rest}, {size}"));
                b.i(format!("mul.lo.u32 {rem}, {q}, {size}"));
                b.i(format!("sub.u32 {rem}, {rest}, {rem}"));
                for (k, o) in offs.iter().enumerate() {
                    let stride = b.r();
                    b.i(format!("ld.param.u32 {stride}, [strides{k}+{}]", 4 * axis));
                    b.i(format!("mad.lo.s32 {o}, {rem}, {stride}, {o}"));
                }
                b.i(format!("mov.u32 {rest}, {q}"));
            }
            offs
        }
    };
    let args: Vec<String> = bases
        .iter()
        .zip(&offsets)
        .map(|(base, off)| {
            let addr = b.address(base, off, dtype.size());
            b.load(dtype, &addr)
        })
        .collect();
    let y = b.apply(op, &args);
    let addr = b.address(&out, &i, dtype.size());
    b.store(dtype, &addr, &y);
    b.at(&done);
    b.i("ret");
    Kernel {
        ptx: b.finish(&name, &params, None),
        name,
    }
}

/// Threads per block of [`row_sum`].
pub const ROW_THREADS: u32 = 256;

/// The sum of each contiguous row: parameters `(out, in, len, rows)`, one block of
/// [`ROW_THREADS`] per row. Each thread accumulates a strided share in f32, warps reduce
/// with shuffles, and the first warp reduces the warps' sums; the result is rounded to
/// `dtype`. Rows of a few elements leave most of the block idle.
pub fn row_sum(dtype: DType) -> Kernel {
    let name = format!("nuvec_row_sum_{}", dtype.name());
    let params = [
        ".param .u64 out".to_string(),
        ".param .u64 in0".into(),
        ".param .u32 len".into(),
        ".param .u32 rows".into(),
    ];
    let shared = format!(".shared .align 4 .f32 warp_sums[{}];", ROW_THREADS / 32);
    let mut b = Builder::default();
    let (row, tid, len) = (b.r(), b.r(), b.r());
    b.i(format!("mov.u32 {row}, %ctaid.x"));
    b.i(format!("mov.u32 {tid}, %tid.x"));
    b.i(format!("ld.param.u32 {len}, [len]"));
    let (out, input) = (b.rd(), b.rd());
    b.i(format!("ld.param.u64 {out}, [out]"));
    b.i(format!("cvta.to.global.u64 {out}, {out}"));
    b.i(format!("ld.param.u64 {input}, [in0]"));
    b.i(format!("cvta.to.global.u64 {input}, {input}"));
    // The row's first element: in + row · len · size (64-bit).
    let (start, base) = (b.rd(), b.rd());
    b.i(format!("mul.wide.u32 {start}, {row}, {len}"));
    b.i(format!("mul.lo.u64 {start}, {start}, {}", dtype.size()));
    b.i(format!("add.s64 {base}, {input}, {start}"));
    let (acc, j) = (b.f(), b.r());
    b.i(format!("mov.f32 {acc}, 0f00000000"));
    b.i(format!("mov.u32 {j}, {tid}"));
    let (top, end, stop) = (b.label(), b.label(), b.p());
    b.at(&top);
    b.i(format!("setp.ge.u32 {stop}, {j}, {len}"));
    b.i(format!("@{stop} bra {end}"));
    let (wide, addr) = (b.rd(), b.rd());
    b.i(format!("mul.wide.u32 {wide}, {j}, {}", dtype.size()));
    b.i(format!("add.s64 {addr}, {base}, {wide}"));
    let v = b.load(dtype, &addr);
    b.i(format!("add.f32 {acc}, {acc}, {v}"));
    b.i(format!("add.u32 {j}, {j}, {ROW_THREADS}"));
    b.i(format!("bra {top}"));
    b.at(&end);
    let warp = |b: &mut Builder, acc: &str| {
        for d in [16, 8, 4, 2, 1] {
            let o = b.f();
            b.i(format!(
                "shfl.sync.bfly.b32 {o}, {acc}, {d}, 31, 0xffffffff"
            ));
            b.i(format!("add.f32 {acc}, {acc}, {o}"));
        }
    };
    warp(&mut b, &acc);
    let (lane, wid, first, slot, sbase) = (b.r(), b.r(), b.p(), b.r(), b.r());
    b.i(format!("and.b32 {lane}, {tid}, 31"));
    b.i(format!("shr.u32 {wid}, {tid}, 5"));
    b.i(format!("setp.eq.u32 {first}, {lane}, 0"));
    b.i(format!("mov.u32 {sbase}, warp_sums"));
    b.i(format!("mad.lo.u32 {slot}, {wid}, 4, {sbase}"));
    b.i(format!("@{first} st.shared.f32 [{slot}], {acc}"));
    b.i("bar.sync 0");
    let (skip, warp0) = (b.label(), b.p());
    b.i(format!("setp.ne.u32 {warp0}, {wid}, 0"));
    b.i(format!("@{warp0} bra {skip}"));
    let (total, valid, lslot) = (b.f(), b.p(), b.r());
    b.i(format!("mov.f32 {total}, 0f00000000"));
    b.i(format!("setp.lt.u32 {valid}, {lane}, {}", ROW_THREADS / 32));
    b.i(format!("mad.lo.u32 {lslot}, {lane}, 4, {sbase}"));
    b.i(format!("@{valid} ld.shared.f32 {total}, [{lslot}]"));
    warp(&mut b, &total);
    let (not_first, oaddr, owide) = (b.p(), b.rd(), b.rd());
    b.i(format!("setp.ne.u32 {not_first}, {lane}, 0"));
    b.i(format!("@{not_first} bra {skip}"));
    b.i(format!("mul.wide.u32 {owide}, {row}, {}", dtype.size()));
    b.i(format!("add.s64 {oaddr}, {out}, {owide}"));
    b.store(dtype, &oaddr, &total);
    b.at(&skip);
    b.i("ret");
    Kernel {
        ptx: b.finish(&name, &params, Some(&shared)),
        name,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernels_are_well_formed() {
        for op in [Op::Add, Op::Mul, Op::Exp] {
            for dtype in [DType::F32, DType::F16, DType::BF16] {
                for dims in [None, Some(1), Some(4)] {
                    let k = elementwise(op, dtype, dims);
                    assert!(k.ptx.contains(&format!(".visible .entry {}(", k.name)));
                    assert_eq!(k.ptx.matches('{').count(), k.ptx.matches('}').count());
                }
            }
        }
        assert!(row_sum(DType::BF16).ptx.contains("shfl.sync.bfly.b32"));
    }
}
