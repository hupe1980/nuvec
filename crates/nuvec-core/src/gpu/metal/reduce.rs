//! MLX's reduction dispatch (`backend/metal/reduce.cpp` and `common/reduce.cpp`, v0.32.4):
//! a plan from the input layout and the reduced axes, then an all-reduction, a row
//! reduction (the reduced axis is contiguous) or a column reduction (it is strided).

use super::*;

/// Values per thread (MLX's `REDUCE_N_READS`).
const N_READS: usize = 4;
/// Outputs per threadgroup of `row_reduce_simple` (MLX's `REDUCE_N_WRITES`).
const N_WRITES: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlanType {
    ContiguousAll,
    ContiguousRow,
    ContiguousStrided,
    GeneralRow,
    GeneralStrided,
    General,
}

/// MLX's `ReductionPlan`: the reduced dimensions (merged where contiguous), innermost last.
struct Plan {
    kind: PlanType,
    shape: Vec<usize>,
    strides: Vec<isize>,
}

/// MLX's `get_reduction_plan`.
fn plan(meta: &Output, axes: &[usize]) -> Plan {
    let (sh, st) = (&meta.shape, &meta.strides);
    let make = |kind, shape, strides| Plan {
        kind,
        shape,
        strides,
    };
    if axes.len() == sh.len() && meta.flags.contiguous {
        return make(PlanType::ContiguousAll, vec![], vec![]);
    }
    if meta.flags.row_contiguous {
        let (mut shape, mut strides) = (vec![sh[axes[0]]], vec![st[axes[0]]]);
        for w in axes.windows(2) {
            let a = w[1];
            if a - 1 == w[0] && sh[a] > 1 {
                *shape.last_mut().expect("non-empty") *= sh[a];
                *strides.last_mut().expect("non-empty") = st[a];
            } else {
                shape.push(sh[a]);
                strides.push(st[a]);
            }
        }
        let keep: Vec<usize> = (0..shape.len()).filter(|&i| shape[i] != 1).collect();
        let (shape, strides): (Vec<usize>, Vec<isize>) =
            keep.iter().map(|&i| (shape[i], strides[i])).unzip();
        match strides.last() {
            Some(1) => return make(PlanType::ContiguousRow, shape, strides),
            Some(&s) if s > 1 => return make(PlanType::ContiguousStrided, shape, strides),
            _ => {}
        }
    }
    // Reduced axes by decreasing stride (broadcast axes first), merged where contiguous.
    let mut r: Vec<(usize, isize)> = axes
        .iter()
        .filter(|&&a| sh[a] > 1)
        .map(|&a| (sh[a], st[a]))
        .collect();
    r.sort_by(|a, b| {
        let (az, bz) = (a.1 == 0, b.1 == 0);
        if az != bz {
            a.1.cmp(&b.1)
        } else {
            b.1.cmp(&a.1)
        }
    });
    for i in (1..r.len()).rev() {
        let (a, b) = (r[i], r[i - 1]);
        if b.1 == a.0 as isize * a.1 {
            r.remove(i);
            r[i - 1] = (a.0 * b.0, a.1);
        }
    }
    let (shape, strides): (Vec<usize>, Vec<isize>) = r.into_iter().unzip();
    match strides.last() {
        Some(1) => return make(PlanType::GeneralRow, shape, strides),
        Some(&last) if last > 1 => {
            let (mut size, mut expand) = (1isize, false);
            for i in (0..sh.len()).rev() {
                if axes.last() == Some(&i) {
                    continue;
                }
                if st[i] == 0 {
                    if sh[i] == 1 {
                        continue;
                    }
                    expand = true;
                    break;
                }
                if st[i] != size && sh[i] != 1 {
                    break;
                }
                size *= sh[i] as isize;
            }
            if size > last || (size == last && !expand) {
                return make(PlanType::GeneralStrided, shape, strides);
            }
        }
        _ => {}
    }
    make(PlanType::General, shape, strides)
}

/// The input's kept dimensions (MLX's `shapes_without_reduction_axes`).
fn kept(meta: &Output, axes: &[usize]) -> (Vec<usize>, Vec<isize>) {
    (0..meta.shape.len())
        .filter(|d| !axes.contains(d))
        .map(|d| (meta.shape[d], meta.strides[d]))
        .unzip()
}

/// MLX's `get_kernel_reduce_ndim`.
fn kernel_ndim(n: usize) -> usize {
    match n {
        0 | 1 => 1,
        2 => 2,
        _ => 5,
    }
}

/// MLX's `threadgroup_size_from_row_size`.
fn row_threads(row: usize) -> usize {
    match row {
        ..=512 => 32,
        513..=1024 => 128,
        _ => (row.div_ceil(N_READS).div_ceil(32) * 32).min(1024),
    }
}

/// Shape and stride bytes, never empty (MLX pushes a 0).
fn dims(shape: &[usize], strides: &[isize]) -> (Vec<u8>, Vec<u8>) {
    let s: Vec<i32> = if shape.is_empty() {
        vec![0]
    } else {
        shape.iter().map(|&d| d as i32).collect()
    };
    let t: Vec<i64> = if strides.is_empty() {
        vec![0]
    } else {
        strides.iter().map(|&d| d as i64).collect()
    };
    (
        bytemuck::cast_slice(&s).to_vec(),
        bytemuck::cast_slice(&t).to_vec(),
    )
}

/// The arguments of the row and column kernels (slots 2–10): the reduced size and count,
/// the kept dimensions, and the other reduced dimensions.
struct Args {
    a: [u8; 8],
    b: [u8; 8],
    shape: Vec<u8>,
    strides: Vec<u8>,
    ndim: [u8; 4],
    rshape: Vec<u8>,
    rstrides: Vec<u8>,
    rndim: [u8; 4],
    non: [u8; 8],
}

impl Args {
    fn new(
        a: i64,
        b: i64,
        kept: (&[usize], &[isize]),
        reduce: (&[usize], &[isize]),
        non: usize,
    ) -> Self {
        let (shape, strides) = dims(kept.0, kept.1);
        let (rshape, rstrides) = dims(reduce.0, reduce.1);
        Self {
            a: a.to_le_bytes(),
            b: b.to_le_bytes(),
            shape,
            strides,
            ndim: (kept.0.len() as i32).to_le_bytes(),
            rshape,
            rstrides,
            rndim: (reduce.0.len() as i32).to_le_bytes(),
            non: (non as u64).to_le_bytes(),
        }
    }

    fn push<'a>(&'a self, args: &mut Vec<Arg<'a>>, with_non: bool) {
        args.extend([
            Arg::bytes(&self.a),
            Arg::bytes(&self.b),
            Arg::bytes(&self.shape),
            Arg::bytes(&self.strides),
            Arg::bytes(&self.ndim),
            Arg::bytes(&self.rshape),
            Arg::bytes(&self.rstrides),
            Arg::bytes(&self.rndim),
        ]);
        if with_non {
            args.push(Arg::bytes(&self.non));
        }
    }
}

/// Whether MLX's kernels are instantiated for this reduction and element type.
fn instantiated(op: &str, dtype: DType) -> bool {
    use DType::*;
    match op {
        "and" | "or" => dtype == Bool,
        "sum" | "prod" => matches!(dtype, I32 | I64 | U32 | U64 | F16 | BF16 | F32),
        _ => matches!(
            dtype,
            I8 | I16 | I32 | I64 | U8 | U16 | U32 | U64 | F16 | BF16 | F32
        ),
    }
}

impl GpuEval {
    /// MLX's `Reduce::eval_gpu`. Returns `false` (nothing encoded) where the generic
    /// kernel must run instead: empty inputs, negative strides, more than 2³¹ elements and
    /// element types without instantiations.
    pub(super) fn reduce_mlx(
        &mut self,
        op: ReduceOp,
        axes: &[usize],
        operand: Operand<'_>,
        buffer: &mut Buffer,
    ) -> Result<bool> {
        let (meta, data) = operand;
        let dtype = meta.dtype;
        let name = match (op, dtype) {
            (ReduceOp::Max, DType::Bool) | (ReduceOp::Or, _) => "or",
            (ReduceOp::Min, DType::Bool) | (ReduceOp::And, _) => "and",
            (ReduceOp::Sum, _) => "sum",
            (ReduceOp::Prod, _) => "prod",
            (ReduceOp::Max, _) => "max",
            (ReduceOp::Min, _) => "min",
        };
        if meta.size == 0
            || meta.size > i32::MAX as usize
            || data.offset < 0
            || meta.strides.iter().any(|&s| s < 0)
            || axes.is_empty()
            || !instantiated(name, dtype)
        {
            return Ok(false);
        }
        let out_size: usize = (0..meta.shape.len())
            .filter(|d| !axes.contains(d))
            .map(|d| meta.shape[d])
            .product();
        if out_size == meta.size {
            // Every reduced axis has size 1: the reduction is a copy.
            self.copy(operand, buffer)?;
            return Ok(true);
        }
        let tag = format!("{name}_{}", dtype.name());
        // Half-precision sums and products accumulate in float32 (MLX: in the input type)
        // and are rounded once at the end.
        let widen = matches!(name, "sum" | "prod") && matches!(dtype, DType::F16 | DType::BF16);
        let acc = if widen { DType::F32 } else { dtype };
        let mut p = plan(meta, axes);
        let copy;
        let (meta, offset, input) = if p.kind == PlanType::General {
            // Copy to row-major order and plan again (as MLX).
            let mut c = Buffer::try_alloc(meta.size * dtype.size())?;
            self.copy(operand, &mut c)?;
            copy = (
                crate::kernels::strided(
                    dtype,
                    &meta.shape,
                    &crate::shape::row_contiguous_strides(&meta.shape),
                ),
                c,
            );
            p = plan(&copy.0, axes);
            (&copy.0, 0usize, copy.1.metal().expect("device storage"))
        } else {
            (
                meta,
                data.offset as usize * dtype.size(),
                data.buffer.metal().expect("checked"),
            )
        };
        let mut wide = if widen {
            Some(Buffer::try_alloc(out_size * 4)?)
        } else {
            None
        };
        let out = match &mut wide {
            Some(w) => w.metal_mut().expect("device storage"),
            None => buffer.metal_mut().expect("device storage"),
        };
        let second = format!("{name}_{}", acc.name());
        let tags = (tag.as_str(), second.as_str(), acc);
        match p.kind {
            PlanType::ContiguousAll => self.all_reduce(tags, (input, offset), meta.size, out)?,
            PlanType::ContiguousRow | PlanType::GeneralRow => {
                self.row_reduce(&tag, meta, axes, &p, (input, offset), out_size, out)?
            }
            _ => self.col_reduce(tags, meta, axes, &p, (input, offset), out_size, out)?,
        }
        if let Some(w) = wide {
            // Round the float32 result into the output (the element-wise cast kernel).
            let name = format!("nuvec_cast_float32_{}_c", dtype.name());
            let (zero, flags) = (0i64.to_le_bytes(), 0u32.to_le_bytes());
            let args = [
                Arg::read(w.metal().expect("device storage")),
                Arg::write(buffer.metal_mut().expect("device storage")),
                Arg::bytes(&zero),
                Arg::bytes(&flags),
            ];
            self.dispatch(&name, &args, out_size)?;
        }
        Ok(true)
    }

    fn all_reduce(
        &mut self,
        (tag, second_tag, acc): (&str, &str, DType),
        (input, offset): (&nuvec_gpu::Buffer, usize),
        n: usize,
        out: &mut nuvec_gpu::Buffer,
    ) -> Result<()> {
        let name = format!("nuvec_all_reduce_{tag}");
        let pipeline = self.pipeline(&name)?;
        let u64b = |v: usize| (v as u64).to_le_bytes();
        if n <= N_READS * 1024 {
            let threads = n.div_ceil(N_READS).div_ceil(32) * 32;
            let (a, b) = (u64b(n), u64b(n));
            let args = [
                Arg::Read(input, offset),
                Arg::write(out),
                Arg::bytes(&a),
                Arg::bytes(&b),
            ];
            self.command_buffer()?
                .dispatch_threads(&pipeline, &args, [threads, 1, 1], [threads, 1, 1])
                .map_err(backend)?;
            return self.counted();
        }
        // Two passes: rows of the input into an intermediate, then the intermediate.
        let (rows, second) = if n * acc.size() <= 1 << 26 {
            (32 * N_READS, 32)
        } else {
            (1024 * N_READS, 1024)
        };
        let mut partial = Buffer::try_alloc(rows * acc.size())?;
        let row = n.div_ceil(rows);
        let threads = row.div_ceil(N_READS).min(1024).div_ceil(32) * 32;
        let (a, b) = (u64b(n), u64b(row));
        {
            let args = [
                Arg::Read(input, offset),
                Arg::write(partial.metal_mut().expect("device storage")),
                Arg::bytes(&a),
                Arg::bytes(&b),
            ];
            self.command_buffer()?
                .dispatch_threads(&pipeline, &args, [threads, rows, 1], [threads, 1, 1])
                .map_err(backend)?;
            self.counted()?;
        }
        let pipeline = self.pipeline(&format!("nuvec_all_reduce_{second_tag}"))?;
        let (a, b) = (u64b(rows), u64b(rows));
        let args = [
            Arg::read(partial.metal().expect("device storage")),
            Arg::write(out),
            Arg::bytes(&a),
            Arg::bytes(&b),
        ];
        self.command_buffer()?
            .dispatch_threads(&pipeline, &args, [second, 1, 1], [second, 1, 1])
            .map_err(backend)?;
        self.counted()
    }

    #[allow(clippy::too_many_arguments)]
    fn row_reduce(
        &mut self,
        tag: &str,
        meta: &Output,
        axes: &[usize],
        p: &Plan,
        (input, offset): (&nuvec_gpu::Buffer, usize),
        out_size: usize,
        out: &mut nuvec_gpu::Buffer,
    ) -> Result<()> {
        let n = p.shape.len();
        let row = p.shape[n - 1];
        let (rs, rt) = (&p.shape[..n - 1], &p.strides[..n - 1]);
        let non_row: usize = rs.iter().product();
        let (ks, kt) = kept(meta, axes);
        let (ks, [kt]) = nuvec_cpu::collapse_dims(&ks, [&kt]);
        if row > 64 && p.kind == PlanType::ContiguousRow && rs.is_empty() && meta.size / row >= 32 {
            // Contiguous rows: N_WRITES outputs per threadgroup.
            let name = format!("nuvec_row_reduce_simple_{tag}");
            let pipeline = self.pipeline(&name)?;
            let mut threads = row_threads(row);
            if meta.dtype.size() == 8 {
                threads = threads.min(512);
            }
            let (a, b) = ((row as u64).to_le_bytes(), (out_size as i64).to_le_bytes());
            let args = [
                Arg::Read(input, offset),
                Arg::write(out),
                Arg::bytes(&a),
                Arg::bytes(&b),
            ];
            self.command_buffer()?
                .dispatch_threads(
                    &pipeline,
                    &args,
                    [threads, out_size.div_ceil(N_WRITES), 1],
                    [threads, 1, 1],
                )
                .map_err(backend)?;
            return self.counted();
        }
        let args = Args::new(row as i64, non_row as i64, (&ks, &kt), (rs, rt), non_row);
        let small = row <= 64;
        let name = format!(
            "nuvec_row_reduce_{}_{}_{tag}",
            if small { "small" } else { "looped" },
            kernel_ndim(rs.len())
        );
        let pipeline = self.pipeline(&name)?;
        let mut list = vec![Arg::Read(input, offset), Arg::write(out)];
        args.push(&mut list, false);
        let (grid, group) = if !small {
            let t = row_threads(row);
            ([t, out_size, 1], [t, 1, 1])
        } else if (non_row < 32 && row <= 8) || non_row <= 8 {
            ([out_size, 1, 1], [out_size.min(1024), 1, 1])
        } else {
            ([32, out_size, 1], [32, 1, 1])
        };
        self.command_buffer()?
            .dispatch_threads(&pipeline, &list, grid, group)
            .map_err(backend)?;
        self.counted()
    }

    #[allow(clippy::too_many_arguments)]
    fn col_reduce(
        &mut self,
        (tag, second_tag, acc): (&str, &str, DType),
        meta: &Output,
        axes: &[usize],
        p: &Plan,
        (input, offset): (&nuvec_gpu::Buffer, usize),
        out_size: usize,
        out: &mut nuvec_gpu::Buffer,
    ) -> Result<()> {
        let n = p.shape.len();
        let (size, stride) = (p.shape[n - 1], p.strides[n - 1]);
        let non_col: usize = p.shape[..n - 1].iter().product();
        let total = size * non_col;
        // Kept dimensions outside the contiguous block that each column reduction covers.
        let (mut ks, mut kt) = kept(meta, axes);
        let mut back = 1isize;
        while !ks.is_empty() && back < stride {
            back *= *ks.last().expect("non-empty") as isize;
            ks.pop();
            kt.pop();
        }
        let (ks, [kt]) = nuvec_cpu::collapse_dims(&ks, [&kt]);
        // MLX's `output_grid_for_col_reduce`: the (row-major, reduced axes kept as 1) output
        // without its trailing dimensions of stride below the reduction stride.
        let out_shape: Vec<usize> = (0..meta.shape.len())
            .map(|d| if axes.contains(&d) { 1 } else { meta.shape[d] })
            .collect();
        let out_strides = crate::shape::row_contiguous_strides(&out_shape);
        let mut keep = out_shape.len();
        while keep > 0 && out_strides[keep - 1] < stride {
            keep -= 1;
        }
        let grid_out: usize = out_shape[..keep].iter().product();
        // Every kernel reduces over all reduced dimensions, the contiguous one last.
        let rs: Vec<usize> = p.shape.clone();
        let rt: Vec<isize> = p.strides.clone();
        let args = Args::new(size as i64, stride as i64, (&ks, &kt), (&rs, &rt), non_col);
        let nd = kernel_ndim(rs.len());
        if total < 32 {
            let name = format!("nuvec_col_reduce_small_{nd}_{tag}");
            let pipeline = self.pipeline(&name)?;
            let blocks = (stride as usize).div_ceil(N_READS);
            let tx = blocks.min(32);
            let ty = 8
                .min(pipeline.max_threads_per_threadgroup() / tx)
                .min(total);
            let mut list = vec![Arg::Read(input, offset), Arg::write(out)];
            args.push(&mut list, true);
            self.command_buffer()?
                .dispatch_groups(
                    &pipeline,
                    &list,
                    [blocks.div_ceil(tx), grid_out, 1],
                    [tx, ty, 1],
                )
                .map_err(backend)?;
            return self.counted();
        }
        if stride < 32 && total >= 1024 {
            let outer = if total >= 32768 { 128 } else { 32 };
            let mut partial = Buffer::try_alloc(outer * out_size * acc.size())?;
            let name = format!("nuvec_col_reduce_longcolumn_{nd}_{tag}");
            let pipeline = self.pipeline(&name)?;
            let ty = total.div_ceil(outer).min(32);
            let os = (out_size as u64).to_le_bytes();
            {
                let mut list = vec![
                    Arg::Read(input, offset),
                    Arg::write(partial.metal_mut().expect("device storage")),
                ];
                args.push(&mut list, true);
                list.push(Arg::bytes(&os));
                self.command_buffer()?
                    .dispatch_groups(
                        &pipeline,
                        &list,
                        [grid_out, 1, outer],
                        [stride as usize, ty, 1],
                    )
                    .map_err(backend)?;
                self.counted()?;
            }
            return self.col_second_pass(second_tag, partial, outer, out_size, out);
        }
        if total > 256 && out_size / 32 < 1024 {
            let outer = 32;
            let mut partial = Buffer::try_alloc(outer * out_size * acc.size())?;
            let name = format!("nuvec_col_reduce_2pass_{nd}_32_32_{tag}");
            let pipeline = self.pipeline(&name)?;
            let os = ((out_size / stride as usize) as u64).to_le_bytes();
            {
                let part = partial.metal_mut().expect("device storage");
                self.col_blocks(
                    &pipeline,
                    (input, offset),
                    part,
                    &args,
                    stride as usize,
                    [grid_out * outer, 1],
                    Some(&os),
                )?;
            }
            return self.col_second_pass(second_tag, partial, outer, out_size, out);
        }
        let name = format!("nuvec_col_reduce_looped_{nd}_32_32_{tag}");
        let pipeline = self.pipeline(&name)?;
        self.col_blocks(
            &pipeline,
            (input, offset),
            out,
            &args,
            stride as usize,
            [grid_out, 1],
            None,
        )
    }

    /// MLX's `dispatch_col_reduce`: 32-column blocks in launches of at most 2³² threads,
    /// each with its first column at slot 11.
    #[allow(clippy::too_many_arguments)]
    fn col_blocks(
        &mut self,
        pipeline: &nuvec_gpu::Pipeline,
        (input, offset): (&nuvec_gpu::Buffer, usize),
        out: &mut nuvec_gpu::Buffer,
        args: &Args,
        columns: usize,
        [gy, gz]: [usize; 2],
        extra: Option<&[u8; 8]>,
    ) -> Result<()> {
        let (bn, threads) = (32usize, 256usize);
        let blocks = columns.div_ceil(bn);
        let max_blocks = u32::MAX as usize / threads;
        let mut start = 0;
        while start < blocks {
            let count = max_blocks.min(blocks - start);
            let first = ((start * bn) as i64).to_le_bytes();
            let mut list = vec![Arg::Read(input, offset), Arg::Write(&mut *out, 0)];
            args.push(&mut list, true);
            list.push(Arg::bytes(&first));
            if let Some(e) = extra {
                list.push(Arg::bytes(e));
            }
            self.command_buffer()?
                .dispatch_threads(pipeline, &list, [threads * count, gy, gz], [threads, 1, 1])
                .map_err(backend)?;
            self.counted()?;
            start += count;
        }
        Ok(())
    }

    /// Folds the `outer` partial results (`[outer, out_size]`) into `out`.
    fn col_second_pass(
        &mut self,
        tag: &str,
        partial: Buffer,
        outer: usize,
        out_size: usize,
        out: &mut nuvec_gpu::Buffer,
    ) -> Result<()> {
        let name = format!("nuvec_col_reduce_looped_1_32_32_{tag}");
        let pipeline = self.pipeline(&name)?;
        let args = Args::new(
            outer as i64,
            out_size as i64,
            (&[], &[]),
            (&[outer], &[out_size as isize]),
            1,
        );
        self.col_blocks(
            &pipeline,
            (partial.metal().expect("device storage"), 0),
            out,
            &args,
            out_size,
            [1, 1],
            None,
        )
    }
}
