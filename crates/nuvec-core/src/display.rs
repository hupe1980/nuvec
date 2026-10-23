//! Printing values as MLX and NumPy do: nested brackets, `...` for large arrays.

use crate::{Array, Complex64, DType, Result, bf16, f16};

/// Arrays with more elements are summarised.
const THRESHOLD: usize = 1000;
/// Elements kept at each end of a summarised axis.
const EDGE: usize = 3;

fn float(x: f64) -> String {
    if x.is_nan() {
        "nan".into()
    } else if x.is_infinite() {
        if x > 0.0 { "inf".into() } else { "-inf".into() }
    } else {
        format!("{x}")
    }
}

/// Every element as text, in row-major order (evaluates the array).
fn elements(a: &Array) -> Result<Vec<String>> {
    fn map<T: crate::Element>(a: &Array, f: impl Fn(T) -> String) -> Result<Vec<String>> {
        Ok(a.to_vec::<T>()?.into_iter().map(f).collect())
    }
    match a.dtype() {
        DType::Bool => map(a, |x: bool| if x { "True" } else { "False" }.into()),
        DType::U8 => map(a, |x: u8| x.to_string()),
        DType::U16 => map(a, |x: u16| x.to_string()),
        DType::U32 => map(a, |x: u32| x.to_string()),
        DType::U64 => map(a, |x: u64| x.to_string()),
        DType::I8 => map(a, |x: i8| x.to_string()),
        DType::I16 => map(a, |x: i16| x.to_string()),
        DType::I32 => map(a, |x: i32| x.to_string()),
        DType::I64 => map(a, |x: i64| x.to_string()),
        DType::F16 => map(a, |x: f16| float(x.to_f64())),
        DType::BF16 => map(a, |x: bf16| float(x.to_f64())),
        DType::F32 => map(a, |x: f32| float(f64::from(x))),
        DType::F64 => map(a, float),
        DType::Complex64 => map(a, |z: Complex64| {
            let (re, im) = (f64::from(z.re), f64::from(z.im));
            let sign = if im < 0.0 || (im == 0.0 && im.is_sign_negative()) {
                '-'
            } else {
                '+'
            };
            format!("{}{sign}{}j", float(re), float(im.abs()))
        }),
    }
}

/// The values of `a` as nested brackets (`[[1, 2],\n       [3, 4]]`), indented for an
/// `array(` prefix; a 0-D array is its element.
pub(crate) fn values(a: &Array) -> Result<String> {
    let items = elements(a)?;
    let shape: Vec<usize> = a.dims().to_vec();
    if shape.is_empty() {
        return Ok(items.into_iter().next().unwrap_or_default());
    }
    let summarise = a.size() > THRESHOLD;
    let mut out = String::new();
    write_level(&mut out, &items, &shape, 0, 0, summarise);
    Ok(out)
}

fn write_level(
    out: &mut String,
    items: &[String],
    shape: &[usize],
    offset: usize,
    depth: usize,
    summarise: bool,
) {
    let n = shape[0];
    let stride: usize = shape[1..].iter().product();
    let indices: Vec<Option<usize>> = if summarise && n > 2 * EDGE {
        (0..EDGE)
            .map(Some)
            .chain(std::iter::once(None))
            .chain((n - EDGE..n).map(Some))
            .collect()
    } else {
        (0..n).map(Some).collect()
    };
    out.push('[');
    for (k, i) in indices.iter().enumerate() {
        if k > 0 {
            out.push(',');
            if shape.len() > 1 {
                // Rows of a matrix on their own lines, aligned under `array([`.
                out.push_str(&"\n".repeat(shape.len() - 1));
                out.push_str(&" ".repeat("array(".len() + depth + 1));
            } else {
                out.push(' ');
            }
        }
        match i {
            None => out.push_str("..."),
            Some(i) if shape.len() == 1 => out.push_str(&items[offset + i]),
            Some(i) => write_level(
                out,
                items,
                &shape[1..],
                offset + i * stride,
                depth + 1,
                summarise,
            ),
        }
    }
    out.push(']');
}

#[cfg(test)]
mod tests {
    use crate::Array;

    #[test]
    fn mlx_style() {
        let a = crate::array![[1, 2], [3, 4]];
        assert_eq!(
            a.to_string(),
            "array([[1, 2],\n       [3, 4]], dtype=int32)"
        );
        assert_eq!(
            crate::array![1.5, -0.0].to_string(),
            "array([1.5, -0], dtype=float32)"
        );
        assert_eq!(Array::from(true).to_string(), "array(True, dtype=bool)");
        let big = crate::ops::arange(0, 2000, 1);
        assert_eq!(
            big.to_string(),
            "array([0, 1, 2, ..., 1997, 1998, 1999], dtype=int32)"
        );
        let bad = big.reshape([7, -1]);
        assert!(bad.to_string().starts_with("<failed: reshape at "), "{bad}");
        let debug = format!("{bad:?}");
        assert!(
            debug.contains("failed") && debug.contains("reshape"),
            "{debug}"
        );
        let lazy = &big + 1;
        assert!(format!("{lazy:?}").contains("unevaluated"));
        assert!(!lazy.is_evaluated(), "Debug never evaluates");
    }
}
