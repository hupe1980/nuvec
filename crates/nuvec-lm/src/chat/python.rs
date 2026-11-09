//! Python's spelling of values in rendered templates: `json.dumps`, `str`/`repr` and the
//! `strftime` directives templates use.
//!
//! The float formatting follows CPython's `float_repr_style = 'short'` (`Python/pystrtod.c`,
//! `format_float_short`, the `'r'` mode with `Py_DTSF_ADD_DOT_0`).

use std::fmt::Write;
use std::time::{SystemTime, UNIX_EPOCH};

use minijinja::value::{Value, ValueKind};
use minijinja::{Error, ErrorKind};

/// `json.dumps`'s options.
pub(crate) struct Dumps {
    pub ensure_ascii: bool,
    /// `None` for one line; otherwise the indent of each level.
    pub indent: Option<String>,
    pub item_separator: String,
    pub key_separator: String,
    pub sort_keys: bool,
}

impl Default for Dumps {
    fn default() -> Self {
        Self {
            ensure_ascii: false,
            indent: None,
            item_separator: ", ".into(),
            key_separator: ": ".into(),
            sort_keys: false,
        }
    }
}

/// `json.dumps(value, ...)`.
pub(crate) fn dumps(value: &Value, o: &Dumps) -> Result<String, Error> {
    let mut out = String::new();
    dump(value, o, 0, &mut out)?;
    Ok(out)
}

fn dump(value: &Value, o: &Dumps, depth: usize, out: &mut String) -> Result<(), Error> {
    match value.kind() {
        ValueKind::None => out.push_str("null"),
        ValueKind::Bool => out.push_str(if value.is_true() { "true" } else { "false" }),
        ValueKind::Number => out.push_str(&json_number(value)),
        ValueKind::String => json_string(value.as_str().unwrap_or_default(), o.ensure_ascii, out),
        ValueKind::Seq | ValueKind::Iterable => {
            let items: Vec<Value> = value.try_iter()?.collect();
            container(out, o, depth, '[', ']', items.len(), |i, out| {
                dump(&items[i], o, depth + 1, out)
            })?;
        }
        ValueKind::Map => {
            let mut pairs: Vec<(String, Value)> = Vec::new();
            for key in value.try_iter()? {
                let item = value.get_item(&key)?;
                pairs.push((json_key(&key)?, item));
            }
            if o.sort_keys {
                pairs.sort_by(|a, b| a.0.cmp(&b.0));
            }
            container(out, o, depth, '{', '}', pairs.len(), |i, out| {
                json_string(&pairs[i].0, o.ensure_ascii, out);
                out.push_str(&o.key_separator);
                dump(&pairs[i].1, o, depth + 1, out)
            })?;
        }
        kind => {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                format!("Object of type {kind} is not JSON serializable"),
            ));
        }
    }
    Ok(())
}

/// A container in `json.dumps`'s layout: one line, or one item per line with `indent`.
fn container(
    out: &mut String,
    o: &Dumps,
    depth: usize,
    open: char,
    close: char,
    n: usize,
    mut item: impl FnMut(usize, &mut String) -> Result<(), Error>,
) -> Result<(), Error> {
    out.push(open);
    if n == 0 {
        out.push(close);
        return Ok(());
    }
    let newline = |out: &mut String, level: usize| {
        if let Some(indent) = &o.indent {
            out.push('\n');
            for _ in 0..level {
                out.push_str(indent);
            }
        }
    };
    newline(out, depth + 1);
    for i in 0..n {
        if i > 0 {
            out.push_str(&o.item_separator);
            newline(out, depth + 1);
        }
        item(i, out)?;
    }
    newline(out, depth);
    out.push(close);
    Ok(())
}

/// A map key as `json.dumps` converts it: strings as they are; numbers, booleans and `None`
/// as their JSON spelling.
fn json_key(key: &Value) -> Result<String, Error> {
    Ok(match key.kind() {
        ValueKind::String => key.as_str().unwrap_or_default().to_owned(),
        ValueKind::Number => json_number(key),
        ValueKind::Bool => (if key.is_true() { "true" } else { "false" }).to_owned(),
        ValueKind::None => "null".to_owned(),
        kind => {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                format!("keys must be str, int, float, bool or None, not {kind}"),
            ));
        }
    })
}

fn json_number(value: &Value) -> String {
    if value.is_integer() {
        return value.to_string();
    }
    let x = f64::try_from(value.clone()).unwrap_or(f64::NAN);
    if x.is_nan() {
        "NaN".into()
    } else if x.is_infinite() {
        (if x < 0.0 { "-Infinity" } else { "Infinity" }).into()
    } else {
        float_repr(x)
    }
}

/// `json.dumps`'s string encoding (`py_encode_basestring`, or `_ascii` with `ensure_ascii`).
fn json_string(s: &str, ensure_ascii: bool, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || ensure_ascii && (c as u32) > 0x7e => {
                let mut units = [0u16; 2];
                for u in c.encode_utf16(&mut units) {
                    let _ = write!(out, "\\u{u:04x}");
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// `repr(x)` of a finite or non-finite float, as CPython prints it.
pub(crate) fn float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return (if x < 0.0 { "-inf" } else { "inf" }).into();
    }
    // The shortest round-trip digits and the decimal exponent, from Rust's `{:e}`.
    let sci = format!("{:e}", x.abs());
    let (mantissa, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let sign = if x.is_sign_negative() { "-" } else { "" };
    let n = digits.len() as i32;
    if (-4..16).contains(&exp) {
        let body = if exp >= 0 {
            let whole = (exp + 1) as usize;
            if digits.len() <= whole {
                format!("{digits}{}.0", "0".repeat(whole - digits.len()))
            } else {
                format!("{}.{}", &digits[..whole], &digits[whole..])
            }
        } else {
            format!("0.{}{digits}", "0".repeat((-exp - 1) as usize))
        };
        format!("{sign}{body}")
    } else {
        let mantissa = if n > 1 {
            format!("{}.{}", &digits[..1], &digits[1..])
        } else {
            digits
        };
        let esign = if exp < 0 { '-' } else { '+' };
        format!("{sign}{mantissa}e{esign}{:02}", exp.abs())
    }
}

/// `str(value)` as Python prints it, for the values that differ from minijinja's spelling
/// (floats, sequences and maps); `None` for the rest.
pub(crate) fn py_str(value: &Value) -> Option<String> {
    match value.kind() {
        ValueKind::Number if !value.is_integer() => {
            Some(float_repr(f64::try_from(value.clone()).ok()?))
        }
        ValueKind::Seq | ValueKind::Map => {
            let mut out = String::new();
            repr(value, &mut out).ok()?;
            Some(out)
        }
        _ => None,
    }
}

/// `repr(value)` for JSON-like values.
fn repr(value: &Value, out: &mut String) -> Result<(), Error> {
    match value.kind() {
        ValueKind::String => string_repr(value.as_str().unwrap_or_default(), out),
        ValueKind::Seq => {
            out.push('[');
            for (i, item) in value.try_iter()?.enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                repr(&item, out)?;
            }
            out.push(']');
        }
        ValueKind::Map => {
            out.push('{');
            for (i, key) in value.try_iter()?.enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                let item = value.get_item(&key)?;
                repr(&key, out)?;
                out.push_str(": ");
                repr(&item, out)?;
            }
            out.push('}');
        }
        ValueKind::Number => match py_str(value) {
            Some(s) => out.push_str(&s),
            None => out.push_str(&value.to_string()),
        },
        _ => out.push_str(&value.to_string()),
    }
    Ok(())
}

/// Python's `repr` of a string: single quotes unless it holds `'` and no `"`; the escapes
/// of `unicode_repr` (`\\`, the quote, `\t`, `\n`, `\r`, `\xhh`/`\uhhhh`/`\Uhhhhhhhh` for
/// characters that are not printable).
fn string_repr(s: &str, out: &mut String) {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if !printable(c) => {
                let v = c as u32;
                let _ = if v < 0x100 {
                    write!(out, "\\x{v:02x}")
                } else if v < 0x10000 {
                    write!(out, "\\u{v:04x}")
                } else {
                    write!(out, "\\U{v:08x}")
                };
            }
            c => out.push(c),
        }
    }
    out.push(quote);
}

/// `str.isprintable` for one character: not in the categories Cc, Cf, Cs, Co, Cn, Zl, Zp,
/// or Zs other than the space.
fn printable(c: char) -> bool {
    use unicode_general_category::{GeneralCategory as G, get_general_category};
    if c == ' ' {
        return true;
    }
    !matches!(
        get_general_category(c),
        G::Control
            | G::Format
            | G::Surrogate
            | G::PrivateUse
            | G::Unassigned
            | G::LineSeparator
            | G::ParagraphSeparator
            | G::SpaceSeparator
    )
}

const MONTHS: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];
const DAYS: [&str; 7] = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
];

/// `time.strftime(format)` at `now` in UTC, for the directives `%d %m %Y %y %H %M %S %b %B
/// %a %A %j %%`; any other is an error naming it.
pub(crate) fn strftime(format: &str, now: SystemTime) -> Result<String, Error> {
    let secs = match now.duration_since(UNIX_EPOCH) {
        Ok(d) => d.as_secs() as i64,
        Err(e) => -(e.duration().as_secs() as i64),
    };
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    // 1970-01-01 was a Thursday (index 3 with Monday first).
    let weekday = (days + 3).rem_euclid(7) as usize;
    let yday = days - days_from_civil(year, 1, 1) + 1;
    let (h, m, s) = (rem / 3600, rem % 3600 / 60, rem % 60);
    let mut out = String::new();
    let mut chars = format.chars();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        let d = chars.next().unwrap_or('%');
        let _ = match d {
            'd' => write!(out, "{day:02}"),
            'm' => write!(out, "{month:02}"),
            'Y' => write!(out, "{year}"),
            'y' => write!(out, "{:02}", year.rem_euclid(100)),
            'H' => write!(out, "{h:02}"),
            'M' => write!(out, "{m:02}"),
            'S' => write!(out, "{s:02}"),
            'B' => write!(out, "{}", MONTHS[month as usize - 1]),
            'b' => write!(out, "{}", &MONTHS[month as usize - 1][..3]),
            'A' => write!(out, "{}", DAYS[weekday]),
            'a' => write!(out, "{}", &DAYS[weekday][..3]),
            'j' => write!(out, "{yday:03}"),
            '%' => write!(out, "%"),
            other => {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    format!("strftime_now: the directive %{other} is not supported"),
                ));
            }
        };
    }
    Ok(out)
}

/// Howard Hinnant's `civil_from_days`: the proleptic Gregorian date of a day count from
/// 1970-01-01.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = yoe + era * 400 + i64::from(m <= 2);
    (y, m, d)
}

/// Howard Hinnant's `days_from_civil`.
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let m = i64::from(m);
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn floats_print_as_python_repr() {
        for (x, want) in [
            (1.0, "1.0"),
            (1e-05, "1e-05"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (0.0001, "0.0001"),
            (0.00012, "0.00012"),
            (123456789012345.6, "123456789012345.6"),
            (1.5e-7, "1.5e-07"),
            (-0.0, "-0.0"),
            (0.1, "0.1"),
            (21.5, "21.5"),
            (2.5e300, "2.5e+300"),
            (f64::NAN, "nan"),
            (f64::NEG_INFINITY, "-inf"),
        ] {
            assert_eq!(float_repr(x), want, "{x}");
        }
    }

    #[test]
    fn dumps_equals_json_dumps() {
        let v = Value::from_pairs([
            ("b", Value::from(vec![Value::from(1), Value::from(1e-05)])),
            ("a", Value::from("é \"q\" \\ \n\t\u{1}")),
            ("c", Value::from(())),
            ("d", Value::from(true)),
            ("e", Value::from(Vec::<Value>::new())),
        ]);
        let o = Dumps::default();
        assert_eq!(
            dumps(&v, &o).unwrap(),
            r#"{"b": [1, 1e-05], "a": "é \"q\" \\ \n\t\u0001", "c": null, "d": true, "e": []}"#
        );
        let o = Dumps {
            indent: Some("  ".into()),
            item_separator: ",".into(),
            sort_keys: true,
            ensure_ascii: true,
            ..Dumps::default()
        };
        assert_eq!(
            dumps(&v, &o).unwrap(),
            "{\n  \"a\": \"\\u00e9 \\\"q\\\" \\\\ \\n\\t\\u0001\",\n  \"b\": [\n    1,\n    1e-05\n  ],\n  \"c\": null,\n  \"d\": true,\n  \"e\": []\n}"
        );
        assert_eq!(
            dumps(
                &Value::from("🌍"),
                &Dumps {
                    ensure_ascii: true,
                    ..Dumps::default()
                }
            )
            .unwrap(),
            "\"\\ud83c\\udf0d\""
        );
    }

    #[test]
    fn containers_print_as_python_str() {
        let v = Value::from(vec![
            Value::from("a"),
            Value::from(1),
            Value::from("it's"),
            Value::from(true),
            Value::from(()),
            Value::from_pairs([("k", 0.5)]),
        ]);
        assert_eq!(
            py_str(&v).unwrap(),
            r#"['a', 1, "it's", True, None, {'k': 0.5}]"#
        );
        assert_eq!(py_str(&Value::from(1e-05)).unwrap(), "1e-05");
        assert_eq!(py_str(&Value::from("x")), None);
        let mut s = String::new();
        string_repr("a'\"\u{7}\u{2028}é", &mut s);
        assert_eq!(s, "'a\\'\"\\x07\\u2028é'");
    }

    #[test]
    fn strftime_formats_utc_dates() {
        // 2026-11-08 12:00:00 UTC, a Sunday.
        let t = UNIX_EPOCH + Duration::from_secs(1_794_139_200);
        assert_eq!(
            strftime("%d %b %Y %H:%M:%S %A %a %B %m %y %j %%", t).unwrap(),
            "08 Nov 2026 12:00:00 Sunday Sun November 11 26 312 %"
        );
        assert_eq!(strftime("%Y-%m-%d", UNIX_EPOCH).unwrap(), "1970-01-01");
        assert!(strftime("%Z", t).is_err());
    }
}
