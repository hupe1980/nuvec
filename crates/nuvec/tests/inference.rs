//! Which closure forms the tree transforms accept: `trybuild` cases over
//! the transforms' signatures with stub types (`tests/ui/inference/stub`), so the rules are
//! pinned independently of the implementation. Annotated closures, `?` inside closures
//! returning a generic `R: IntoResult`, and `in_axes((None, 0))` compile; unannotated
//! closure parameters fail with E0282, and `checkpoint` rejects borrowing closures.

#[test]
fn closure_inference() {
    let t = trybuild::TestCases::new();
    t.pass("tests/ui/inference/pass_*.rs");
    t.compile_fail("tests/ui/inference/fail_*.rs");
}
