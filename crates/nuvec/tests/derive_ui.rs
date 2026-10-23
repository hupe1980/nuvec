//! Compile-time behaviour of `#[derive(Tree)]`, `#[derive(Module)]` and `#[derive(Static)]`
//! (`trybuild`): accepted edge cases and the documented error messages.

#[test]
fn derive_ui() {
    let t = trybuild::TestCases::new();
    t.pass("tests/ui/derive/pass_*.rs");
    t.compile_fail("tests/ui/derive/fail_*.rs");
}
