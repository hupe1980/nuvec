#!/usr/bin/env bash
# Style gate of the array API: array operations return arrays (failures are sticky and
# surface at `check()`, `eval()`, `item()` or `to_vec()`), so user-facing code never applies
# `?` to an op. Fails if the README, the site, the crate-level docs or a crate example
# contains `?` directly after a call of a public array op or method.
#
# Usage: scripts/check_no_op_try.sh [files…] (default: the user-facing files above).

set -euo pipefail
cd "$(dirname "$0")/.."

# Every public array-producing op, method and constructor (the `ops` table, indexing,
# creation, `fast`, `random`, quantization). Boundaries (`eval`, `item`, `to_vec`, `check`,
# I/O, loading, transforms) are not listed: `?` belongs there.
ops='add|subtract|multiply|divide|maximum|minimum|logaddexp|equal|not_equal|less|less_equal'
ops+='|greater|greater_equal|negative|abs|exp|log|sqrt|sin|cos|tanh|sigmoid|erfinv|sign'
ops+='|conjugate|select|where_|astype|stop_gradient|contiguous|matmul|addmm'
ops+='|sum|mean|max|min|prod|all|any|logsumexp|var|std|argmax|argmin|softmax'
ops+='|sum_keepdims|mean_keepdims|max_keepdims|min_keepdims|prod_keepdims|all_keepdims'
ops+='|any_keepdims|logsumexp_keepdims|var_keepdims|std_keepdims|argmax_keepdims'
ops+='|argmin_keepdims|var_with|std_with'
ops+='|cumsum|cumprod|cummax|cummin|logcumsumexp|cumsum_with|cumprod_with|cummax_with'
ops+='|cummin_with|logcumsumexp_with|sort|argsort|partition|argpartition|topk'
ops+='|reshape|transpose|broadcast_to|expand_dims|squeeze|repeat|split|split_n|split_at'
ops+='|concatenate|stack|take|take_along_axis|put_along_axis|scatter_add_along_axis'
ops+='|slice_update|i|at|set|t|from_slice|zeros|ones|full|arange|zeros_like|ones_like'
ops+='|full_like|quantize|dequantize|quantized_matmul|gather_mm|gather_qmm|conv1d'
ops+='|rms_norm|add_rms_norm|layer_norm|rope|norm_rope|cross_entropy'
ops+='|scaled_dot_product_attention|paged_attention|paged_write|gated_delta'
ops+='|gated_delta_steps|gated_delta_slots|uniform|normal|bits|gumbel|categorical|randint'
ops+='|bernoulli|next_key'

if [ "$#" -gt 0 ]; then
    files=("$@")
else
    files=(README.md)
    while IFS= read -r f; do files+=("$f"); done < <(
        find site/content -name '*.md' 2>/dev/null
        ls crates/*/src/lib.rs
        ls crates/*/examples/*.rs 2>/dev/null
    )
fi

# A call `name(…)` (one level of nested parentheses, possibly a `::<T>` turbofish)
# immediately followed by `?`.
# (`Option::transpose()?`, with no argument, is not the array op.)
pattern="(?<![A-Za-z0-9_])(?!transpose\(\))(?:${ops})(?:::<[^>]*>)?\((?:[^()]|\((?:[^()]|\([^()]*\))*\))*\)\?"

bad=0
for f in "${files[@]}"; do
    if perl -0777 -ne '
        BEGIN { $pat = shift @ARGV; $file = $ARGV[0]; }
        while (/$pat/g) {
            my $pre = substr($_, 0, $-[0]);
            my $line = ($pre =~ tr/\n//) + 1;
            (my $hit = $&) =~ s/\s+/ /g;
            print "$ARGV:$line: $hit\n";
            $found = 1;
        }
        END { exit($found ? 1 : 0) }
    ' "$pattern" "$f"; then
        :
    else
        bad=1
    fi
done

if [ "$bad" -ne 0 ]; then
    echo "✗ \`?\` applied to an array op (above): ops return arrays; use check()/eval() at the boundary" >&2
    exit 1
fi
echo "✓ no \`?\` on array ops in user-facing code"
