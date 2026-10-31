+++
title = "Distributed"
description = "nuvec::distributed runs one program across processes or Macs with MLX's collectives (all_sum, all_gather, send, recv) over MLX's TCP ring, and splits linear layers for tensor parallelism."
weight = 115
[extra]
group = "Libraries"
+++

`nuvec::distributed` is MLX's `mx.distributed`: a group of processes, each with a rank, runs the same program, and collective operations combine their arrays. The collectives are ordinary lazy operations: they run when their result is evaluated, and they compose with `grad`, `vmap` and `compile`.

## Launching a group

Every process calls `init`, which reads MLX's launch environment: `MLX_HOSTFILE`, a JSON list with one entry per rank of `"host:port"` addresses, and `MLX_RANK`. The processes can share one Mac (over loopback) or run one per Mac (over Ethernet or Thunderbolt IP).

```json
[["10.0.0.1:5000"], ["10.0.0.2:5000"]]
```

Each address is one TCP connection to that rank's ring neighbour; list several per rank to spread large transfers over more connections. Without the environment, `init(false)` returns the group of this process alone, where every collective is the identity, so the same code runs on one machine unchanged.

```rust
use nuvec::prelude::*;
use nuvec::distributed;

# fn main() -> Result<()> {
let group = distributed::init(false)?;
let x = array![1.0, 2.0, 3.0];
let total = distributed::all_sum(&x, None); // the element-wise sum over every rank
let all = distributed::all_gather(&x, None); // every rank's x, in rank order
assert_eq!(total.shape(), [3]);
assert_eq!(all.shape(), [3 * group.size() as i32]);
# Ok(())
# }
```

## Collectives

| Operation | Result |
|---|---|
| `all_sum`, `all_max`, `all_min` | The element-wise reduction over the group; every rank gets the same bits, and reruns repeat them |
| `all_gather` | Every rank's array, concatenated along the first axis in rank order |
| `send(x, dst)`, `recv(shape, dtype, src)`, `recv_like(x, src)` | Point to point between ring neighbours; a receive of another dtype or shape is an error |
| `sum_gradients(x)` | `x`, with its gradient summed over the group |

Each takes the group as its last argument (`None`: the default group, which `init` sets). Collectives run on the CPU over the arrays' memory; on the GPU they wait for the work producing their inputs.

## Gradients and compile

As in MLX, the gradient of `all_sum` passes the cotangent through (each rank's loss is its own), and the gradient of `all_gather` is this rank's rows. A compiled function with collectives communicates on every call. For data-parallel training, average the gradients with `all_sum` before the optimizer step.

## Tensor-parallel layers

`nuvec_nn::sharded` splits a linear layer across a group as `mlx.nn` does: `AllToShardedLinear` keeps each rank's slice of the output features, and `ShardedToAllLinear` each rank's slice of the input features, summing the partial products with one `all_sum`. A pair of them splits an MLP (or attention's projections by heads) with one collective per pair. Both convert full-precision and quantized layers, and an uneven split is an error naming the layer.

```rust
use nuvec::distributed::Group;
use nuvec::{array, random};
use nuvec_nn::Linear;
use nuvec_nn::sharded::{AllToShardedLinear, ShardedToAllLinear};

# fn main() -> nuvec::Result<()> {
random::seed(0);
let group = Group::single();
let up = AllToShardedLinear::from_linear(&Linear::new(4, 16), &group)?;
let down = ShardedToAllLinear::from_linear(&Linear::new(16, 4), &group)?;
let y = down.forward(&up.forward(&array![[1.0, 2.0, 3.0, 4.0]]));
assert_eq!(y.shape(), [1, 4]);
# Ok(())
# }
```

## Failures

A peer that never starts, or exits during a collective, fails every pending operation with an error naming its rank within the timeout (`NUVEC_DISTRIBUTED_TIMEOUT`, in seconds, default 30). Nothing hangs or panics.

## Performance

`just bench-distributed` times `all_sum` against MLX's ring backend on the same Mac. On an M4 Pro over loopback nuvec takes 0.36–0.78× MLX's time with 2 ranks and 0.17–0.66× with 4, from 4 KB to 256 MB. RDMA over Thunderbolt 5 (MLX's JACCL backend) is not supported.
