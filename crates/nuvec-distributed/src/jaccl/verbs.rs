//! The system RDMA library (`librdma.dylib`, ibverbs), loaded at run time as MLX's JACCL
//! does. The declarations mirror the macOS SDK's `infiniband/verbs.h` (checked by
//! `tests/jaccl_layout.rs`); only the fields nuvec reads or writes are named.
//!
//! `ibv_post_send`, `ibv_post_recv` and `ibv_poll_cq` are inline functions in the header
//! that call through the context's operation table; the bindings do the same.

#![allow(unsafe_code)]

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::net::Ipv4Addr;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use nuvec_core::{Error, Result};

use super::endpoint::{Completion, Destination, Endpoint, Fabric, Sequence, Status};
use super::frame::Frame;
use super::{PortState, RdmaDevice};

/// `union ibv_gid` (16 bytes, aligned as its `uint64_t` view).
#[repr(C, align(8))]
#[derive(Clone, Copy, Default)]
pub(crate) struct Gid(pub(crate) [u8; 16]);

/// `struct ibv_wc`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct Wc {
    pub(crate) wr_id: u64,
    pub(crate) status: c_int,
    pub(crate) opcode: c_int,
    pub(crate) vendor_err: u32,
    pub(crate) byte_len: u32,
    pub(crate) imm_data: u32,
    pub(crate) qp_num: u32,
    pub(crate) src_qp: u32,
    pub(crate) wc_flags: u32,
    pub(crate) pkey_index: u16,
    pub(crate) slid: u16,
    pub(crate) sl: u8,
    pub(crate) dlid_path_bits: u8,
}

/// `struct ibv_sge`.
#[repr(C)]
pub(crate) struct Sge {
    pub(crate) addr: u64,
    pub(crate) length: u32,
    pub(crate) lkey: u32,
}

/// `struct ibv_send_wr`: the RDMA, atomic and UD union (`wr`, 32 bytes), the XRC union and
/// the memory-window/TSO union are unused by SEND and kept zero.
#[repr(C)]
pub(crate) struct SendWr {
    pub(crate) wr_id: u64,
    pub(crate) next: *mut SendWr,
    pub(crate) sg_list: *mut Sge,
    pub(crate) num_sge: c_int,
    pub(crate) opcode: c_int,
    pub(crate) send_flags: u32,
    pub(crate) imm_data: u32,
    pub(crate) wr: [u64; 4],
    pub(crate) qp_type: u32,
    pub(crate) _pad: u32,
    pub(crate) bind_mw: [u64; 6],
}

/// `struct ibv_recv_wr`.
#[repr(C)]
pub(crate) struct RecvWr {
    pub(crate) wr_id: u64,
    pub(crate) next: *mut RecvWr,
    pub(crate) sg_list: *mut Sge,
    pub(crate) num_sge: c_int,
}

/// `struct ibv_qp_cap`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct QpCap {
    pub(crate) max_send_wr: u32,
    pub(crate) max_recv_wr: u32,
    pub(crate) max_send_sge: u32,
    pub(crate) max_recv_sge: u32,
    pub(crate) max_inline_data: u32,
}

/// `struct ibv_qp_init_attr`.
#[repr(C)]
pub(crate) struct QpInitAttr {
    pub(crate) qp_context: *mut c_void,
    pub(crate) send_cq: *mut Cq,
    pub(crate) recv_cq: *mut Cq,
    pub(crate) srq: *mut c_void,
    pub(crate) cap: QpCap,
    pub(crate) qp_type: c_int,
    pub(crate) sq_sig_all: c_int,
}

/// `struct ibv_global_route`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct GlobalRoute {
    pub(crate) dgid: Gid,
    pub(crate) flow_label: u32,
    pub(crate) sgid_index: u8,
    pub(crate) hop_limit: u8,
    pub(crate) traffic_class: u8,
}

/// `struct ibv_ah_attr`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct AhAttr {
    pub(crate) grh: GlobalRoute,
    pub(crate) dlid: u16,
    pub(crate) sl: u8,
    pub(crate) src_path_bits: u8,
    pub(crate) static_rate: u8,
    pub(crate) is_global: u8,
    pub(crate) port_num: u8,
}

/// `struct ibv_qp_attr`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct QpAttr {
    pub(crate) qp_state: c_int,
    pub(crate) cur_qp_state: c_int,
    pub(crate) path_mtu: c_int,
    pub(crate) path_mig_state: c_int,
    pub(crate) qkey: u32,
    pub(crate) rq_psn: u32,
    pub(crate) sq_psn: u32,
    pub(crate) dest_qp_num: u32,
    pub(crate) qp_access_flags: u32,
    pub(crate) cap: QpCap,
    pub(crate) ah_attr: AhAttr,
    pub(crate) alt_ah_attr: AhAttr,
    pub(crate) pkey_index: u16,
    pub(crate) alt_pkey_index: u16,
    pub(crate) en_sqd_async_notify: u8,
    pub(crate) sq_draining: u8,
    pub(crate) max_rd_atomic: u8,
    pub(crate) max_dest_rd_atomic: u8,
    pub(crate) min_rnr_timer: u8,
    pub(crate) port_num: u8,
    pub(crate) timeout: u8,
    pub(crate) retry_cnt: u8,
    pub(crate) rnr_retry: u8,
    pub(crate) alt_port_num: u8,
    pub(crate) alt_timeout: u8,
    pub(crate) rate_limit: u32,
}

/// `struct ibv_port_attr`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct PortAttr {
    pub(crate) state: c_int,
    pub(crate) max_mtu: c_int,
    pub(crate) active_mtu: c_int,
    pub(crate) gid_tbl_len: c_int,
    pub(crate) port_cap_flags: u32,
    pub(crate) max_msg_sz: u32,
    pub(crate) bad_pkey_cntr: u32,
    pub(crate) qkey_viol_cntr: u32,
    pub(crate) pkey_tbl_len: u16,
    pub(crate) lid: u16,
    pub(crate) sm_lid: u16,
    pub(crate) lmc: u8,
    pub(crate) max_vl_num: u8,
    pub(crate) sm_sl: u8,
    pub(crate) subnet_timeout: u8,
    pub(crate) init_type_reply: u8,
    pub(crate) active_width: u8,
    pub(crate) active_speed: u8,
    pub(crate) phys_state: u8,
    pub(crate) link_layer: u8,
    pub(crate) flags: u8,
    pub(crate) port_cap_flags2: u16,
}

type PollCq = unsafe extern "C" fn(*mut Cq, c_int, *mut Wc) -> c_int;
type PostSend = unsafe extern "C" fn(*mut Qp, *mut SendWr, *mut *mut SendWr) -> c_int;
type PostRecv = unsafe extern "C" fn(*mut Qp, *mut RecvWr, *mut *mut RecvWr) -> c_int;

/// `struct ibv_context_ops` (32 function pointers; `poll_cq` is the 12th, `post_send` and
/// `post_recv` the 26th and 27th).
#[repr(C)]
pub(crate) struct ContextOps {
    pub(crate) _before_poll: [*const c_void; 11],
    pub(crate) poll_cq: Option<PollCq>,
    pub(crate) _before_post: [*const c_void; 13],
    pub(crate) post_send: Option<PostSend>,
    pub(crate) post_recv: Option<PostRecv>,
    pub(crate) _after: [*const c_void; 5],
}

/// The start of `struct ibv_context` (its device and operation table).
#[repr(C)]
pub(crate) struct Context {
    pub(crate) device: *mut c_void,
    pub(crate) ops: ContextOps,
}

/// The start of `struct ibv_qp`.
#[repr(C)]
pub(crate) struct Qp {
    pub(crate) context: *mut Context,
    pub(crate) qp_context: *mut c_void,
    pub(crate) pd: *mut c_void,
    pub(crate) send_cq: *mut Cq,
    pub(crate) recv_cq: *mut Cq,
    pub(crate) srq: *mut c_void,
    pub(crate) handle: u32,
    pub(crate) qp_num: u32,
}

/// The start of `struct ibv_cq`.
#[repr(C)]
pub(crate) struct Cq {
    pub(crate) context: *mut Context,
}

/// `struct ibv_mr`.
#[repr(C)]
pub(crate) struct Mr {
    pub(crate) context: *mut Context,
    pub(crate) pd: *mut c_void,
    pub(crate) addr: *mut c_void,
    pub(crate) length: usize,
    pub(crate) handle: u32,
    pub(crate) lkey: u32,
    pub(crate) rkey: u32,
}

/// The sizes and field offsets the bindings rely on, for the layout test against the SDK
/// header (`C name`, `size or offset`).
#[doc(hidden)]
pub fn layout() -> Vec<(&'static str, usize)> {
    use std::mem::{offset_of, size_of};
    vec![
        ("sizeof(union ibv_gid)", size_of::<Gid>()),
        ("sizeof(struct ibv_wc)", size_of::<Wc>()),
        ("offsetof(struct ibv_wc, status)", offset_of!(Wc, status)),
        (
            "offsetof(struct ibv_wc, byte_len)",
            offset_of!(Wc, byte_len),
        ),
        ("sizeof(struct ibv_sge)", size_of::<Sge>()),
        ("offsetof(struct ibv_sge, lkey)", offset_of!(Sge, lkey)),
        ("sizeof(struct ibv_send_wr)", size_of::<SendWr>()),
        (
            "offsetof(struct ibv_send_wr, sg_list)",
            offset_of!(SendWr, sg_list),
        ),
        (
            "offsetof(struct ibv_send_wr, opcode)",
            offset_of!(SendWr, opcode),
        ),
        (
            "offsetof(struct ibv_send_wr, send_flags)",
            offset_of!(SendWr, send_flags),
        ),
        ("offsetof(struct ibv_send_wr, wr)", offset_of!(SendWr, wr)),
        (
            "offsetof(struct ibv_send_wr, qp_type)",
            offset_of!(SendWr, qp_type),
        ),
        (
            "offsetof(struct ibv_send_wr, bind_mw)",
            offset_of!(SendWr, bind_mw),
        ),
        ("sizeof(struct ibv_recv_wr)", size_of::<RecvWr>()),
        (
            "offsetof(struct ibv_recv_wr, num_sge)",
            offset_of!(RecvWr, num_sge),
        ),
        ("sizeof(struct ibv_qp_init_attr)", size_of::<QpInitAttr>()),
        (
            "offsetof(struct ibv_qp_init_attr, cap)",
            offset_of!(QpInitAttr, cap),
        ),
        (
            "offsetof(struct ibv_qp_init_attr, qp_type)",
            offset_of!(QpInitAttr, qp_type),
        ),
        (
            "offsetof(struct ibv_qp_init_attr, sq_sig_all)",
            offset_of!(QpInitAttr, sq_sig_all),
        ),
        ("sizeof(struct ibv_global_route)", size_of::<GlobalRoute>()),
        (
            "offsetof(struct ibv_global_route, sgid_index)",
            offset_of!(GlobalRoute, sgid_index),
        ),
        (
            "offsetof(struct ibv_global_route, hop_limit)",
            offset_of!(GlobalRoute, hop_limit),
        ),
        ("sizeof(struct ibv_ah_attr)", size_of::<AhAttr>()),
        (
            "offsetof(struct ibv_ah_attr, dlid)",
            offset_of!(AhAttr, dlid),
        ),
        (
            "offsetof(struct ibv_ah_attr, is_global)",
            offset_of!(AhAttr, is_global),
        ),
        (
            "offsetof(struct ibv_ah_attr, port_num)",
            offset_of!(AhAttr, port_num),
        ),
        ("sizeof(struct ibv_qp_attr)", size_of::<QpAttr>()),
        (
            "offsetof(struct ibv_qp_attr, path_mtu)",
            offset_of!(QpAttr, path_mtu),
        ),
        (
            "offsetof(struct ibv_qp_attr, rq_psn)",
            offset_of!(QpAttr, rq_psn),
        ),
        (
            "offsetof(struct ibv_qp_attr, sq_psn)",
            offset_of!(QpAttr, sq_psn),
        ),
        (
            "offsetof(struct ibv_qp_attr, dest_qp_num)",
            offset_of!(QpAttr, dest_qp_num),
        ),
        (
            "offsetof(struct ibv_qp_attr, qp_access_flags)",
            offset_of!(QpAttr, qp_access_flags),
        ),
        (
            "offsetof(struct ibv_qp_attr, ah_attr)",
            offset_of!(QpAttr, ah_attr),
        ),
        (
            "offsetof(struct ibv_qp_attr, pkey_index)",
            offset_of!(QpAttr, pkey_index),
        ),
        (
            "offsetof(struct ibv_qp_attr, port_num)",
            offset_of!(QpAttr, port_num),
        ),
        (
            "offsetof(struct ibv_qp_attr, rate_limit)",
            offset_of!(QpAttr, rate_limit),
        ),
        ("sizeof(struct ibv_port_attr)", size_of::<PortAttr>()),
        (
            "offsetof(struct ibv_port_attr, gid_tbl_len)",
            offset_of!(PortAttr, gid_tbl_len),
        ),
        (
            "offsetof(struct ibv_port_attr, lid)",
            offset_of!(PortAttr, lid),
        ),
        (
            "offsetof(struct ibv_port_attr, port_cap_flags2)",
            offset_of!(PortAttr, port_cap_flags2),
        ),
        ("sizeof(struct ibv_context_ops)", size_of::<ContextOps>()),
        (
            "offsetof(struct ibv_context, ops)",
            offset_of!(Context, ops),
        ),
        (
            "offsetof(struct ibv_context, ops.poll_cq)",
            offset_of!(Context, ops) + offset_of!(ContextOps, poll_cq),
        ),
        (
            "offsetof(struct ibv_context, ops.post_send)",
            offset_of!(Context, ops) + offset_of!(ContextOps, post_send),
        ),
        (
            "offsetof(struct ibv_context, ops.post_recv)",
            offset_of!(Context, ops) + offset_of!(ContextOps, post_recv),
        ),
        ("offsetof(struct ibv_qp, context)", offset_of!(Qp, context)),
        ("offsetof(struct ibv_qp, qp_num)", offset_of!(Qp, qp_num)),
        ("offsetof(struct ibv_cq, context)", offset_of!(Cq, context)),
        ("sizeof(struct ibv_mr)", size_of::<Mr>()),
        ("offsetof(struct ibv_mr, lkey)", offset_of!(Mr, lkey)),
        ("IBV_QPT_UC", QPT_UC as usize),
        ("IBV_QPS_INIT", QPS_INIT as usize),
        ("IBV_QPS_RTR", QPS_RTR as usize),
        ("IBV_QPS_RTS", QPS_RTS as usize),
        ("IBV_MTU_1024", MTU_1024 as usize),
        (
            "IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ",
            ACCESS as usize,
        ),
        ("IBV_QP_STATE", QP_STATE as usize),
        ("IBV_QP_ACCESS_FLAGS", QP_ACCESS_FLAGS as usize),
        ("IBV_QP_PKEY_INDEX", QP_PKEY_INDEX as usize),
        ("IBV_QP_PORT", QP_PORT as usize),
        ("IBV_QP_AV", QP_AV as usize),
        ("IBV_QP_PATH_MTU", QP_PATH_MTU as usize),
        ("IBV_QP_RQ_PSN", QP_RQ_PSN as usize),
        ("IBV_QP_SQ_PSN", QP_SQ_PSN as usize),
        ("IBV_QP_DEST_QPN", QP_DEST_QPN as usize),
        ("IBV_WR_SEND_WITH_IMM", WR_SEND_WITH_IMM as usize),
        ("IBV_SEND_SIGNALED", SEND_SIGNALED as usize),
        ("IBV_WC_WITH_IMM", WC_WITH_IMM as usize),
        ("IBV_PORT_ACTIVE", 4),
    ]
}

const QPT_UC: c_int = 3;
const QPS_INIT: c_int = 1;
const QPS_RTR: c_int = 2;
const QPS_RTS: c_int = 3;
const MTU_1024: c_int = 3;
const ACCESS: u32 = 1 | 2 | 4; // local write, remote write, remote read
const QP_STATE: c_int = 1;
const QP_ACCESS_FLAGS: c_int = 1 << 3;
const QP_PKEY_INDEX: c_int = 1 << 4;
const QP_PORT: c_int = 1 << 5;
const QP_AV: c_int = 1 << 7;
const QP_PATH_MTU: c_int = 1 << 8;
const QP_RQ_PSN: c_int = 1 << 12;
const QP_SQ_PSN: c_int = 1 << 16;
const QP_DEST_QPN: c_int = 1 << 20;
/// `IBV_WR_SEND_WITH_IMM`: a SEND whose immediate data is the sequence number.
const WR_SEND_WITH_IMM: c_int = 3;
/// `IBV_WC_WITH_IMM`.
const WC_WITH_IMM: u32 = 1 << 1;
const SEND_SIGNALED: u32 = 2;
/// MLX's queue depths and packet sequence number.
const MAX_WR: u32 = 32;
const PSN: u32 = 7;

/// The library's entry points (MLX's `IBVWrapper`).
struct Lib {
    get_device_list: unsafe extern "C" fn(*mut c_int) -> *mut *mut c_void,
    get_device_name: unsafe extern "C" fn(*mut c_void) -> *const c_char,
    open_device: unsafe extern "C" fn(*mut c_void) -> *mut Context,
    free_device_list: unsafe extern "C" fn(*mut *mut c_void),
    close_device: unsafe extern "C" fn(*mut Context) -> c_int,
    alloc_pd: unsafe extern "C" fn(*mut Context) -> *mut c_void,
    create_qp: unsafe extern "C" fn(*mut c_void, *mut QpInitAttr) -> *mut Qp,
    create_cq:
        unsafe extern "C" fn(*mut Context, c_int, *mut c_void, *mut c_void, c_int) -> *mut Cq,
    destroy_cq: unsafe extern "C" fn(*mut Cq) -> c_int,
    destroy_qp: unsafe extern "C" fn(*mut Qp) -> c_int,
    dealloc_pd: unsafe extern "C" fn(*mut c_void) -> c_int,
    query_port: unsafe extern "C" fn(*mut Context, u8, *mut PortAttr) -> c_int,
    query_gid: unsafe extern "C" fn(*mut Context, u8, c_int, *mut Gid) -> c_int,
    modify_qp: unsafe extern "C" fn(*mut Qp, *mut QpAttr, c_int) -> c_int,
    reg_mr: unsafe extern "C" fn(*mut c_void, *mut c_void, usize, c_int) -> *mut Mr,
    dereg_mr: unsafe extern "C" fn(*mut Mr) -> c_int,
    wc_status_str: unsafe extern "C" fn(c_int) -> *const c_char,
}

// SAFETY: the table holds function pointers of a library that is never unloaded.
unsafe impl Send for Lib {}
// SAFETY: as above.
unsafe impl Sync for Lib {}

fn lib() -> Result<&'static Lib> {
    static LIB: OnceLock<std::result::Result<Lib, String>> = OnceLock::new();
    LIB.get_or_init(load).as_ref().map_err(|e| {
        Error::Backend(format!(
            "jaccl: RDMA is not available ({e}; the library ships with macOS 26.2 and later)"
        ))
    })
}

// Each transmute's target is its field's type, declared in `Lib`.
#[allow(clippy::missing_transmute_annotations)]
fn load() -> std::result::Result<Lib, String> {
    // SAFETY: a NUL-terminated name; the handle is kept for the process's lifetime.
    let handle = unsafe {
        libc::dlopen(
            c"librdma.dylib".as_ptr(),
            libc::RTLD_NOW | libc::RTLD_GLOBAL,
        )
    };
    if handle.is_null() {
        return Err("librdma.dylib not found".into());
    }
    let sym = |name: &str| -> std::result::Result<*mut c_void, String> {
        let c = CString::new(name).expect("no NUL in symbol names");
        // SAFETY: a valid handle and a NUL-terminated name.
        let p = unsafe { libc::dlsym(handle, c.as_ptr()) };
        if p.is_null() {
            Err(format!("librdma.dylib has no {name}"))
        } else {
            Ok(p)
        }
    };
    macro_rules! load {
        ($($field:ident = $name:literal),* $(,)?) => {
            Lib {
                // SAFETY: each symbol is the function of that name in verbs.h, whose C
                // signature the field's type declares.
                $($field: unsafe { std::mem::transmute::<*mut c_void, _>(sym($name)?) },)*
            }
        };
    }
    Ok(load! {
        get_device_list = "ibv_get_device_list",
        get_device_name = "ibv_get_device_name",
        open_device = "ibv_open_device",
        free_device_list = "ibv_free_device_list",
        close_device = "ibv_close_device",
        alloc_pd = "ibv_alloc_pd",
        create_qp = "ibv_create_qp",
        create_cq = "ibv_create_cq",
        destroy_cq = "ibv_destroy_cq",
        destroy_qp = "ibv_destroy_qp",
        dealloc_pd = "ibv_dealloc_pd",
        query_port = "ibv_query_port",
        query_gid = "ibv_query_gid",
        modify_qp = "ibv_modify_qp",
        reg_mr = "ibv_reg_mr",
        dereg_mr = "ibv_dereg_mr",
        wc_status_str = "ibv_wc_status_str",
    })
}

/// Entry `i` of a device list of more than `i` entries, and its name.
fn device_entry(lib: &Lib, list: *mut *mut c_void, i: usize) -> (*mut c_void, String) {
    // SAFETY: the caller's list has more than `i` entries, so entry `i` is in bounds.
    let at = unsafe { list.add(i) };
    // SAFETY: an initialised entry of the live list.
    let dev = unsafe { *at };
    // SAFETY: a device of the live list; its name is library-owned.
    let name = unsafe { (lib.get_device_name)(dev) };
    // SAFETY: a NUL-terminated string that lives as long as the list.
    let name = unsafe { CStr::from_ptr(name) };
    (dev, name.to_string_lossy().into_owned())
}

/// An open device context, closed on drop.
struct Device {
    lib: &'static Lib,
    ctx: *mut Context,
}

impl Drop for Device {
    fn drop(&mut self) {
        // SAFETY: `ctx` came from `open_device` and is closed once, after every object on it.
        unsafe { (self.lib.close_device)(self.ctx) };
    }
}

impl Device {
    /// Opens the device named `name` (MLX's `create_connections`).
    fn open(lib: &'static Lib, name: &str) -> Result<Self> {
        let mut n: c_int = 0;
        // SAFETY: `n` receives the count; the list is freed below.
        let list = unsafe { (lib.get_device_list)(&mut n) };
        if list.is_null() {
            return Err(Error::Backend("jaccl: listing RDMA devices failed".into()));
        }
        let mut found = None;
        for i in 0..usize::try_from(n).unwrap_or(0) {
            let (dev, dev_name) = device_entry(lib, list, i);
            if dev_name == name {
                // SAFETY: an entry of the live list.
                found = Some(unsafe { (lib.open_device)(dev) });
                break;
            }
        }
        // SAFETY: the list from `get_device_list`, freed once; opened contexts outlive it.
        unsafe { (lib.free_device_list)(list) };
        match found {
            Some(ctx) if !ctx.is_null() => Ok(Self { lib, ctx }),
            Some(_) => Err(Error::Backend(format!(
                "jaccl: could not open device {name}"
            ))),
            None => Err(Error::Backend(format!(
                "jaccl: device {name} not found ({n} available)"
            ))),
        }
    }

    fn port(&self) -> Result<PortAttr> {
        let mut attr = PortAttr::default();
        // SAFETY: a live context and a zeroed attribute struct of the header's layout.
        let rc = unsafe { (self.lib.query_port)(self.ctx, 1, &mut attr) };
        if rc != 0 {
            return Err(Error::Backend(format!(
                "jaccl: querying port 1 failed ({rc})"
            )));
        }
        Ok(attr)
    }

    /// The first IPv4-mapped GID (`::ffff:a.b.c.d`) of port 1.
    fn ipv4_gid(&self, port: &PortAttr) -> Option<Gid> {
        (0..port.gid_tbl_len).find_map(|i| {
            let mut gid = Gid::default();
            // SAFETY: a live context; `gid` is 16 writable bytes.
            let rc = unsafe { (self.lib.query_gid)(self.ctx, 1, i, &mut gid) };
            let g = gid.0;
            (rc == 0 && g[..10].iter().all(|&b| b == 0) && g[10] == 0xff && g[11] == 0xff)
                .then_some(gid)
        })
    }
}

/// One RDMA connection end: a protection domain, a completion queue and a UC queue pair on
/// a device (MLX's `Connection`).
struct VerbsEnd {
    lib: &'static Lib,
    device: Device,
    pd: *mut c_void,
    cq: *mut Cq,
    qp: *mut Qp,
    peer: usize,
    regions: Mutex<Vec<(usize, *mut Mr, u32)>>,
    sequence: Sequence,
}

// SAFETY: the verbs objects are used from one thread at a time (the group's order lock and
// one thread per cable), and ibverbs objects may move between threads.
unsafe impl Send for VerbsEnd {}
// SAFETY: as above; `regions` is behind a lock.
unsafe impl Sync for VerbsEnd {}

impl Drop for VerbsEnd {
    fn drop(&mut self) {
        // Each object was created on this end and is destroyed once, in MLX's order (memory
        // regions, queue pair, completion queue, protection domain; the device closes last,
        // in `Device::drop`).
        for &(_, mr, _) in self
            .regions
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
        {
            // SAFETY: a region registered on this end's protection domain.
            unsafe { (self.lib.dereg_mr)(mr) };
        }
        if !self.qp.is_null() {
            // SAFETY: this end's queue pair, no longer used.
            unsafe { (self.lib.destroy_qp)(self.qp) };
        }
        if !self.cq.is_null() {
            // SAFETY: this end's completion queue; its queue pair is gone.
            unsafe { (self.lib.destroy_cq)(self.cq) };
        }
        if !self.pd.is_null() {
            // SAFETY: this end's protection domain; its regions and queue pair are gone.
            unsafe { (self.lib.dealloc_pd)(self.pd) };
        }
    }
}

impl VerbsEnd {
    fn open(lib: &'static Lib, name: &str, peer: usize) -> Result<Self> {
        let device = Device::open(lib, name)?;
        let mut end = Self {
            lib,
            device,
            pd: std::ptr::null_mut(),
            cq: std::ptr::null_mut(),
            qp: std::ptr::null_mut(),
            peer,
            regions: Mutex::new(Vec::new()),
            sequence: Sequence::default(),
        };
        let failed = |what: &str| Error::Backend(format!("jaccl: {name}: {what} failed"));
        let ctx = end.device.ctx;
        // SAFETY: a live context.
        end.pd = unsafe { (lib.alloc_pd)(ctx) };
        if end.pd.is_null() {
            return Err(failed("allocating a protection domain"));
        }
        // SAFETY: a live context; no completion channel.
        end.cq = unsafe {
            (lib.create_cq)(
                ctx,
                (2 * MAX_WR) as c_int,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            )
        };
        if end.cq.is_null() {
            return Err(failed("creating a completion queue"));
        }
        let mut init = QpInitAttr {
            qp_context: ctx.cast(),
            send_cq: end.cq,
            recv_cq: end.cq,
            srq: std::ptr::null_mut(),
            cap: QpCap {
                max_send_wr: MAX_WR,
                max_recv_wr: MAX_WR,
                max_send_sge: 1,
                max_recv_sge: 1,
                max_inline_data: 0,
            },
            qp_type: QPT_UC,
            sq_sig_all: 0,
        };
        // SAFETY: a live protection domain and an attribute struct of the header's layout.
        end.qp = unsafe { (lib.create_qp)(end.pd, &mut init) };
        if end.qp.is_null() {
            return Err(Error::Backend(format!(
                "jaccl: {name}: creating the queue pair failed: {}",
                std::io::Error::last_os_error()
            )));
        }
        let mut attr = QpAttr {
            qp_state: QPS_INIT,
            port_num: 1,
            qp_access_flags: ACCESS,
            ..QpAttr::default()
        };
        end.modify(
            &mut attr,
            QP_STATE | QP_PKEY_INDEX | QP_PORT | QP_ACCESS_FLAGS,
            "INIT",
        )?;
        Ok(end)
    }

    fn modify(&self, attr: &mut QpAttr, mask: c_int, state: &str) -> Result<()> {
        // SAFETY: a live queue pair and an attribute struct of the header's layout.
        let rc = unsafe { (self.lib.modify_qp)(self.qp, attr, mask) };
        if rc != 0 {
            return Err(Error::Backend(format!(
                "jaccl: rank {}: moving the queue pair to {state} failed ({rc})",
                self.peer
            )));
        }
        Ok(())
    }

    fn ops(&self) -> &ContextOps {
        // SAFETY: the context outlives the end; its operation table is set by the library.
        unsafe { &(*self.device.ctx).ops }
    }

    fn lkey(&self, frame: &Frame) -> Result<u32> {
        let regions = self.regions.lock().unwrap_or_else(PoisonError::into_inner);
        regions
            .iter()
            .find(|(addr, _, _)| *addr == frame.as_ptr() as usize)
            .map(|&(_, _, lkey)| lkey)
            .ok_or_else(|| Error::Backend("jaccl: a frame was not registered".into()))
    }
}

impl Endpoint for VerbsEnd {
    fn destination(&self) -> Result<Destination> {
        let port = self.device.port()?;
        let gid = self.device.ipv4_gid(&port).ok_or_else(|| {
            Error::Backend(
                "jaccl: the RDMA device has no IPv4-mapped GID: Thunderbolt RDMA ports publish \
                 one once the interface has an IPv4 address; assign a link-local address, e.g. \
                 ifconfig <interface> inet 169.254.0.1 netmask 255.255.0.0 alias"
                    .into(),
            )
        })?;
        // SAFETY: a live queue pair; `qp_num` is set at creation.
        let qpn = unsafe { (*self.qp).qp_num };
        Ok(Destination {
            lid: i32::from(port.lid),
            qpn: qpn as i32,
            psn: PSN as i32,
            gid: gid.0,
        })
    }

    fn connect(&self, peer: &Destination) -> Result<()> {
        let mut attr = QpAttr {
            qp_state: QPS_RTR,
            path_mtu: MTU_1024,
            rq_psn: peer.psn as u32,
            dest_qp_num: peer.qpn as u32,
            ..QpAttr::default()
        };
        attr.ah_attr.dlid = peer.lid as u16;
        attr.ah_attr.port_num = 1;
        if peer.gid[8..] != [0; 8] {
            attr.ah_attr.is_global = 1;
            attr.ah_attr.grh.hop_limit = 1;
            attr.ah_attr.grh.dgid = Gid(peer.gid);
            attr.ah_attr.grh.sgid_index = 1;
        }
        self.modify(
            &mut attr,
            QP_STATE | QP_AV | QP_PATH_MTU | QP_DEST_QPN | QP_RQ_PSN,
            "RTR",
        )?;
        let mut attr = QpAttr {
            qp_state: QPS_RTS,
            sq_psn: PSN,
            ..QpAttr::default()
        };
        self.modify(&mut attr, QP_STATE | QP_SQ_PSN, "RTS")
    }

    fn register(&self, frame: &Frame) -> Result<()> {
        // SAFETY: a live protection domain; the frame's memory lives as long as the
        // registration (the frames outlive the ends that deregister them).
        let mr = unsafe {
            (self.lib.reg_mr)(self.pd, frame.as_ptr().cast(), frame.len(), ACCESS as c_int)
        };
        if mr.is_null() {
            return Err(Error::Backend("jaccl: registering a frame failed".into()));
        }
        // SAFETY: a memory region the library returned.
        let lkey = unsafe { (*mr).lkey };
        self.regions
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push((frame.as_ptr() as usize, mr, lkey));
        Ok(())
    }

    fn post_send(&self, frame: &Frame, wr_id: u64) -> Result<()> {
        let mut sge = Sge {
            addr: frame.as_ptr() as u64,
            length: frame.len() as u32,
            lkey: self.lkey(frame)?,
        };
        let mut wr = SendWr {
            wr_id,
            next: std::ptr::null_mut(),
            sg_list: &mut sge,
            num_sge: 1,
            opcode: WR_SEND_WITH_IMM,
            send_flags: SEND_SIGNALED,
            imm_data: self.sequence.next_send().to_be(),
            wr: [0; 4],
            qp_type: 0,
            _pad: 0,
            bind_mw: [0; 6],
        };
        let mut bad = std::ptr::null_mut();
        let post = self
            .ops()
            .post_send
            .ok_or_else(|| Error::Backend("jaccl: no post_send".into()))?;
        // SAFETY: a live queue pair; `wr` and `sge` outlive the call (the library copies
        // them); the frame is registered and stays alive until its completion.
        let rc = unsafe { post(self.qp, &mut wr, &mut bad) };
        if rc != 0 {
            return Err(Error::Backend(format!(
                "jaccl: rank {}: send failed ({rc})",
                self.peer
            )));
        }
        Ok(())
    }

    fn post_recv(&self, frame: &Frame, wr_id: u64) -> Result<()> {
        let mut sge = Sge {
            addr: frame.as_ptr() as u64,
            length: frame.len() as u32,
            lkey: self.lkey(frame)?,
        };
        let mut wr = RecvWr {
            wr_id,
            next: std::ptr::null_mut(),
            sg_list: &mut sge,
            num_sge: 1,
        };
        let mut bad = std::ptr::null_mut();
        let post = self
            .ops()
            .post_recv
            .ok_or_else(|| Error::Backend("jaccl: no post_recv".into()))?;
        // SAFETY: as in `post_send`; the NIC writes the frame until its completion.
        let rc = unsafe { post(self.qp, &mut wr, &mut bad) };
        if rc != 0 {
            return Err(Error::Backend(format!(
                "jaccl: rank {}: receive failed ({rc})",
                self.peer
            )));
        }
        Ok(())
    }

    fn poll(&self, out: &mut Vec<Completion>, max: usize) -> Result<usize> {
        let mut wc = [Wc::default(); 32];
        let max = max.min(wc.len());
        let poll = self
            .ops()
            .poll_cq
            .ok_or_else(|| Error::Backend("jaccl: no poll_cq".into()))?;
        // SAFETY: a live completion queue and room for `max` completions.
        let n = unsafe { poll(self.cq, max as c_int, wc.as_mut_ptr()) };
        let n = usize::try_from(n).map_err(|_| {
            Error::Backend(format!(
                "jaccl: rank {}: polling completions failed ({n})",
                self.peer
            ))
        })?;
        for c in &wc[..n] {
            let status = if c.status == 0 {
                Status::Success
            } else {
                // SAFETY: returns the library's static name of a status code.
                let name = unsafe { (self.lib.wc_status_str)(c.status) };
                // SAFETY: a static NUL-terminated string.
                let name = unsafe { CStr::from_ptr(name) };
                Status::Error {
                    code: c.status,
                    name: name.to_string_lossy().into_owned(),
                }
            };
            let sequence = (c.wc_flags & WC_WITH_IMM != 0).then(|| u32::from_be(c.imm_data));
            let mut c = Completion {
                wr_id: c.wr_id,
                status,
                sequence,
            };
            self.sequence.check(&mut c);
            out.push(c);
        }
        Ok(n)
    }
}

/// The system RDMA library as a fabric.
pub(crate) struct Verbs(&'static Lib);

impl Verbs {
    pub(crate) fn load() -> Result<Self> {
        lib().map(Self)
    }
}

impl Fabric for Verbs {
    fn open(&self, device: &str, peer: usize) -> Result<Arc<dyn Endpoint>> {
        Ok(Arc::new(VerbsEnd::open(self.0, device, peer)?))
    }
}

/// This Mac's RDMA devices, their port 1's state and IPv4-mapped address.
pub(crate) fn devices() -> Result<Vec<RdmaDevice>> {
    let lib = lib()?;
    let mut n: c_int = 0;
    // SAFETY: `n` receives the count; the list is freed below.
    let list = unsafe { (lib.get_device_list)(&mut n) };
    if list.is_null() {
        return Err(Error::Backend("jaccl: listing RDMA devices failed".into()));
    }
    let names: Vec<String> = (0..usize::try_from(n).unwrap_or(0))
        .map(|i| device_entry(lib, list, i).1)
        .collect();
    // SAFETY: the list from `get_device_list`, freed once.
    unsafe { (lib.free_device_list)(list) };
    names
        .into_iter()
        .map(|name| {
            let d = Device::open(lib, &name)?;
            let port = d.port()?;
            let ipv4 = d
                .ipv4_gid(&port)
                .map(|g| Ipv4Addr::new(g.0[12], g.0[13], g.0[14], g.0[15]));
            let port_state = match port.state {
                1 => PortState::Down,
                2 => PortState::Init,
                3 => PortState::Armed,
                4 => PortState::Active,
                5 => PortState::ActiveDefer,
                s => PortState::Unknown(s),
            };
            Ok(RdmaDevice {
                name,
                port_state,
                ipv4,
            })
        })
        .collect()
}
