//! The two socket calls std does not offer: waiting for either of two sockets, and peeking
//! without blocking whatever the socket's mode.

#![allow(unsafe_code)]

use std::net::TcpStream;
use std::os::fd::AsRawFd;
use std::time::Duration;

/// Waits until `write` can take bytes or `read` has some (either may be absent), at most
/// `timeout`; `false` if neither became ready.
pub(crate) fn wait(
    write: Option<&TcpStream>,
    read: Option<&TcpStream>,
    timeout: Duration,
) -> std::io::Result<bool> {
    let fd = |s: Option<&TcpStream>| s.map_or(-1, AsRawFd::as_raw_fd);
    let mut fds = [
        libc::pollfd {
            fd: fd(write),
            events: libc::POLLOUT,
            revents: 0,
        },
        libc::pollfd {
            fd: fd(read),
            events: libc::POLLIN,
            revents: 0,
        },
    ];
    let ms = libc::c_int::try_from(timeout.as_millis()).unwrap_or(libc::c_int::MAX);
    loop {
        // SAFETY: `fds` is an array of two `pollfd` for the call's duration (a negative
        // descriptor is ignored).
        let ready = unsafe { libc::poll(fds.as_mut_ptr(), 2, ms) };
        match ready {
            0 => return Ok(false),
            1.. => return Ok(true),
            _ => {
                let e = std::io::Error::last_os_error();
                if e.kind() != std::io::ErrorKind::Interrupted {
                    return Err(e);
                }
            }
        }
    }
}

/// Peeks at one byte without blocking: `Ok(0)` once the peer closed, `Ok(1)` with bytes
/// pending, `WouldBlock` with none.
pub(crate) fn peek(s: &TcpStream) -> std::io::Result<usize> {
    let mut b = 0u8;
    // SAFETY: a live socket's descriptor and a one-byte buffer.
    let n = unsafe {
        libc::recv(
            s.as_raw_fd(),
            (&raw mut b).cast(),
            1,
            libc::MSG_PEEK | libc::MSG_DONTWAIT,
        )
    };
    usize::try_from(n).map_err(|_| std::io::Error::last_os_error())
}
