//! TCP keepalive on every connection, so a peer that vanished without closing (a laptop
//! asleep, a cable pulled) is reported within about 15 s: probes after 5 s idle, 3 s apart,
//! 3 of them.

#![allow(unsafe_code)]

use std::net::TcpStream;

/// Seconds idle before the first probe, between probes, and probes before giving up.
pub(crate) const IDLE: i32 = 5;
pub(crate) const INTERVAL: i32 = 3;
pub(crate) const PROBES: i32 = 3;

#[cfg(target_os = "macos")]
const IDLE_OPTION: libc::c_int = libc::TCP_KEEPALIVE;
#[cfg(all(unix, not(target_os = "macos")))]
const IDLE_OPTION: libc::c_int = libc::TCP_KEEPIDLE;

#[cfg(unix)]
fn set(
    s: &TcpStream,
    level: libc::c_int,
    option: libc::c_int,
    value: libc::c_int,
) -> std::io::Result<()> {
    use std::os::fd::AsRawFd;
    // SAFETY: a live socket's descriptor; `value` is a `c_int` of the size passed.
    let rc = unsafe {
        libc::setsockopt(
            s.as_raw_fd(),
            level,
            option,
            std::ptr::from_ref(&value).cast(),
            size_of::<libc::c_int>() as libc::socklen_t,
        )
    };
    if rc == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

/// Turns keepalive on with the probes above.
#[cfg(unix)]
pub(crate) fn enable(s: &TcpStream) -> std::io::Result<()> {
    set(s, libc::SOL_SOCKET, libc::SO_KEEPALIVE, 1)?;
    set(s, libc::IPPROTO_TCP, IDLE_OPTION, IDLE)?;
    set(s, libc::IPPROTO_TCP, libc::TCP_KEEPINTVL, INTERVAL)?;
    set(s, libc::IPPROTO_TCP, libc::TCP_KEEPCNT, PROBES)
}

#[cfg(not(unix))]
pub(crate) fn enable(_: &TcpStream) -> std::io::Result<()> {
    Ok(())
}

/// The socket's keepalive settings: (on, idle, interval, probes).
#[cfg(all(test, unix))]
pub(crate) fn settings(s: &TcpStream) -> (i32, i32, i32, i32) {
    use std::os::fd::AsRawFd;
    let get = |level, option| {
        let mut v: libc::c_int = 0;
        let mut len = size_of::<libc::c_int>() as libc::socklen_t;
        // SAFETY: a live socket's descriptor and a `c_int` with its length.
        let rc = unsafe {
            libc::getsockopt(
                s.as_raw_fd(),
                level,
                option,
                std::ptr::from_mut(&mut v).cast(),
                &mut len,
            )
        };
        assert_eq!(rc, 0);
        v
    };
    (
        get(libc::SOL_SOCKET, libc::SO_KEEPALIVE),
        get(libc::IPPROTO_TCP, IDLE_OPTION),
        get(libc::IPPROTO_TCP, libc::TCP_KEEPINTVL),
        get(libc::IPPROTO_TCP, libc::TCP_KEEPCNT),
    )
}

#[cfg(all(test, unix))]
mod tests {
    use std::net::{TcpListener, TcpStream};

    #[test]
    fn keepalive_reads_back() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let s = TcpStream::connect(l.local_addr().unwrap()).unwrap();
        super::enable(&s).unwrap();
        let (on, idle, interval, probes) = super::settings(&s);
        assert!(on != 0);
        assert_eq!(
            (idle, interval, probes),
            (super::IDLE, super::INTERVAL, super::PROBES)
        );
    }
}
