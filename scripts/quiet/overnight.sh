#!/usr/bin/env bash
# The unattended night: wait until the machine is quiet (load below 3 for 10 minutes, at
# most 6 hours; nothing runs if it never is), then the quiet session and the measurements that need no password.
# Logs to target/quiet/overnight-<timestamp>.log.
set -u
cd "$(dirname "$0")/../.."
LOG=$PWD/target/quiet/overnight-$(date +%Y%m%d-%H%M).log
mkdir -p "$(dirname "$LOG")"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
caffeinate -dimsu -w $$ &
quiet=0
for _ in $(seq 72); do
    load=$(sysctl -n vm.loadavg | awk '{print $2}')
    if awk "BEGIN{exit !($load < 3)}"; then quiet=$((quiet + 1)); else quiet=0; fi
    log "load $load (quiet for $quiet × 5 min)"
    [ $quiet -ge 2 ] && break
    sleep 300
done
[ $quiet -ge 2 ] || { log "never quiet: nothing measured"; exit 1; }
log "starting the quiet session"
scripts/quiet_session.sh 2>&1 | tee -a "$LOG"
log "starting the manual measurements"
scripts/quiet/manual.sh 2>&1 | tee -a "$LOG"
log "overnight done"
