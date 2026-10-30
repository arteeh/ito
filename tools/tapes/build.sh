#!/bin/bash
# build.sh REF: release.ps1 -Local of a ref of ~/Projects/ito/main under the GPU lock
# (C:\Users\me\Projects\ito\builds\<ref with / as ->). Refuses from 19:55 and while USER-ON-CERES.
set -u
ref=$1
while true; do
    [ "$(date +%H%M)" -ge 1955 ] && { echo "not starting: Ceres is the user's from 19:55"; exit 75; }
    flock ~/.unii/work/ceres-session.lock bash -c '
        [ -e "$HOME/.unii/work/USER-ON-CERES" ] && exit 75
        echo "$(date +%T) building '"$ref"'"
        cd /mnt/c/Users/me/Projects/ito && powershell.exe -NoProfile -ExecutionPolicy Bypass -File release.ps1 -Branch '"$ref"' -Local < /dev/null > /dev/null 2>&1
        rc=$?; [ $rc = 75 ] && rc=76; exit $rc'
    rc=$?
    [ $rc = 75 ] && { sleep 30; continue; }
    break
done
name=${ref//\//-}
echo "$(date +%T) rc=$rc"; tail -2 "/mnt/c/Users/me/Projects/ito/builds/$name.log"
