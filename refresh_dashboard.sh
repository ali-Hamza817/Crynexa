#!/bin/bash
cd /home/administrator/Desktop/Crynexa
exec 9>/tmp/crynexa_refresh.lock
flock -n 9 || exit 0          # never run two copies
while true; do
  python3 build_web.py >/dev/null 2>&1
  python3 dashboard.py >/dev/null 2>&1
  cp -f reports/dashboard.html web/dashboard.html 2>/dev/null
  sleep 45
done
