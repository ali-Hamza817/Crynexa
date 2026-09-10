#!/bin/bash
# Serve the results dashboard on localhost only.
# View from a remote machine via the VS Code PORTS panel, or:
#   ssh -L 8010:localhost:8010 <user>@<host>
# then open http://localhost:8010/dashboard.html
PORT="${1:-8010}"
cd /home/administrator/Desktop/Crynexa
pkill -f "http.server $PORT" 2>/dev/null
python3 dashboard.py
echo "serving http://localhost:$PORT/dashboard.html  (localhost-only)"
exec python3 -m http.server "$PORT" --bind 127.0.0.1 --directory reports
