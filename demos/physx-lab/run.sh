#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PHYSX_LAB_PORT:-8766}"
command -v php >/dev/null
if [[ ! -f "$DIR/node_modules/three/build/three.module.js" || ! -f "$DIR/node_modules/lucide/dist/umd/lucide.min.js" ]]; then
    printf 'Missing viewer dependencies. Run: cd "%s" && npm ci --ignore-scripts\n' "$DIR" >&2
    exit 1
fi
if [[ ! "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1024 || PORT > 65535 )); then
    printf 'PHYSX_LAB_PORT must be an integer from 1024 to 65535.\n' >&2
    exit 1
fi
printf 'PhysX on Metal: http://127.0.0.1:%s\n' "$PORT"
exec php -S "127.0.0.1:$PORT" -t "$DIR" "$DIR/index.php"
