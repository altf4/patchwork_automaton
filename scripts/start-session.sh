#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
credentials_path="${1:-${repo_dir}/../.creds}"
mkdir -p "${repo_dir}/.local"
chmod 700 "${repo_dir}/.local"
if systemctl --user is-active --quiet patchwork-mtgo-session.service; then
  echo "Session service is already running."
  exit 0
fi
rm -f "${repo_dir}/.local/stop"
systemd-run --user --unit=patchwork-mtgo-session --collect --service-type=exec \
  --property="WorkingDirectory=${repo_dir}" \
  --property="Environment=PYTHONPATH=${repo_dir}/src:${repo_dir}" \
  --property=Restart=on-failure --property=RestartPreventExitStatus=2 \
  --property=RestartSec=30s --property=StartLimitIntervalSec=600 \
  --property=StartLimitBurst=3 \
  /usr/bin/python3 -m examples.idle_session --credentials "${credentials_path}" --run-dir "${repo_dir}/.local"
