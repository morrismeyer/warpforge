#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SETTINGS_FILE="${ROOT_DIR}/settings.gradle"

if [[ ! -f "${SETTINGS_FILE}" ]]; then
  echo "ERROR: settings.gradle not found at: ${SETTINGS_FILE}" >&2
  exit 1
fi

PLUGIN_ID="org.gradle.toolchains.foojay-resolver-convention"
PLUGIN_LINE="id 'org.gradle.toolchains.foojay-resolver-convention' version '1.0.0'"

if grep -q "${PLUGIN_ID}" "${SETTINGS_FILE}"; then
  echo "Foojay resolver convention plugin already present in settings.gradle"
  exit 0
fi

tmp="$(mktemp)"
inserted=0
in_block_comment=0

while IFS= read -r line; do
  if [[ ${inserted} -eq 0 ]]; then
    # block comment start
    if [[ ${in_block_comment} -eq 0 && "${line}" =~ ^[[:space:]]*/\* ]]; then
      in_block_comment=1
      echo "${line}" >> "${tmp}"
      continue
    fi
    # inside block comment
    if [[ ${in_block_comment} -eq 1 ]]; then
      echo "${line}" >> "${tmp}"
      if [[ "${line}" =~ \*/[[:space:]]*$ ]]; then
        in_block_comment=0
      fi
      continue
    fi
    # line comment or blank
    if [[ "${line}" =~ ^[[:space:]]*// ]] || [[ "${line}" =~ ^[[:space:]]*$ ]]; then
      echo "${line}" >> "${tmp}"
      continue
    fi

    # first real statement, insert plugins block before it
    cat >> "${tmp}" <<EOF
plugins {
  ${PLUGIN_LINE}
}

EOF
    inserted=1
  fi

  echo "${line}" >> "${tmp}"
done < "${SETTINGS_FILE}"

if [[ ${inserted} -eq 0 ]]; then
  cat >> "${tmp}" <<EOF
plugins {
  ${PLUGIN_LINE}
}

EOF
fi

cp "${tmp}" "${SETTINGS_FILE}"
rm -f "${tmp}"

echo "Inserted Foojay resolver convention plugin into settings.gradle"
