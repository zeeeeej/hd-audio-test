#!/usr/bin/env bash
set -euo pipefail

cert_dir=${1:-/data/voice-stack/config/tls}
public_ip=${2:?usage: create_self_signed_cert.sh CERT_DIR PUBLIC_IP}

mkdir -p "$cert_dir"
openssl req -x509 -newkey rsa:3072 -sha256 -nodes -days 365 \
  -keyout "$cert_dir/server.key" \
  -out "$cert_dir/server.crt" \
  -subj "/CN=$public_ip" \
  -addext "subjectAltName=IP:$public_ip"
chmod 600 "$cert_dir/server.key"
chmod 644 "$cert_dir/server.crt"
