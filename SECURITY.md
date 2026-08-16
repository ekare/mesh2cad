# Security policy

## Supported versions

Only the latest release on the default branch receives security fixes while the project is in
alpha.

## Reporting a vulnerability

Do not open a public issue for a vulnerability that could expose files, execute code, exhaust a
machine, or compromise a release. Use GitHub's private security-advisory reporting feature for the
repository. Include the affected version, platform, minimal reproduction, impact, and any proposed
mitigation. Do not attach proprietary or personal CAD data; use a synthetic reproducer.

The local web application binds to `127.0.0.1` by design and is not hardened as an internet-facing
multi-user service. Exposing it through a reverse proxy or changing its bind address is outside the
supported threat model.
