# Security Policy

## Supported Versions

Security fixes are provided for the latest `main` branch and the latest tagged release.

| Version | Supported |
| --- | --- |
| `main` (latest commit) | Yes |
| Latest release tag | Yes |
| Older releases | Best effort |

## Reporting a Vulnerability

Please do not open public issues for security vulnerabilities.

1. Email the maintainer directly with subject: `Security report: <short title>`.
2. Include reproduction steps, impact, and any proof-of-concept details.
3. Include commit hash or release version where the issue was observed.

Expected response targets:

- Initial acknowledgment: within 3 business days
- Triage decision: within 7 business days
- Patch or mitigation plan: as soon as safely possible

## Coordinated Disclosure

- We prefer coordinated disclosure.
- Please allow time to patch before public disclosure.
- We will credit reporters in release notes unless anonymity is requested.

## Scope

In scope examples:

- Authentication and authorization flaws
- Secret handling and credential exposure
- Unsafe deserialization and command execution paths
- Dependency vulnerabilities with exploit paths in this repository
- CI/CD workflow privilege escalation or secret exfiltration risk

Out of scope examples:

- Requests for support or configuration help
- Feature requests without a security impact
- Vulnerabilities only present in unsupported custom forks
