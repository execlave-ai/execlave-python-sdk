# Security Policy — `execlave-sdk`

## Reporting a Vulnerability

If you discover a security issue in the Python SDK (`execlave-sdk` on PyPI), please report it privately. **Do not open a public GitHub issue for security vulnerabilities.**

Email **security@execlave.com** with:

1. A description of the vulnerability
2. Steps to reproduce
3. Potential impact (confidentiality / integrity / availability)
4. Suggested fix if known
5. The affected `execlave-sdk` version(s)

PGP key available on request.

## Response SLA

| Phase                             | Target                                  |
| --------------------------------- | --------------------------------------- |
| Acknowledgment                    | within 48 hours                         |
| Initial assessment                | within 5 business days                  |
| Fix for confirmed vulnerabilities | within 14 days (coordinated disclosure) |

## Supported Versions

We provide security patches for the latest minor version only. To receive a fix,
upgrade to the latest release.

<!-- Update this table whenever a new minor version is released. -->

| Version | Supported |
| ------- | --------- |
| 1.9.x   | ✅        |
| < 1.9   | ❌        |

## Scope

**In scope:**

- The `execlave-sdk` package published on PyPI
- Source code under `sdk-python/` in the public repository
- Optional integrations: `execlave.integrations.langchain`, `execlave.integrations.openai_agents`, `execlave.integrations.crewai`

**Out of scope for this SDK** (report via [execlave.com/security](https://www.execlave.com/security)):

- Backend API (`api.execlave.com`)
- Dashboard (`www.execlave.com`)
- Infrastructure or hosting

## Safe Harbor

We will not pursue legal action against researchers who:

- Report in good faith
- Avoid privacy violations, destruction of data, and service interruption
- Do not publicly disclose before a coordinated fix is released

## Recognition

Valid reports that lead to a fix are acknowledged in release notes (with your permission) and, for significant findings, may be eligible for a reward through our bug-bounty programme.
