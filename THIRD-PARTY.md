# Third-Party Notices — `execlave-sdk`

This package bundles no third-party code directly. Its runtime dependencies, installed transitively through PyPI, retain their original licences. The full list is reproduced here for convenience; the authoritative source is each dependency's PyPI page.

## Required runtime dependencies

| Package    | Licence    | Purpose                                    |
| ---------- | ---------- | ------------------------------------------ |
| `requests` | Apache-2.0 | HTTP transport to the Execlave API         |
| `pydantic` | MIT        | Input validation on the public SDK surface |

## Optional runtime dependencies

Installed only when the user opts into an extras group.

| Extras            | Package                                  | Licence    |
| ----------------- | ---------------------------------------- | ---------- |
| `[otel]`          | `opentelemetry-api`, `opentelemetry-sdk` | Apache-2.0 |
| `[langchain]`     | `langchain-core`                         | MIT        |
| `[openai-agents]` | `openai-agents`                          | Apache-2.0 |
| `[crewai]`        | `crewai`                                 | MIT        |

## Development dependencies

Not shipped with the published wheel. See `pyproject.toml` under `[project.optional-dependencies].test`.

## Licence of this package

`execlave-sdk` itself is released under the **MIT licence** — see `LICENSE`.

## Attribution updates

If you believe a dependency is missing from this notice or incorrectly attributed, please open an issue or email `support@execlave.com`.
