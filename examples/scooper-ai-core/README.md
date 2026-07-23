# Scooper `ai_core` reference (not part of Agentomatic)

Copied from a production project for migration reference. Generic batteries
(artifacts, task progress, JSON repair, local_npz, ingestion text helpers,
audit, language) now live in `src/agentomatic/`. Domain-specific estimation
math, schemas, Cosmos adapters, and Scooper telemetry stay **out of core** —
keep them in your project package if you need them.

See the Unreleased section of `CHANGELOG.md` and the User Guide for the
migrated APIs.
