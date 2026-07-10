# Changelog

## 0.2.0 — 2026-07-09

### Added

- New commands: `lookup` (resolve short-index `@N` references), `contact-sheet` (assemble a contact sheet from image indices, with optional badges), `style` (save/load/apply reusable style presets), `prov` (W3C PROV provenance tracking/export)
- New providers: Azure OpenAI, HuggingFace, LiteLLM
- Short-index (`@N`) reference resolution across commands
- W3C PROV provenance tracking subsystem
- Style save/load library with multi-image group style extraction and closed-loop style refinement
- Generation options: `--style`, `--no-badges`, `--contact-badge-radius`
- Built-in presets: `neon-noir`, `editorial-light`
- Docs: `docs/short-index-schema.md`

### Changed

- SKILL relocated to `skills/image-creator/SKILL.md`
- Removed sunsetting Imagen 4 aliases from the Vertex provider

### Fixed

- API-key redaction: inactive-profile API key no longer leaks to a `-p`-selected provider (security fix)
- Removed stray Vertex `region` kwarg

## 0.1.1 — 2026-06-04

- Fix: `init` command no longer corrupts `[profile.*]` sections in config.toml
- Fix: gracefully handle malformed profile data in config
- Updated `config.example.toml` with profile examples for all providers

## 0.1.0 — 2026-05-28

Initial public release.
