# Short Image Index — Schema (Phase 5)

A **short index** is a compact, human-friendly identifier for a generated image
(e.g. `IISDSXS3`). It lets follow-up commands refer to an image as `@IISDSXS3`
instead of a long timestamped path. The short index is the companion to the
Phase 4 W3C PROV provenance records — it is derived from, and keyed against, the
PROV record for each image.

## Format

| Property   | Value |
|------------|-------|
| Alphabet   | RFC 4648 Base32 — `A`–`Z` and `2`–`7` (uppercase) |
| Length     | 8 characters by default (`INDEX_LENGTH`) |
| Excluded   | `0`, `1`, `8`, `9` and lowercase — avoids ambiguity when read aloud/transcribed |
| Example    | `IISDSXS3` |

## Derivation

```
index = base32( sha256( prov_entity_id ) )[:8]
```

The key is the image's **PROV entity id** (`provenance.ProvenanceRecord.entity_id`),
which is itself derived from `output_path` + `timestamp`. Consequences:

- **Deterministic** — the same provenance record always yields the same index.
- **PROV-keyed** — the index is one-to-one with a provenance record, so `lookup`
  can join back to the full PROV metadata (prompt, model, provider, seed, …).
- **Regeneration-safe** — regenerating to the same path produces a new timestamp,
  hence a new entity id and a distinct index.

## Uniqueness & persistence

- Uniqueness is guaranteed **within an output directory**.
- Each directory holds an index file, `.image-index.json`, mapping
  `index -> image filename`:

  ```json
  {
    "IISDSXS3": "20260707-120000-a-red-barn-9f3a.png"
  }
  ```

- The file persists on disk, so indices **survive across sessions**.
- On the (hash-)improbable collision between two *different* images, the newcomer's
  index is lengthened one Base32 character at a time until unique.

## Usage

```bash
# Generation prints the index for each image:
image-creator generate "a red barn" --provider gemini
#   ✓ /…/20260707-120000-a-red-barn-9f3a.png [@IISDSXS3] (2.1s)

# Reference it in edit / reference options instead of a path:
image-creator generate "make it snowy" --edit @IISDSXS3 --provider gemini

# Look it up:
image-creator lookup IISDSXS3        # path + provenance metadata
image-creator lookup --list          # recent indices with images
```

## Implementation

| Concern            | Location |
|--------------------|----------|
| Index domain logic | `src/image_creator_tool/indexer.py` |
| Lookup CLI command | `src/image_creator_tool/commands/lookup.py` |
| Generation wiring  | `src/image_creator_tool/generation_core.py` (`_record_provenance`) |
| `@INDEX` expansion  | `src/image_creator_tool/cli.py` (`generate` command) |
