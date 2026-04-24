# CIF Projection Contract v1

**Status:** Draft
**Applies to:** CIF v2 and later
**Purpose:** Defines the compatibility contract between CIF artifacts and dumb raster codecs.

## Core Principle

CIF is the source. PNG and JPEG are projections.

CIF artifact -> identity
- preview.png: mandatory thumbnail projection
- render.png: optional arbitrary-resolution projection
- future codecs: same artifact, different projection parameters

A dumb PNG/JPEG decoder sees only preview.png.

## Artifact Layer Model

Layer 1 - Identity:
- manifest.json
- receipt.json

Layer 2 - Executable:
- canonical_lms.tensor
- lambda_real.bin
- edges.cifedge
- procedural.json
- inr.siren
- lambda_mod.bin

Layer 3 - Projection:
- preview.png
- render.png

## Projection Rule

Pixels are projections. Artifact digest is identity.

artifact_digest is invariant.
render_digest = f(artifact_digest, width, height).

## Renderer Contract

A CIF-aware renderer verifies the artifact, accepts width and height as render parameters, renders from the executable layer, and writes a projection receipt.

A dumb codec only opens preview.png.
