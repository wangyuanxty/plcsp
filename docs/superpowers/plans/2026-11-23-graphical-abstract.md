# Graphical Abstract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Prepare an editable one-page HTML graphical abstract and a matching standalone PDF.

**Architecture:** Use a single HTML document with inline CSS and an embedded crop of manuscript Fig. 1(a). Export with local Chromium printing and check the PDF with Poppler and a physical-size rendering.

**Tech Stack:** HTML, CSS, existing Chromium browser, Poppler.

**Spec:** `docs/superpowers/specs/2026-11-23-graphical-abstract-design.md`

## Global Constraints

- Output to `paper/graphical-abstract.html` and `paper/graphical-abstract.pdf`.
- Use English labels and a landscape canvas at least 1328 CSS px wide.
- Check legibility at 130 mm wide. Do not alter the manuscript or use generative image tools or external uploads.
- Do not commit unless requested. Put conversion intermediates in the system temporary directory and clean them up after verification.

## Review Focus

- The figure crop does not include unrelated panels or omit the transfer route.
- Copy distinguishes joint selection of vehicles from the order in which transport legs occur.
- The 19%–37% range is explicitly relative to the best-performing DRL baseline in each test family; the RGA comparison names both groups.
- A local browser prints a single landscape page without headers, clipping, missing imagery or unexpected font substitution.
- The PDF remains readable at 130 mm width.

### Task 1: Compose and verify the slide

**Files:** Create `paper/graphical-abstract.html` and `paper/graphical-abstract.pdf`.

**Interfaces:** Consume `paper/figures/fig_environment.pdf` and the comparisons reported in `paper/sections/05-experiments.tex`. Produce a self-contained HTML file and a separate printable PDF.

- [ ] Inspect Fig. 1(a) and verify the route and result copy against the manuscript.
- [ ] Embed a high-resolution crop from Fig. 1(a) and compose the three-block slide. Use a print canvas at least 1328 CSS px wide.
- [ ] Print the local HTML with Chromium. Run `pdfinfo` to verify one landscape page and `pdftotext` to confirm the result text is present and selectable.
- [ ] Render the exported PDF at 130 mm viewing width. Inspect the cropped route, text, layout, and comparison qualifiers. Fix any clipping or loss of detail and repeat the checks.
- [ ] Return links to the local HTML and PDF. Do not modify the manuscript or submit the files.
