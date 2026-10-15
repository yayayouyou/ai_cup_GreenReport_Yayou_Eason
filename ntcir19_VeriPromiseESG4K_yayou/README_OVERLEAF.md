# Building this paper on Overleaf

**Set the compiler to LuaLaTeX** (Menu → Compiler → LuaLaTeX). XeLaTeX also works.
The default, pdfLaTeX, will compile but every Chinese character becomes the placeholder `[zh]`
— usable for drafting, **not** for the submitted PDF.

## What to upload

    main.tex
    bibliography.tex
    refs.bib
    sections/          (01-09)
    fonts/             (NotoSerifTC-paper-subset.ttf, LICENSE-NOTE.txt)

`acmart` is already on Overleaf; nothing else needs installing.

## Checks on the built PDF

| | expected |
|---|---|
| pages | **8**, including references — this is a hard limit |
| Chinese | Table 4's quoted paragraphs and Section 6 render as real characters, not `[zh]` |
| fonts | all embedded (14) |
| undefined references | 0 |

Run LuaLaTeX **twice** — the first pass writes the cross-references, the second resolves them.

## The font

`fonts/NotoSerifTC-paper-subset.ttf` is a subset of Noto Serif TC (SIL Open Font License 1.1)
containing only the characters this paper quotes. See `fonts/LICENSE-NOTE.txt`. It is loaded by
`\newfontfamily` in the preamble, which is why a Unicode engine is required.
