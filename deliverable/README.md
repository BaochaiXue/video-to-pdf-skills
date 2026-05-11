# Deliverables

This folder is the top-level local handoff area for completed textbook PDFs.

The PDF files here are copied from each run's own `deliverable/` folder or, for older runs that originally only had `build/` outputs, from the final compiled build artifact. These PDFs are intentionally ignored by Git because some are large local handoff files. Keep this README as the tracked index.

| Course / Run | PDF |
|--------------|-----|
| CME295 Fall 2025 | `cme295_complete_notes.pdf` |
| CS224R Spring 2025 | `cs224r_complete_notes.pdf` |
| CS231N Spring 2025 | `cs231n_complete_notes.pdf` |
| UCB CS294/194-196 Agentic AI Fall 2025 | `agentic_ai_complete_notes.pdf` |
| CS294/194-280 SP25 Agents Textbook | `cs294_194_280_sp25_agents_textbook_complete_notes.pdf` |
| CS336 | `cs336_complete_notes.pdf` |
| MIT RES.9-009 Neuroblox IAP 2025 | `mit_res9_009_iap2025_neuroblox_textbook_complete_notes.pdf` |
| Speech Recognition and Understanding Fall 2023 | `speech_recognition_understanding_fall2023_textbook.pdf` |
| S294-277 Robots That Learn Fall 2024 | `s294_277_complete_textbook.pdf` |

Refresh command:

```bash
mkdir -p deliverable
find runs -type f \( -path '*/deliverable/*.pdf' -o -path '*/deliverable/book/*.pdf' \) \
  ! -path '*/deliverable/lectures/*' \
  ! -path '*/lectures/*/deliverable/*' \
  -print0 | while IFS= read -r -d '' f; do cp -f "$f" deliverable/; done

# Older completed runs whose final PDFs predate the standard deliverable layout.
cp -f runs/cme295_fall2025/build/cme295_complete_notes.pdf deliverable/
cp -f runs/cs224r_spring2025/build/cs224r_complete_notes.pdf deliverable/
```
