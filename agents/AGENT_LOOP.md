# NeurIPS Paper Refinement Loop

An iterative multi-agent loop for refining academic papers to NeurIPS acceptance quality.

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│                        NEURIPS PAPER REFINEMENT LOOP                               │
├────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │                         PHASE 1: QUALITY CONTROL                            │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐            │  │
│  │  │    LLM     │  │  SPELL-    │  │  CITATION  │  │   CLAIM    │            │  │
│  │  │  DETECTOR  │  │   CHECK    │  │   CHECK    │  │   VERIFY   │            │  │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘            │  │
│  │         │              │               │               │                    │  │
│  │         └──────────────┴───────────────┴───────────────┘                    │  │
│  │                              │                                              │  │
│  │                    ┌─────────▼─────────┐                                    │  │
│  │                    │  QUALITY REPORT   │                                    │  │
│  │                    └───────────────────┘                                    │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                              │
│  ┌─────────────────────────────────▼───────────────────────────────────────────┐  │
│  │                         PHASE 2: REVIEW                                     │  │
│  │                                                                             │  │
│  │    ┌──────────────────────────────────────────────────────┐                 │  │
│  │    │              REVIEWER POOL (10 personas)             │                 │  │
│  │    │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐    │                 │  │
│  │    │  │ ML  │ │ RL  │ │ NOV │ │ CLR │ │ REP │ │ ETH │    │                 │  │
│  │    │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘    │                 │  │
│  │    │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                    │                 │  │
│  │    │  │ IND │ │ JR  │ │ SKP │ │ MAS │                    │                 │  │
│  │    │  └─────┘ └─────┘ └─────┘ └─────┘                    │                 │  │
│  │    └──────────────────────────────────────────────────────┘                 │  │
│  │                              │                                              │  │
│  │                     SAMPLE 3 REVIEWERS                                      │  │
│  │                              │                                              │  │
│  │    ┌────────────┐   ┌───────▼────────┐   ┌────────────┐                    │  │
│  │    │ REVIEWER 1 │   │  REVIEWER 2    │   │ REVIEWER 3 │                    │  │
│  │    │  (sampled) │   │   (sampled)    │   │  (sampled) │                    │  │
│  │    └────────────┘   └────────────────┘   └────────────┘                    │  │
│  │          │                  │                   │                          │  │
│  │          └──────────────────┼───────────────────┘                          │  │
│  │                             │                                              │  │
│  │                    ┌────────▼────────┐                                     │  │
│  │                    │  3 REVIEWS      │                                     │  │
│  │                    └─────────────────┘                                     │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                              │
│  ┌─────────────────────────────────▼───────────────────────────────────────────┐  │
│  │                         PHASE 3: META-REVIEW                                │  │
│  │                                                                             │  │
│  │                    ┌─────────────────────┐                                  │  │
│  │                    │    META-REVIEWER    │                                  │  │
│  │                    │    (Area Chair)     │                                  │  │
│  │                    └─────────────────────┘                                  │  │
│  │                              │                                              │  │
│  │              ┌───────────────┼───────────────┐                              │  │
│  │              ▼               ▼               ▼                              │  │
│  │       ┌──────────┐    ┌──────────┐    ┌──────────┐                         │  │
│  │       │ DECISION │    │ PRIORITY │    │ ACTION   │                         │  │
│  │       │          │    │  ISSUES  │    │  ITEMS   │                         │  │
│  │       └──────────┘    └──────────┘    └──────────┘                         │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                              │
│                          ┌─────────▼─────────┐                                    │
│                          │  ACCEPT/CONTINUE? │                                    │
│                          └───────────────────┘                                    │
│                                    │                                              │
│                     ┌──────────────┴──────────────┐                               │
│                     │                             │                               │
│              ┌──────▼──────┐              ┌───────▼───────┐                       │
│              │    EXIT     │              │   CONTINUE    │                       │
│              │  (Success)  │              │    (Loop)     │                       │
│              └─────────────┘              └───────────────┘                       │
│                                                  │                                │
│  ┌───────────────────────────────────────────────▼─────────────────────────────┐  │
│  │                         PHASE 4: IMPROVE                                    │  │
│  │                                                                             │  │
│  │                    ┌─────────────────────┐                                  │  │
│  │                    │    IMPROVER AGENT   │                                  │  │
│  │                    │  (applies fixes)    │                                  │  │
│  │                    └─────────────────────┘                                  │  │
│  │                              │                                              │  │
│  │                    ┌─────────▼─────────┐                                    │  │
│  │                    │  MODIFIED .tex    │                                    │  │
│  │                    └───────────────────┘                                    │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                              │
│  ┌─────────────────────────────────▼───────────────────────────────────────────┐  │
│  │                         PHASE 5: COMPILE & VERIFY                           │  │
│  │                                                                             │  │
│  │    ┌────────────┐        ┌────────────┐        ┌────────────┐              │  │
│  │    │  pdflatex  │───────▶│   bibtex   │───────▶│  pdflatex  │ (x2)        │  │
│  │    └────────────┘        └────────────┘        └────────────┘              │  │
│  │                                                       │                     │  │
│  │                                              ┌────────▼────────┐            │  │
│  │                                              │  VISUAL AGENT   │            │  │
│  │                                              │  (PDF inspect)  │            │  │
│  │                                              └─────────────────┘            │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                              │
│                                    │                                              │
│                          ┌─────────▼─────────┐                                    │
│                          │    LOOP BACK      │────────────────────────────┐       │
│                          │  to PHASE 1       │                            │       │
│                          └───────────────────┘                            │       │
│                                                                           │       │
│                                    ┌──────────────────────────────────────┘       │
│                                    ▼                                              │
└────────────────────────────────────────────────────────────────────────────────────┘
```

## Inputs

```yaml
TEX_FILE: root.tex                    # Main paper file
BIB_FILE: bib.bib                     # Bibliography
STYLE_GUIDE: agents/STYLE_GUIDE.md    # Writing style guide
REVIEWER_POOL: agents/REVIEWER_POOL.md # Reviewer personas
MAX_ITERATIONS: 5                     # Maximum loop iterations
TARGET_SCORE: 6.5                     # Minimum weighted average for accept
PAGE_LIMIT: 9                         # NeurIPS page limit (main text)
ANONYMOUS: true                       # Check for anonymity
SOURCE_PAPERS: []                     # Optional: PDFs for claim verification
```

## Loop Execution

### Phase 1: Quality Control (Parallel)

Run all quality agents in parallel:

**1.1 LLM Detector**
- Agent: `AGENT_LLM_DETECTOR.md`
- Input: TEX_FILE, STYLE_GUIDE
- Output: `output/iter_N/llm_report.md`

**1.2 Spellcheck**
- Agent: `AGENT_SPELLCHECK.md`
- Input: TEX_FILE, BIB_FILE
- Output: `output/iter_N/spell_report.md`

**1.3 Citation Check**
- Agent: `AGENT_CITATION_CHECK.md`
- Input: TEX_FILE, BIB_FILE
- Output: `output/iter_N/citation_report.md`

**1.4 Claim Verification**
- Agent: `AGENT_CLAIM_VERIFY.md`
- Input: TEX_FILE, BIB_FILE, SOURCE_PAPERS
- Output: `output/iter_N/claim_report.md`

**Aggregate:** Combine into `quality_report.md`

---

### Phase 2: Review (Parallel)

**2.1 Sample Reviewers**
- Randomly select 3 reviewers from REVIEWER_POOL
- Ensure diversity: at least 1 technical, 1 domain, 1 presentation-focused
- Log selected reviewers

**2.2 Generate Reviews**
Run 3 reviewers in parallel:
- Each reviewer follows their persona from REVIEWER_POOL.md
- Input: TEX_FILE, BIB_FILE, quality_report.md
- Output: `output/iter_N/review_1.md`, `review_2.md`, `review_3.md`

**Review format:**
- Summary (2-3 sentences)
- Strengths (3-5 points)
- Weaknesses (3-5 points with line references)
- Questions for authors
- Scores (Soundness, Contribution, Presentation, Overall, Confidence)
- Recommendation

---

### Phase 3: Meta-Review

**3.1 Synthesize**
- Agent: `AGENT_METAREVIEWER.md`
- Input: All reviews, quality_report.md, iteration number
- Output: `output/iter_N/meta_review.md`

**3.2 Decision**
Evaluate termination conditions:

```python
def should_exit(meta_review, iteration):
    if iteration >= MAX_ITERATIONS:
        return True, "MAX_ITERATIONS reached"

    if meta_review.all_scores >= 6 and meta_review.avg_score >= TARGET_SCORE:
        return True, "ACCEPT: All scores meet threshold"

    if meta_review.decision == "REJECT" and meta_review.has_fatal_flaw:
        return True, "REJECT: Fatal flaw identified"

    if meta_review.no_improvement_from_last:
        return True, "STALLED: No progress in last iteration"

    return False, "CONTINUE: Issues remain fixable"
```

---

### Phase 4: Improve

**4.1 Apply Fixes**
- Agent: `AGENT_IMPROVER.md`
- Input: TEX_FILE, meta_review.md, STYLE_GUIDE
- Output: Modified TEX_FILE, `output/iter_N/changes.md`

**4.2 Track Changes**
- Log all edits with line numbers
- Snapshot pre-edit version
- Track which issues were addressed

---

### Phase 5: Compile & Verify

**5.1 Compile**
```bash
pdflatex root.tex
bibtex root
pdflatex root.tex
pdflatex root.tex
```

**5.2 Check Compilation**
- If errors: revert changes, log error, try alternative fixes
- If warnings: log for visual inspection

**5.3 Visual Inspection**
- Agent: `AGENT_VISUAL.md`
- Input: Compiled PDF, PAGE_LIMIT, ANONYMOUS
- Output: `output/iter_N/visual_report.md`

**5.4 Aggregate Issues**
- Combine visual issues with any remaining meta-review issues
- Feed into next iteration

---

### Phase 6: Loop or Exit

**If EXIT:**
- Generate final summary
- Move best version to `output/final/`
- Report final scores and decision

**If CONTINUE:**
- Increment iteration counter
- Update input files
- Return to Phase 1

---

## Termination Conditions

### Success (ACCEPT)
All of:
- [ ] Weighted average score >= 6.5
- [ ] All individual scores >= 6
- [ ] No critical issues remaining
- [ ] Quality checks pass
- [ ] Visual inspection passes

### Good Enough (CONDITIONAL ACCEPT)
All of:
- [ ] Weighted average score >= 6.0
- [ ] No scores below 5
- [ ] All Priority 1 issues addressed
- [ ] MAX_ITERATIONS reached

### Failure (REJECT)
Any of:
- [ ] Fatal flaw identified (cannot be fixed)
- [ ] Any score <= 3 with high confidence
- [ ] Weighted average < 5.0 after MAX_ITERATIONS

### Stalled (HUMAN NEEDED)
- Same issues flagged 2+ consecutive iterations
- Improver unable to make progress
- Conflicting fixes required

---

## State Tracking

Each iteration produces:
```
output/
├── iter_1/
│   ├── quality/
│   │   ├── llm_report.md
│   │   ├── spell_report.md
│   │   ├── citation_report.md
│   │   └── claim_report.md
│   ├── reviews/
│   │   ├── reviewers_sampled.json
│   │   ├── review_1.md
│   │   ├── review_2.md
│   │   └── review_3.md
│   ├── meta_review.md
│   ├── changes.md
│   ├── visual_report.md
│   ├── root_snapshot.tex      # Pre-edit snapshot
│   └── iteration_summary.md
├── iter_2/
│   └── ...
├── iter_N/
│   └── ...
└── final/
    ├── root.tex               # Final version
    ├── root.pdf               # Final PDF
    ├── summary.md             # Full refinement summary
    └── scores_history.json    # Score progression
```

---

## Score Tracking

Track score progression across iterations:

```json
{
  "iterations": [
    {
      "iteration": 1,
      "reviewers": ["Prof_Methods", "Dr_Industry", "PhD_Student"],
      "scores": {
        "reviewer_1": {"overall": 5, "confidence": 4},
        "reviewer_2": {"overall": 6, "confidence": 3},
        "reviewer_3": {"overall": 4, "confidence": 2}
      },
      "weighted_avg": 5.1,
      "issues_fixed": 0,
      "issues_remaining": 12
    },
    {
      "iteration": 2,
      "reviewers": ["Prof_RL", "Dr_Clarity", "Prof_Skeptic"],
      "scores": {
        "reviewer_1": {"overall": 6, "confidence": 5},
        "reviewer_2": {"overall": 7, "confidence": 4},
        "reviewer_3": {"overall": 5, "confidence": 4}
      },
      "weighted_avg": 6.0,
      "issues_fixed": 8,
      "issues_remaining": 5
    }
  ]
}
```

---

## Usage

### Run Full Loop
```bash
claude "Run the NeurIPS paper refinement loop on root.tex. Max 5 iterations. Target score 6.5."
```

### Run Single Phase
```bash
# Just quality check
claude "Run quality control agents on root.tex (LLM detector, spellcheck, citations, claims)"

# Just reviews
claude "Generate 3 NeurIPS reviews for root.tex using the reviewer pool"

# Just meta-review
claude "Generate meta-review from the 3 reviews in output/iter_1/reviews/"
```

### Resume From Iteration
```bash
claude "Resume NeurIPS refinement loop from iteration 3"
```

---

## Configuration Options

```yaml
# Reviewer sampling
reviewer_diversity:
  require_technical: true      # At least 1 technical reviewer
  require_domain: true         # At least 1 domain expert
  require_presentation: true   # At least 1 presentation-focused
  avoid_repeat: true          # Don't repeat reviewers across iterations

# Quality control
quality_gates:
  block_on_critical_llm: true  # Stop if Tier-1 LLM issues found
  block_on_missing_citation: true
  block_on_false_claim: true

# Score thresholds
accept_threshold:
  min_individual: 6
  min_average: 6.5
  max_std: 1.5

# Loop control
loop_control:
  max_iterations: 5
  early_exit_on_accept: true
  stall_detection_window: 2
```

---

## Agent Reference

| Agent | File | Purpose |
|-------|------|---------|
| LLM Detector | `AGENT_LLM_DETECTOR.md` | Detect AI-generated prose |
| Spellcheck | `AGENT_SPELLCHECK.md` | Grammar, spelling, LaTeX |
| Citation Check | `AGENT_CITATION_CHECK.md` | Bibliography verification |
| Claim Verify | `AGENT_CLAIM_VERIFY.md` | Fact-checking |
| Reviewer Pool | `REVIEWER_POOL.md` | 10 reviewer personas |
| Meta-Reviewer | `AGENT_METAREVIEWER.md` | Synthesize reviews, decide |
| Improver | `AGENT_IMPROVER.md` | Apply targeted fixes |
| Visual | `AGENT_VISUAL.md` | PDF inspection |
| Style Guide | `STYLE_GUIDE.md` | Writing standards |

---

## Example Run Output

```markdown
# NeurIPS Refinement Summary

## Paper: The PokéAgent Challenge
## Iterations: 3
## Final Status: ACCEPT

## Score Progression
| Iter | Avg Score | Min Score | Issues |
|------|-----------|-----------|--------|
| 1    | 5.1       | 4         | 12     |
| 2    | 6.0       | 5         | 5      |
| 3    | 6.8       | 6         | 1      |

## Key Improvements Made
1. Removed 23 em-dashes and AI-style phrases
2. Added 5 missing baseline comparisons
3. Fixed 3 citation formatting issues
4. Clarified novelty claims in introduction
5. Added confidence intervals to Table 2

## Final Reviewer Scores
- Prof_Novelty: 7 (Confidence: 4) - "Solid contribution"
- Dr_Repro: 6 (Confidence: 5) - "Reproducible with code"
- Prof_MAS: 7 (Confidence: 4) - "Strong multi-agent work"

## Remaining Minor Issues
1. Line 234: Consider adding runtime comparison
2. Figure 3: Could be larger

## Files
- Final PDF: output/final/root.pdf
- Final .tex: output/final/root.tex
- Full history: output/
```
