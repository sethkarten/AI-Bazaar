# Writing Assistant Orchestrator

A collaborative writing system for drafting NeurIPS papers from early stage to submission-ready.

## Overview

Unlike the Review Loop (which polishes complete drafts), the Writing Assistant helps **during active writing** to:
- Flesh out sections from notes/outlines
- Identify experiment gaps before running experiments
- Plan figures and tables
- Position the paper in related work
- Map claims to evidence

## Architecture

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                         WRITING ASSISTANT SYSTEM                               │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│                        ┌─────────────────────────┐                             │
│                        │    CLAIM-EVIDENCE MAP   │                             │
│                        │  (Foundation for all)   │                             │
│                        └───────────┬─────────────┘                             │
│                                    │                                           │
│          ┌─────────────────────────┼─────────────────────────┐                 │
│          │                         │                         │                 │
│          ▼                         ▼                         ▼                 │
│  ┌───────────────┐        ┌───────────────┐        ┌───────────────┐          │
│  │   EXPERIMENT  │        │    FIGURE     │        │    RELATED    │          │
│  │   GAP AGENT   │        │   PLANNER     │        │  WORK AGENT   │          │
│  │               │        │               │        │               │          │
│  │ "What experi- │        │ "What figures │        │ "What papers  │          │
│  │  ments do we  │        │  and tables   │        │  should we    │          │
│  │  need?"       │        │  do we need?" │        │  cite?"       │          │
│  └───────┬───────┘        └───────┬───────┘        └───────┬───────┘          │
│          │                         │                         │                 │
│          └─────────────────────────┼─────────────────────────┘                 │
│                                    │                                           │
│                                    ▼                                           │
│                        ┌─────────────────────────┐                             │
│                        │    SECTION WRITER       │                             │
│                        │                         │                             │
│                        │ "Help me write this     │                             │
│                        │  section with proper    │                             │
│                        │  evidence and style"    │                             │
│                        └─────────────────────────┘                             │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

## Usage Modes

### Mode 1: Full Analysis (Start of Writing)

Run when starting a paper or major revision:

```bash
claude "Run full writing analysis on root.tex:
1. Map all claims to evidence (AGENT_CLAIM_EVIDENCE_MAP)
2. Identify experiment gaps (AGENT_EXPERIMENT_GAP)
3. Plan figures and tables (AGENT_FIGURE_PLANNER)
4. Analyze related work coverage (AGENT_RELATED_WORK)"
```

**Output:** Comprehensive analysis with prioritized TODO lists

---

### Mode 2: Section Drafting

When you need help writing a specific section:

```bash
claude "Help me write the Introduction section for root.tex.
Use AGENT_SECTION_WRITER with these notes:
- Main contribution: dual-track Pokemon benchmark
- Gap: no unified benchmark for competitive + exploration
- Key results: baseline performance, 300+ participants expected"
```

**Output:** LaTeX-ready section draft

---

### Mode 3: Experiment Planning

When planning what experiments to run:

```bash
claude "Run AGENT_EXPERIMENT_GAP on root.tex.
I have results for: [list what you have]
Identify what experiments are still needed to support the claims."
```

**Output:** Prioritized experiment list with effort estimates

---

### Mode 4: Related Work Deep Dive

When checking literature coverage:

```bash
claude "Run AGENT_RELATED_WORK on root.tex.
Focus on: game AI benchmarks, LLM agents, competition papers.
Identify must-cite papers I'm missing."
```

**Output:** Citation gap analysis with .bib entries

---

### Mode 5: Figure/Table Planning

When planning visualizations:

```bash
claude "Run AGENT_FIGURE_PLANNER on root.tex.
I need to show: main results, ablations, scaling.
What figures and tables should I create?"
```

**Output:** Figure/table specifications with data requirements

---

## Workflow: From Outline to Draft

### Phase 1: Foundation (Before Writing)

```
1. Start with outline in .tex file
2. Run CLAIM-EVIDENCE MAP
   → Understand what claims you're making
   → Identify what evidence each claim needs

3. Run EXPERIMENT GAP
   → Get list of experiments to run
   → Prioritize by importance to claims

4. Run FIGURE PLANNER
   → Know what visualizations you need
   → Plan data collection accordingly
```

### Phase 2: Parallel Work

```
While Running Experiments:
├── Write Method section (knows what to describe)
├── Write Related Work (AGENT_RELATED_WORK helps)
├── Draft Introduction (knows claims)
└── Create figure templates (know what's needed)

When Results Come In:
├── Update CLAIM-EVIDENCE MAP
├── Fill in Tables
├── Create Figures
└── Write Experiments section
```

### Phase 3: Section-by-Section Drafting

```
For each section:
1. Gather inputs (notes, results, related work)
2. Run SECTION_WRITER with inputs
3. Review and edit output
4. Update claim-evidence map
5. Move to next section
```

### Phase 4: Handoff to Review Loop

```
When draft is complete:
→ Transition to AGENT_LOOP.md (Review system)
→ Run quality checks
→ Simulate reviews
→ Iterate to acceptance
```

---

## Agent Reference

| Agent | File | When to Use |
|-------|------|-------------|
| Claim-Evidence Map | `AGENT_CLAIM_EVIDENCE_MAP.md` | Start of writing, after major changes |
| Experiment Gap | `AGENT_EXPERIMENT_GAP.md` | Before running experiments |
| Figure Planner | `AGENT_FIGURE_PLANNER.md` | Before creating visualizations |
| Related Work | `AGENT_RELATED_WORK.md` | When writing related work, before submission |
| Section Writer | `AGENT_SECTION_WRITER.md` | When drafting any section |

---

## Example Session

### Starting a New Paper Section

**You:** "I need to write the Methods section. Here are my notes:
- Transformer architecture for game state
- Trained with PPO
- Key innovation: opponent modeling module
- Handles partial observability"

**Assistant runs:** `AGENT_SECTION_WRITER` with mode=expand

**Output:**
```latex
\section{Method}

We present our approach for learning game-playing agents
in partially observable environments. Our method combines
a transformer-based architecture with explicit opponent
modeling.

\subsection{Problem Setup}

Let $s_t \in \mathcal{S}$ denote the game state at time $t$...

\subsection{Architecture}

Our agent processes observations using a transformer encoder...

\subsection{Opponent Modeling}

A key challenge in competitive games is reasoning about
opponent behavior. We address this with a dedicated
opponent modeling module that...

\subsection{Training}

We train using Proximal Policy Optimization (PPO)~\citep{schulman2017}
with the following objective...
```

---

### Identifying Experiment Gaps

**You:** "What experiments do I need for this competition paper?"

**Assistant runs:** `AGENT_EXPERIMENT_GAP`

**Output:**
```markdown
## Critical Gaps (Must Address)

1. **Baseline Performance Comparison**
   - Claim: "Our baselines span difficulty levels"
   - Need: Win rates for all baselines against each other
   - Effort: Medium (need to run matches)

2. **Human Comparison**
   - Claim: "Baselines approach human-level"
   - Need: Elo comparison with human players
   - Effort: Low (can use existing ladder data)

3. **Track 2 Completion Metrics**
   - Claim: "Baseline completes early milestones"
   - Need: Milestone completion times
   - Effort: Medium (need to run baseline)

## Recommended Experiment Order
1. Run baseline battles (supports multiple claims)
2. Collect human comparison data
3. Run Track 2 baseline
4. Scaling analysis (nice to have)
```

---

## Integration with Review Loop

The Writing Assistant and Review Loop are complementary:

```
┌─────────────────────────────────────────────────────────────────┐
│                        PAPER LIFECYCLE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   EARLY STAGE              MID STAGE              LATE STAGE   │
│                                                                 │
│  ┌─────────────┐       ┌─────────────┐       ┌─────────────┐   │
│  │   WRITING   │       │   WRITING   │       │   REVIEW    │   │
│  │  ASSISTANT  │──────▶│  ASSISTANT  │──────▶│    LOOP     │   │
│  │             │       │             │       │             │   │
│  │ - Outline   │       │ - Drafting  │       │ - Polish    │   │
│  │ - Planning  │       │ - Filling   │       │ - Review    │   │
│  │ - Gaps      │       │ - Refining  │       │ - Iterate   │   │
│  └─────────────┘       └─────────────┘       └─────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Handoff criteria** (Writing → Review):
- All sections have content (not placeholders)
- Main claims have evidence
- Figures and tables exist
- Related work is written

---

## Quick Commands

```bash
# Full analysis
claude "Analyze root.tex: map claims, find gaps, plan figures"

# Single agent
claude "Run AGENT_EXPERIMENT_GAP on root.tex"
claude "Run AGENT_SECTION_WRITER on Introduction with notes: [notes]"
claude "Run AGENT_RELATED_WORK, focus on [topic]"

# Specific questions
claude "What experiments do I need to support the claim on line 45?"
claude "Help me write the abstract (150 words max)"
claude "What papers am I missing in related work?"
claude "Plan the figures for the experiments section"

# Transition to review
claude "Draft is complete. Run the review loop on root.tex"
```

---

## Best Practices

### Do:
- Run claim-evidence map early and update often
- Plan experiments before running them
- Use section writer for first drafts, then edit
- Check related work coverage before submission

### Don't:
- Run review loop on incomplete drafts
- Write without knowing what evidence you need
- Create figures without a plan
- Submit without checking citation coverage

### Iterative Process:
```
Write section → Update claim map → Check gaps →
Run experiments → Update section → Repeat
```
