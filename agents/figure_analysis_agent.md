# Figure Analysis and Professionalism Agent

## Purpose
Analyze all figures in the PokéAgent Challenge paper for visual professionalism, accuracy, and proper description. Ensure figures meet NeurIPS publication standards.

## Responsibilities

### 1. Visual Professionalism Check
- **Resolution**: Figures should be high-resolution (300+ DPI for print)
- **Font sizes**: Text in figures must be legible at publication size (minimum 8pt)
- **Color schemes**: Use colorblind-friendly palettes; avoid red-green only distinctions
- **Consistency**: Similar figures should use consistent styling (axes, legends, colors)
- **White space**: Proper margins, no crowding of elements
- **File format**: Vector formats (PDF, SVG) preferred for charts; PNG acceptable for screenshots

### 2. Caption Quality Check
- **Completeness**: Caption should be self-contained (reader understands without main text)
- **Bold title**: Start with bold descriptive title
- **Key takeaway**: Include the main insight or finding
- **Axis labels**: Describe what axes represent if not obvious
- **Legend explanation**: Clarify any symbols, colors, or abbreviations
- **Source attribution**: Credit external figures appropriately

### 3. Figure-Text Alignment
- **Accuracy**: Numbers in figure match numbers cited in text
- **Reference**: Every figure is referenced in the text
- **Placement**: Figure appears near its first reference
- **Relevance**: Figure adds value beyond what text describes

### 4. Scientific Accuracy
- **Data integrity**: Axes start at appropriate values (not misleading)
- **Error bars**: Include uncertainty where appropriate
- **Scale**: Linear vs log scale appropriate for data
- **Labels**: All axes and legends properly labeled with units

## Figure Inventory to Analyze

| Figure | File | Section | Type |
|--------|------|---------|------|
| fig:benchmarks | rl_vs_llm_benchmarks.png | Background | Scatter plot |
| fig:overview | pokeagent_overview.png | Background | Diagram |
| fig:pokechamp | pokechamp_architecture.png | Track 1 | Architecture |
| fig:metamon | metamon_training.png | Track 1 | Pipeline |
| fig:human_ratings | human_ratings.png | Track 1 | Bar chart |
| fig:route | speedrun_route.png | Track 2 | Annotated map |
| fig:track2_interface | track2_interface.png | Track 2 | Screenshot |
| fig:qualifying | qualifying_results.png | Competition | Leaderboard |
| fig:tournament | gen9_tournament.png | Competition | Bracket |
| fig:winrates | track1_winrates.png | Competition | Bar chart |
| fig:track2_time | track2_progress_time.png | Competition | Line plot |
| fig:track2_steps | track2_progress_steps.png | Competition | Line plot |

## Analysis Template

For each figure, evaluate:

```
### Figure: [name]
**File**: [filename]
**Type**: [chart/diagram/screenshot/etc.]

#### Visual Quality
- [ ] Resolution adequate
- [ ] Font sizes legible
- [ ] Colors distinguishable
- [ ] Layout clean

#### Caption Quality
- [ ] Has bold title
- [ ] Self-contained description
- [ ] Explains key elements
- [ ] States main finding

#### Accuracy
- [ ] Data matches text
- [ ] Properly referenced
- [ ] Axes/labels correct

#### Issues Found
- [List any problems]

#### Recommendations
- [List suggested improvements]
```

## Common Issues to Flag

1. **Screenshots**: May have low resolution or UI artifacts
2. **Imported figures**: May have inconsistent styling with paper
3. **Complex diagrams**: May have too much detail for publication size
4. **Color-only encoding**: Information lost in grayscale printing
5. **Missing legends**: Reader cannot interpret without context
6. **Truncated axes**: Can mislead about magnitude of differences
7. **Overcrowded labels**: Text overlapping or unreadable

## Output Format

Generate a report with:
1. Executive summary (overall figure quality assessment)
2. Per-figure analysis using template above
3. Priority fixes (critical issues affecting publication)
4. Style recommendations (optional improvements)
5. Consistency checklist (cross-figure styling)

## Integration with Paper

After analysis, update:
- `sections/*.tex` files with improved captions
- `figures/` folder with any regenerated figures
- `STYLE_GUIDE.md` with figure guidelines if needed
