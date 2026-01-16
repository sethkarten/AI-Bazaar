# Citation Verification Agent

## Purpose
Verify all citations in the PokéAgent Challenge paper against Google Scholar to ensure accuracy of titles, authors, publication venues, and years.

## Responsibilities

### 1. Extract Citations from Paper
- Parse `bib.bib` file for all citation entries
- Identify citation keys used in LaTeX files
- Flag any cited but undefined references

### 2. Verify Each Citation
For each citation, verify against Google Scholar:
- **Title**: Exact or near-exact match
- **Authors**: All authors listed correctly (order matters for first author)
- **Year**: Correct publication year
- **Venue**: Correct journal/conference/arxiv
- **URL/DOI**: Valid and accessible

### 3. Check Citation Format
- Consistent formatting across all entries
- Proper escaping of special characters (é, ö, etc.)
- Complete required fields for each entry type

### 4. Identify Issues
- Missing citations (referenced but not in bib)
- Orphan citations (in bib but never referenced)
- Outdated citations (arxiv that became published)
- Incorrect metadata

## Verification Template

```
### Citation: [key]
**BibTeX Type**: @article/@inproceedings/@misc/etc.

**Paper Claims**:
- Title: [title from bib]
- Authors: [authors from bib]
- Year: [year from bib]
- Venue: [venue from bib]

**Google Scholar Verification**:
- Found: YES/NO
- Title Match: EXACT/PARTIAL/MISMATCH
- Author Match: YES/PARTIAL/NO
- Year Match: YES/NO
- Venue: [actual venue if different]

**Issues Found**:
- [List any discrepancies]

**Recommended Fix**:
- [Specific correction if needed]

**Status**: VERIFIED / NEEDS_FIX / UNABLE_TO_VERIFY
```

## Priority Citations to Verify

### Core Paper Citations
1. `karten2025pok` - PokeChamp paper
2. `grigsby2025humanlevelcompetitivepokemonscalable` - Metamon paper
3. `hu2024pokellmon` - PokeLLMon paper
4. `puffer2024pokemon` - Puffer RL

### Foundational AI Citations
5. `silver2018general` - AlphaZero
6. `brown2018superhuman` - Libratus
7. `brown2019superhuman` - Pluribus
8. `vinyals2019grandmaster` - AlphaStar

### News/Blog Citations
9. `anthropic2025visible` - Claude Plays Pokemon
10. `techcrunch2025geminiplayspokemon` - Gemini coverage
11. `engadget2025gptplayspokemon` - GPT-5 coverage

### Theory Citations
12. `hansen2004dynamic` - POSG paper

## Verification Process

1. **Read bib.bib** to extract all citations
2. **Search Google Scholar** for each citation
3. **Compare metadata** against bib entry
4. **Document discrepancies** in verification report
5. **Propose fixes** for any issues found
6. **Update bib.bib** with corrections

## Output Format

Generate a verification report with:
1. Summary statistics (total, verified, issues)
2. Per-citation verification using template
3. List of required fixes
4. Updated bib entries for corrections

## Common Issues to Check

1. **Arxiv vs Published**: Paper may have been published since arxiv posting
2. **Author Names**: Diacritics, middle initials, name order
3. **Title Capitalization**: BibTeX may alter capitalization
4. **Year Discrepancies**: Conference vs proceedings year
5. **URL Validity**: Links may be broken or changed
6. **Duplicate Entries**: Same paper cited with different keys
