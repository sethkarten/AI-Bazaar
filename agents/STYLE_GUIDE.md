# NeurIPS Paper Style Guide

A guide for writing NeurIPS-quality academic papers with formal, scholarly prose.

## Academic Formal Writing Principles

### Paragraph Structure

Every paragraph should follow a clear structure that guides the reader through your argument.

**Topic Sentence**: Begin each paragraph with a sentence that states the main point. This sentence should be specific enough to preview the paragraph's content but general enough to encompass all supporting details. The reader should be able to skim topic sentences and understand the paper's flow.

**Supporting Evidence**: Follow the topic sentence with evidence, examples, data, or reasoning that supports your claim. Each supporting sentence should connect logically to the topic sentence and to adjacent sentences. Avoid introducing new major ideas mid-paragraph.

**Synthesis**: Conclude the paragraph by connecting the evidence back to your broader argument. This synthesis sentence should transition naturally to the next paragraph's topic. Avoid ending paragraphs with citations or raw data; instead, explain what the evidence means.

**Example of Poor Paragraph Structure**:
```
We used Pokemon for evaluation. Many papers have studied games. Our method
achieves 80% accuracy. Games are important for AI research.
```

**Example of Strong Paragraph Structure**:
```
Pokemon battles provide an ideal testbed for evaluating strategic reasoning
under uncertainty. The game's partial observability requires agents to maintain
beliefs about hidden opponent information, while its stochastic mechanics demand
robust decision-making across thousands of possible outcomes. These characteristics
mirror real-world challenges in domains ranging from cybersecurity to financial
trading, making advances in Pokemon AI broadly applicable to sequential
decision-making research.
```

### Formal Tone and Voice

Academic writing requires a measured, objective tone that prioritizes clarity over flair.

**Use Third Person for General Claims**: When discussing methodology or results, prefer constructions that emphasize the work rather than the authors. Write "The method achieves 80% accuracy" rather than "We achieved 80% accuracy" for general findings. Reserve first person plural ("we") for describing specific authorial choices: "We selected this baseline because..."

**Avoid Conversational Language**: Academic prose should not read like speech. Remove filler phrases ("It is interesting to note that"), rhetorical questions ("But what about edge cases?"), and informal transitions ("So, moving on to results..."). Replace these with direct statements and formal connectives.

**Maintain Consistent Register**: Do not shift between formal and informal registers within a section. If you begin a paragraph with precise technical language, maintain that precision throughout. Sudden shifts to casual phrasing undermine credibility.

**Passive Voice Usage**: Use passive voice strategically to emphasize actions over actors, particularly in methods sections ("The model was trained for 100 epochs") or when the agent is unknown or irrelevant. However, active voice often produces clearer, more direct prose. Balance both constructions based on what you wish to emphasize.

### Transitions and Flow

Smooth transitions between sentences and paragraphs are essential for readable academic prose.

**Sentence-Level Transitions**: Connect adjacent sentences through logical relationships. Use transitional phrases sparingly but effectively: "however" for contrast, "moreover" for addition, "consequently" for causation, "specifically" for elaboration. Avoid overusing any single transition.

**Paragraph-Level Transitions**: The final sentence of each paragraph should create expectation for what follows, while the first sentence of the next paragraph should fulfill that expectation. Readers should never feel jarred by topic shifts.

**Section-Level Transitions**: Begin each major section with a brief orienting statement that connects to the previous section and previews the current one. For example: "Having established the theoretical foundations of our approach, we now describe the experimental setup used to evaluate its performance."

### Sentence Construction

Academic sentences should be precise, unambiguous, and appropriately complex.

**Vary Sentence Length**: Alternate between longer sentences that develop complex ideas and shorter sentences that emphasize key points. A paper composed entirely of short sentences reads as choppy; one with only long sentences becomes exhausting. Aim for an average of 15-25 words per sentence with meaningful variation.

**Front-Load Important Information**: Place the most important information at the beginning of sentences and paragraphs. Readers remember openings and endings best. Bury qualifications and caveats in the middle of sentences rather than leading with them.

**Avoid Nominalization**: Prefer verbs over noun forms derived from verbs. Write "We analyzed the results" rather than "We performed an analysis of the results." Nominalizations add unnecessary words and weaken prose.

**Use Parallel Structure**: When listing items or comparing concepts, maintain grammatical parallelism. Write "The method improves accuracy, reduces latency, and simplifies deployment" rather than "The method improves accuracy, latency is reduced, and deployment becomes simpler."

## Avoiding AI-Style Prose

### Punctuation Guidelines

Academic writing uses punctuation conventionally and sparingly.

Em-dashes should be used rarely in formal academic prose. While acceptable for parenthetical asides in some contexts, overuse signals informal or AI-generated writing. Prefer parentheses for brief asides and separate sentences for substantial digressions. When em-dashes are necessary, use the LaTeX convention of triple hyphens (---) or the proper unicode character.

Semicolons connect closely related independent clauses. Use them when two sentences are so tightly linked that a period would create artificial separation. However, excessive semicolons in flowing prose often indicate run-on thinking. When in doubt, use a period.

Colons introduce lists, explanations, or elaborations. The clause before a colon should be grammatically complete. Use colons to signal "here is what I mean" or "here are the items."

### Words and Phrases to Avoid

**Replace Buzzwords with Specifics**:
- "leverage" → "use" or describe the specific mechanism
- "utilize" → "use"
- "facilitate" → "enable" or "allow"
- "streamline" → specify the efficiency gain
- "cutting-edge" → describe what makes it current
- "state-of-the-art" → "best-performing" with citation
- "novel" → describe what is new specifically
- "innovative" → demonstrate novelty through comparison
- "groundbreaking" → avoid entirely

**Quantify Vague Qualifiers**:
- "significantly improved" → provide the percentage or effect size
- "substantially better" → provide the numerical comparison
- "notable improvement" → quantify the improvement
- "promising results" → state the actual results

**Eliminate Hedging Language**:
- "We aim to" → state what you do
- "We attempt to" → describe what you accomplished
- "We hope to show" → "We show" or acknowledge the limitation
- "This paper seeks to" → "This paper presents/demonstrates/analyzes"

**Remove Self-Promotional Language**:
- "successfully developed" → "developed"
- "effectively addresses" → "addresses"
- "elegantly solves" → "solves"
- "comprehensive framework" → "framework"

## NeurIPS-Specific Guidelines

### Abstract Structure (150 words maximum)

The abstract should be a self-contained summary readable without the full paper. Structure it as follows:

Begin with one or two sentences establishing the problem and its importance. Follow with one or two sentences describing your approach at a high level. Present your key results with specific numbers in one or two sentences. Conclude with a single sentence on broader impact or implications.

### Introduction Organization

The introduction should progress logically from broad context to specific contributions.

Open with a hook that establishes why the problem matters. This should connect to broader concerns that readers care about, not just the technical niche. Follow with a description of the current state of the field, identifying the specific gap your work addresses. Present your contributions as a clearly enumerated list of three to four items. Conclude with a preview of your key results, including specific numbers that demonstrate your claims.

### Related Work

Organize related work thematically rather than chronologically. Group papers by approach, methodology, or problem formulation. For each group, explain the common thread, describe representative work, and clearly articulate how your approach differs. Maintain a fair and respectful tone toward prior work; dismissive language reflects poorly on authors.

### Methods Section

Begin with a formal problem statement that defines notation and objectives. Explain the intuition behind your approach before presenting technical details. Use algorithms or pseudocode to clarify complex procedures. Define all variables and notation before first use. Build complexity gradually, starting from simple cases before introducing full generality.

### Experiments Section

State your research questions explicitly at the beginning of the experiments section. Describe baselines thoroughly, including hyperparameter choices and implementation details. Report variance estimates or confidence intervals for all quantitative results. Include ablation studies that isolate the contribution of each component. Discuss failure cases and limitations honestly.

### Claims and Evidence

Every claim in an academic paper requires appropriate support. Numerical claims require citations to prior work or experimental evidence from your own study. Comparative claims require baseline comparisons with fair experimental conditions. Priority claims ("first to...") require thorough related work searches and careful qualification. Percentages should always include denominators or reference points.

Avoid overclaiming by scoping assertions appropriately. Write "addresses" rather than "solves." Write "outperforms tested baselines" rather than "outperforms all methods." Specify the conditions under which your claims hold rather than implying universal applicability.

## LaTeX Best Practices

### Citations

Group multiple citations in a single command: `\citep{paper1, paper2, paper3}`. Use `\citet{}` for inline author references ("Smith et al. showed that...") and `\citep{}` for parenthetical citations ("...as shown in prior work~\citep{smith2024}"). Place citations immediately after the claim they support, not at the end of paragraphs.

### Figures and Tables

Write informative captions that allow figures to stand alone. Begin with a bold summary phrase, then explain what the figure shows and how to interpret it. For tables, use the booktabs package for professional formatting. Bold the best results in comparison tables. Include standard deviations or confidence intervals. Place table captions above the table and figure captions below.

### Mathematical Notation

Define all notation before first use. Use consistent symbols throughout the paper. Place definitions in a notation table if the paper involves extensive mathematics. Use `\text{}` for words appearing in equations. Align multi-line equations at appropriate operators. Number only equations that are referenced elsewhere in the text.

## Common Reviewer Concerns

Reviewers frequently cite the following issues in negative reviews:

Missing obvious baselines signal incomplete experimental work. Lack of ablation studies raises questions about which components matter. Overclaiming without sufficient evidence damages credibility. Poor writing quality suggests rushed or careless work. Missing implementation details prevent reproducibility. Limited evaluation scope questions generalizability. Incomplete related work suggests unfamiliarity with the field.

## Pre-Submission Checklist

### Content Completeness
- Abstract under 150 words with problem, approach, results, and impact
- Contributions clearly enumerated in introduction
- All claims supported by evidence or citations
- Strong and fairly evaluated baselines
- Ablation studies isolating component contributions
- Limitations discussed honestly
- Broader impact statement included

### Writing Quality
- Formal academic tone throughout
- Paragraph structure with clear topic sentences
- Smooth transitions between sections
- No AI-style prose markers (excessive em-dashes, buzzwords)
- Consistent notation and terminology
- Thorough proofreading for grammar and spelling

### Formatting Compliance
- Within page limits for main text and supplementary
- Standard margins (do not modify neurips.sty)
- Anonymous for initial submission
- Complete and correctly formatted references
- Figures readable at print resolution
- Supplementary material clearly organized and labeled
