# Firm re-entry instances

The specific instances, root cause, and safeguards for firms that were out of business and later appeared in business again (CRASH runs) are documented in the paper notes:

**See \texttt{paper/notes.tex}, Section ``Firm Re-Entry Instances (Out of Business \(\to\) Back in Business)''.**

To regenerate the list of instances from your logs, run from project root:

```bash
python scripts/analyze_firm_reentry.py --log-dir logs --pattern "*crash*"
```
