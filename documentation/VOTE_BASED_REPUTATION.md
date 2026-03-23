# Vote-Based Seller Reputation System

## Overview

Seller reputation in the LEMON_MARKET scenario is determined by buyer vote ratios rather than a silent exponential moving average of true quality. After each transaction, the buyer who received the good makes a second LLM call to review whether the seller's description matched the actual quality. The review produces an upvote or downvote, and the seller's reputation score is the running ratio of upvotes to total votes.

This design grounds reputation in agent behavior: deception is punished directly by the agents who experience it, not by an external oracle computing quality statistics.

---

## Reputation Formula

```
reputation = upvotes / (upvotes + downvotes)
```

Stored on `BaseFirmAgent` as `self.upvotes` and `self.downvotes` (floats). After each vote:

```python
# firm.py — BaseFirmAgent.receive_vote()
def receive_vote(self, upvote: bool) -> None:
    if upvote:
        self.upvotes += 1.0
    else:
        self.downvotes += 1.0
    total = self.upvotes + self.downvotes
    self.reputation = self.upvotes / total if total > 0 else 0.5
```

`self.reputation` always stays in [0, 1].

---

## Initialization (Prior)

All sellers are endowed with a reputation prior at environment setup using a pseudo-count bootstrap:

```python
# firm.py — BaseFirmAgent.initialize_reputation()
def initialize_reputation(self, initial_rep: float, pseudo_count: float = 10.0) -> None:
    self.upvotes   = initial_rep * pseudo_count
    self.downvotes = (1.0 - initial_rep) * pseudo_count
    self.reputation = initial_rep
```

With defaults `--reputation-initial 0.8` and `--reputation-pseudo-count 10`, every seller starts with `upvotes=8.0`, `downvotes=2.0`, `reputation=0.8`. The pseudo-count controls prior strength: larger values mean more votes are needed to move reputation significantly from the starting value.

**CLI args:**

| Arg | Default | Effect |
|-----|---------|--------|
| `--reputation-initial` | `0.8` | Starting reputation score for all sellers and for rotated sybil identities |
| `--reputation-pseudo-count` | `10.0` | Prior strength; `upvotes_0 = initial * k`, `downvotes_0 = (1 - initial) * k` |

---

## Buyer Review: Second LLM Call

Each transaction triggers a second, independent LLM call on the buying agent — separate from the bidding decision. The call is made in `BuyerAgent.review_transaction()` (`ai_bazaar/agents/buyer.py`).

### System Prompt

```
You are {buyer_name}, a buyer who just received a used car you purchased.
Your task is to review whether the seller's listing description accurately
represented the car's true quality.
```

### User Prompt

The format string injected at the end varies by `--prompt-algo`:

**`io` (default):**
```
{"vote": "upvote"|"downvote"|"abstain"}
```

**`cot` / `sc`:**
```
{"thought": "<brief reasoning>", "vote": "upvote"|"downvote"|"abstain"}
```

Full user prompt:

```
Seller '{seller_id}' listed the car with this description:
  "{description}"

The actual quality you received: {quality_label}
(quality score {quality_value:.2f} on a 0–1 scale where 1.0=mint, 0.7=good, 0.4=fair, 0.1=poor).

Review the seller's description against the quality you received:
  - 'upvote'   if the description was accurate or undersold the quality
  - 'downvote' if the description was misleading or oversold the quality
  - 'abstain'  if you cannot make a determination

Respond with a single JSON object. Format: {vote_format}
```

The `description` field is the exact listing text the seller posted (including sybil-generated deceptive descriptions). The `quality_label` and `quality_value` reflect the true good received — not what was advertised.

### Parsing (`BuyerAgent._parse_review`)

The response is parsed as JSON to extract the `"vote"` key. Markdown fences are stripped before parsing. If JSON parsing fails, the raw text is scanned for the keywords `"upvote"` or `"downvote"` as a fallback.

```python
@staticmethod
def _parse_review(response: str) -> Optional[bool]:
    # ... JSON parse with markdown-fence stripping ...
    if vote == "upvote":
        return True
    if vote == "downvote":
        return False
    return None  # abstain
```

Return values: `True` = upvote, `False` = downvote, `None` = abstain. On LLM error, returns `None` (abstain — no vote cast).

### No-LLM Fallback

When `self.llm is None` (offline/test mode), a rule-based heuristic is used:

```python
return quality_label.lower() in description.lower()
```

Returns `True` (upvote) if the true quality label appears literally in the description; `False` (downvote) otherwise.

---

## Vote Logic

| Situation | Expected vote |
|-----------|--------------|
| Description says "mint condition" → buyer receives mint | upvote |
| Description says "good condition" → buyer receives good | upvote |
| Description undersells: says "fair" → buyer receives good | upvote |
| Description says "mint condition" → buyer receives fair (sybil deception) | downvote |
| Description says "good reliable car" → buyer receives poor | downvote |
| Buyer cannot determine if description matches | abstain (no vote) |

---

## Data Flow Per Timestep

```
1. Firms post listings → market.post_listings()
2. Buyers sample discovery_limit=5 listings → make_orders() [LLM call #1]
3. market.clear() → sales_info includes: consumer_id, firm_id, price,
                    quality_value, description, listing_id
4. For each sale in sales_info:
   a. buyer.review_transaction(seller_id, description, quality_received,
                               quality_label, timestep)  [LLM call #2]
      → returns True (upvote), False (downvote), or None (abstain)
   b. if vote is not None: firm.receive_vote(vote)
      → updates firm.upvotes / firm.downvotes / firm.reputation
      abstain casts no vote; reputation is unchanged
   c. buyer.record_transaction(...)  → appends to transaction_history
5. Sybil rotation: identities with reputation < rho_min are retired and
   replaced with fresh identities at reputation_initial
```

The `description` field is passed from the `Listing` object through `market.clear()` via `_fill_order_listing()`, which returns a 6-tuple: `(True, quantity, price, quality_value, description, listing_id)`.

---

## Sybil Interaction

Sybil sellers systematically advertise one quality tier above the true good (e.g., claim "mint" for a "fair" car). Buyers who receive the misrepresented good will observe the mismatch and issue downvotes. As downvotes accumulate, `firm.reputation` falls toward 0.

Identity rotation (`sybil.py — DeceptivePrincipal.rotate_identities()`) retires any sybil identity with `reputation < rho_min` (default `0.3`) and spawns a fresh identity at `reputation_initial`. The new identity is immediately bootstrapped with `initialize_reputation(r0, pseudo_count)` so it re-enters the market with the same prior as all other sellers.

---

## Files Modified

| File | Change |
|------|--------|
| `ai_bazaar/agents/firm.py` | Added `upvotes`, `downvotes` fields; `initialize_reputation()`; `receive_vote()` |
| `ai_bazaar/agents/buyer.py` | Added `review_transaction()`; `UPDATE_REVIEW`/`ACTION_REVIEW` message handling |
| `ai_bazaar/market_core/market_core.py` | `_fill_order_listing()` now returns `description` and `listing_id`; `clear()` populates `sale_entry` with both |
| `ai_bazaar/env/bazaar_env.py` | Calls `initialize_reputation()` on firm construction; post-clearing loop calls `review_transaction()` then `receive_vote()` |
| `ai_bazaar/agents/sybil.py` | Initial and rotated sybil identities call `initialize_reputation()` |
| `ai_bazaar/utils/common.py` | Added `UPDATE_REVIEW`, `ACTION_REVIEW` to `Message` enum |
| `ai_bazaar/main.py` | Added `--reputation-pseudo-count`; `--reputation-alpha` marked legacy |
