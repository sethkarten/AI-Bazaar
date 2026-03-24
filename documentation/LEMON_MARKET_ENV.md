# Lemon Market Environment — Status Report

---

## 1. Rating Functionality

### Initialization

All sellers (honest and sybil) are endowed with a reputation prior at environment setup via `BaseFirmAgent.initialize_reputation()`:

```python
def initialize_reputation(self, initial_rep: float, pseudo_count: float = 10.0) -> None:
    self.upvotes   = initial_rep * pseudo_count
    self.downvotes = (1.0 - initial_rep) * pseudo_count
    self.reputation = initial_rep
```

With the defaults `--reputation-initial 0.8` and `--reputation-pseudo-count 10`, every seller starts with `upvotes=8.0`, `downvotes=2.0`, `reputation=0.8`. Larger `pseudo_count` values create a stronger prior — more votes are needed to move reputation significantly.

Sybil identities spawned during rotation are bootstrapped identically: `rotate_identities()` calls `new_ident.initialize_reputation(r0, pseudo_count)` on each fresh identity.

### How Buyers Decide on Ratings

After market clearing, each buyer who purchased a car makes a second, independent LLM call in `BuyerAgent.review_transaction()`. The call is entirely separate from the bidding decision and uses its own system/user prompt pair (not the message history).

**System prompt:**
```
You are {buyer_name}, a buyer who just received a used car you purchased.
Your task is to review whether the seller's listing description accurately
represented the car's true quality.
```

**User prompt:**
```
Seller '{seller_id}' listed the car with this description:
  "{description}"

The actual quality you received: {quality_label}
(quality score {quality_received:.2f} on a 0–1 scale where 1.0=mint, 0.7=good, 0.4=fair, 0.1=poor).

Review the seller's description against the quality you received:
  - 'upvote'   if the description was accurate or undersold the quality
  - 'downvote' if the description was misleading or oversold the quality
  - 'abstain'  if you cannot make a determination

Respond with a single JSON object. Format: {vote_format}
```

The `vote_format` string varies by `--prompt-algo`:
- `io` (default): `{"vote": "upvote"|"downvote"|"abstain"}`
- `cot` / `sc`: `{"thought": "<brief reasoning>", "vote": "upvote"|"downvote"|"abstain"}`

The `description` passed to the buyer is the exact listing text the seller posted — including any deceptive sybil description. The `quality_label` and `quality_received` reflect the true good delivered, not what was advertised.

### Vote Submission and Reputation Update

`_parse_review()` parses the JSON response and returns `True` (upvote), `False` (downvote), or `None` (abstain). JSON parsing failure falls back to keyword scan of the raw text. LLM errors return `None` (abstain — no vote cast).

A vote is submitted only when non-`None`:

```python
# bazaar_env.py — post-clearing loop
if vote is not None and firm is not None and hasattr(firm, "receive_vote"):
    firm.receive_vote(vote)
```

`receive_vote` updates the running ratio:

```python
def receive_vote(self, upvote: bool) -> None:
    if upvote:
        self.upvotes += 1.0
    else:
        self.downvotes += 1.0
    total = self.upvotes + self.downvotes
    self.reputation = self.upvotes / total if total > 0 else 0.5
```

Reputation stays in [0, 1] at all times. Sybil identities whose reputation falls below `--sybil-rho-min` (default 0.3) are retired and replaced with fresh identities at `reputation_initial`.

---

## 2. Seller System Prompt and Persona Endowment

### Honest Sellers — Fixed Policy (`SellerAgent`)

`SellerAgent` uses no LLM. Descriptions are drawn directly from the `SELLER_PERSONAS` template dict, keyed by `(persona, quality_label)`. There is no system prompt. The persona is set at construction from the `--seller-personas` spec (default `"standard"`).

Available honest persona names and their style intent:

| Persona | Style |
|---------|-------|
| `standard` | Factual, no embellishment: `"Used car. Condition: good."` |
| `detailed` | Thorough, covers condition and buyer expectations |
| `terse` | One sentence maximum: `"Car. Good cond."` |
| `optimistic` | Upbeat and enthusiastic, still honest |

### Honest Sellers — LLM-Driven (`LLMSellerAgent`)

`LLMSellerAgent` builds a system prompt at construction from the persona string:

```python
def _build_system_prompt(self) -> str:
    style = SELLER_PERSONA_DESCRIPTIONS.get(self.persona or "standard", ...)
    return (
        "You are an honest used-car seller in a marketplace. "
        "You always describe the car's true condition accurately. "
        f"Description style: {style}"
    )
```

`SELLER_PERSONA_DESCRIPTIONS` maps each persona name to a style instruction string. For example, `optimistic` resolves to: `"Write an upbeat, enthusiastic listing that highlights the car's best qualities honestly. Be positive but accurate — do not exaggerate the condition."` The system prompt is fixed for the seller's lifetime; only the per-listing user prompt varies.

### Persona Endowment

Personas are assigned at world construction via `--seller-personas`. The spec is a comma-separated list of `persona:count` pairs (e.g. `"standard:3,detailed:3,terse:3,optimistic:3"`), parsed by `parse_seller_personas()` and assigned sequentially to the `num_honest` honest sellers. Any remainder is padded with the last persona in the spec.

Seller names are derived from their persona: sellers with unique personas are named after the persona directly (e.g. `"detailed"`); when multiple sellers share a persona, a numeric suffix is appended (e.g. `"standard_1"`, `"standard_2"`).

### Sybil Sellers — `DeceptivePrincipal`

The `DeceptivePrincipal` holds a cluster of `K` `SybilIdentity` objects. Each identity has a stylistic persona drawn round-robin from `SYBIL_PERSONAS`:

```
"formal and professional", "casual and friendly", "technical enthusiast",
"brief and direct", "detailed and descriptive"
```

Each slot's system prompt is drawn from `SYBIL_PERSONA_SYSTEM_PROMPTS` and instructs the LLM to write deceptive copy in the assigned style. For example, the `"formal and professional"` prompt tells the model to write *"polished, authoritative listing copy that projects trustworthiness"* with the explicit goal of presenting every car *"in its best possible light."*

The `DeceptivePrincipal` itself uses a coordinator system prompt:
```
You are a deceptive seller coordinating used-car listings. Your goal is to
maximise revenue by advertising cars as higher quality than they truly are
while keeping descriptions plausible enough to attract buyers.
```

---

## 3. Seller Listing Creation

### Listing Structure

Every listing — honest or sybil — is a dict posted to `market.listings` as a `Listing` dataclass:

```python
@dataclass
class Listing:
    id: str           # e.g. "seller_0_listing_0" or "sybil_1_listing_0"
    firm_id: str      # seller name
    description: str  # text shown to buyers
    price: float      # asking price in dollars
    reputation: float # seller's current reputation score
    quality: str      # TRUE quality label (mint/good/fair/poor) — not shown to buyers
    quality_value: float  # TRUE quality value (0.1–1.0) — not shown to buyers
```

True quality fields are stored for downstream reputation updates but are withheld from buyer observations.

### Fixed Honest Sellers (`SellerAgent.create_listings`)

One listing per unposted car per timestep. Description is the template string for `(persona, quality)`. Price is fixed at `V_MAX * quality_value` ($50,000 × quality_value).

### LLM Honest Sellers (`LLMSellerAgent.create_listings`)

One LLM call per car via `llm.send_msg()` (no message history — thread-safe):

**User prompt:**
```
The car's true condition is '{quality}' (quality value {quality_value:.2f}).
Write a short honest listing description.
Do not mention numeric quality values.
Respond with a single JSON object: {"description": "<text>"}
```

Falls back to the `SELLER_PERSONAS` template if the LLM call fails or returns an empty description. Price is always `V_MAX * quality_value`.

### Sybil Sellers (`DeceptivePrincipal.create_listings`)

Two-phase process per timestep:

**Phase 1 — Coordinated tier decision (one sequential LLM call):**

`decide_advertised_tier()` uses the message history to pick the advertised quality tier:

```
The true car quality is '{true_quality}' (value {true_qv:.2f}).
Available tiers (ascending): ['poor', 'fair', 'good', 'mint'].
Choose any advertised quality tier that is strictly higher than the true tier
to maximise willingness-to-pay.
You may claim good or mint condition even if the car is poor or fair.
Respond with a single JSON object. Format: {"advertised_quality": "<tier>"}
```

Falls back to `advertised_quality_for_sybil()` (one tier above) if the LLM response is invalid.

**Phase 2 — Per-identity description (parallel LLM calls):**

Each identity gets its own `llm.send_msg(..., json_format=True)` call using its stylistic system prompt:

```
You are listing a used car advertised as '{adv_quality}' condition.
Seller: {identity.name}. Reputation score: {identity.reputation:.2f}.
Write a short, compelling listing description.
Do not mention exact numeric quality values.
Price reference by quality tier:
  mint  ≈ $50,000  |  good  ≈ $35,000  |  fair  ≈ $20,000  |  poor  ≈ $5,000
Choose a price that maximises your revenue given your reputation and the competition.
Respond with a single JSON object: {"description": "<text>", "price": <number>}
```

`json_format=True` routes the response through `_extract_json()`, which strips markdown code fences before parsing. This is required for Gemini 2.5 Flash, which wraps JSON responses in triple-backtick blocks.

All K identities run concurrently via `ThreadPoolExecutor`. The resulting listing stores the **advertised** description and price but the **true** `quality` and `quality_value` for correct downstream vote handling.

### Sybil Best-N Slab Reflection

After each market clearing step, `bazaar_env.py` calls:

```python
self.deceptive_principal.record_step_outcome(self.timestep, _sybil_rev)
```

`record_step_outcome()` locates the `message_history` entry for the current timestep and writes two things:

1. `entry["metric"] = sybil_revenue / V_MAX` — normalised score in [0, ∞) where 1.0 means total sybil revenue equalled one full V_MAX sale
2. Appends to `entry["historical"]`: `"Step outcome: sybil revenue=$X,XXX (score=Y.YYY)\n"`

`DeceptivePrincipal` overrides `_build_best_n_slab(n)` from `LLMAgent`. On each prompt assembly, `get_historical_message()` calls this method, which:

- Sorts `message_history` by `metric` descending
- Deduplicates entries with identical (metric, action) pairs
- Takes the top-`n` entries (default `best_n=3`)
- Skips the slab entirely if the best score is 0 (no sales yet)

The formatted slab is injected into the prompt header:

```
Best N timesteps by sybil revenue (score = revenue / $50,000):
Timestep T (score S.SSS):
  <historical text for that timestep>
```

This gives the principal memory of which advertised tiers and price points actually converted, enabling adaptive strategy rather than always defaulting to `mint` at maximum price regardless of buyer response.

---

## 4. Buyer Listing Polling and Assessment

### Discovery Mechanism

At each timestep, the full market pool contains one listing per active seller (honest + sybil). Each buyer independently draws a **random sample without replacement** of up to `discovery_limit` listings from the pool:

```python
visible = random.sample(listings, min(discovery_limit, len(listings)))
```

`discovery_limit` defaults to 5 and is set by `--discovery-limit-consumers`. Each buyer's sample is drawn independently, so two buyers may see overlapping but not identical subsets. Buyers do **not** see the entire pool — they only evaluate the listings in their sample.

### What Buyers See

Each listing in the sample is presented with the following fields:

| Field | Visible? | Notes |
|-------|----------|-------|
| `listing_id` | Yes | Used to place a bid |
| `listed_price` | Yes | Asking price in dollars |
| `description` | Yes | Seller-written text |
| `seller_reputation` | Conditional | Withheld when `--no-buyer-rep` is set |
| `quality` | **No** | True quality label — never shown |
| `quality_value` | **No** | True quality value — never shown |

The full observation dict passed to the LLM also includes `timestep`, `persona`, `your_mean_quality_received` (average true quality across the buyer's own last-10 purchases, or `null` if no history yet), and the last 10 entries of the buyer's own `transaction_history` (each entry contains `timestep`, `seller_id`, `price_paid`, `quality_received`, `quality_label`, `consumer_surplus`). The global market-wide mean quality is intentionally withheld — buyers only have access to their personal experience.

### Bid Decision — Single LLM Call

Each buyer makes exactly one LLM call per timestep via `act_llm()`, seeing all sampled listings simultaneously and deciding on a single action. The prompt is assembled in `add_message(UPDATE_BID)`:

```
Your current observation:
{JSON observation — timestep, persona, your_mean_quality_received,
 transaction_history[-10:], listings_visible[]}

Decide whether to bid on one listing or pass this round.
If you bid, specify the listing_id of the car you want to buy.
Respond with a single JSON object. Format: {bid_format}
```

The `bid_format` string varies by `--prompt-algo`:
- `io`: `{"decision": "bid"|"pass", "listing_id": "<id or null>"}`
- `cot` / `sc`: `{"thought": "<brief reasoning>", "decision": "bid"|"pass", "listing_id": "<id or null>"}`

The buyer system prompt includes a fair market value reference to anchor price evaluation:

```
Fair market values by quality tier:
  mint ≈ $50,000  |  good ≈ $35,000  |  fair ≈ $20,000  |  poor ≈ $5,000
A listing priced above its claimed quality tier's fair value is likely overpriced.
```

The prompt also warns explicitly that some sellers misrepresent quality and instructs the buyer to weigh description, reputation, and transaction history together.

### Bid Mechanics

If the buyer decides `"bid"`, an `Order` is submitted targeting the specific `listing_id`:

```python
Order(consumer_id=self.name, firm_id=matched.firm_id, good="car",
      quantity=1, max_price=matched.price, listing_id=matched.id)
```

`max_price` is set to the listing's asking price — the buyer accepts the posted price or does not bid. At most one order per buyer per timestep. If the targeted listing has already been filled by another buyer before this order clears, the order fails silently.
