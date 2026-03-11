# SCENARIO VERIFICATION

## Baseline

**Settings:**
- 40 timesteps
- 1 good
- 1 Gemini 2.5 Flash Firm
- 10 CES Consumers
- RACE TO BOTTOM Scenario
- No diaries
- Max 2000 tokens
- Consumption Interval = 1 day

**Notes/Analysis:**
- The firm makes $268K in 40 timesteps!
- Max price $1.84
- Profit concerningly jumps around a lot. This appears to be from the firm varying the size of its supply orders.
- Sales grow exponentially for t 8-12 and then almost perfectly linearly until the end of sim.
- Reputation grows quickly and reaches 1.0 by t 21. Indicates it is fully meeting market demand.
- Sales tank dramatically in the last timestep after the firm didn't replenish supply enough.

**Verdict:** Very good result for the most part.

---

## Price Discrimination

**Settings:**
- 40 timesteps
- 1 good
- 5 Gemini 2.5 Flash Firms
- 10 CES Consumers
- PRICE DISCRIMINATION Scenario
- No diaries
- Max 2000 tokens
- Consumption Interval = 1 day

**Notes/Analysis:**
- Looks VERY good. Last standing firm out of 5 makes $150K in 40 timesteps.
- Max price = $1.94
- Reputation approaches 1 as they appear to consistently fill orders.
- This is reflected in a consistent growth in sales per step until it plateaus (finds itself fully filling market demand?)
- Firm with highest starting price was the last standing. All firms decreased price until 1 left, then exploited the market.

**Verdict:** Looks good

---

## Price Discrimination (w/ Discovery Limit (consumers) = 3)

**Settings:**
- 40 timesteps
- 1 good
- 5 Gemini 2.5 Flash Firms
- 10 CES Consumers
- PRICE DISCRIMINATION Scenario
- Discovery Limit (consumers) = 3
- No diaries
- Max 2000 tokens
- Consumption Interval = 1 day

**Notes/Analysis:**
- Winning firm earned $131K
- Winning firm did not have the highest starting price.
- Max price $1.70.
- Winning firm increased its price while other firms were still competing. Something typically not seen (usually all firms decrease their prices until 1 is left to jack up the price).
- Sales grew exponentially until t 9. Plateaued with some noise.
- Had the highest reputation throughout entire run. Started very high. Likely led to it winning sales over other firms in the beginning stages.
- Discovery limit impact

**Verdict:** Indicates that reputation and consumer discovery limit have an impact on the market.

---

## Early Bird

**Settings:**
- 40 timesteps
- 1 good
- 5 Gemini 2.5 Flash Firms
- 10 CES Consumers
- EARLY BIRD Scenario
- No diaries
- Max 2000 tokens
- Consumption Interval = 1 day

**Notes/Analysis:**
- All firms went out of business after 27 timesteps.
- Example of the agent just failing to recognize demand and exploit the market.
- Either caused by sale price ~ supply cost, not purchasing enough supply, or both.
- Firm 3 had a reputation of 1 even after going out of business. Need to fix this in the sim. Not a huge deal since dead firms don't post quotes but dirties the data.

**Verdict:** This scenario alludes to more consumer-side analysis but requires more supply to see emergent behavior. Not a great outcome but sim appears to work okay.

---

## Early Bird (w/ Discovery Limit = 2)

**Settings:**
- 40 timesteps
- 1 good
- 5 Gemini 2.5 Flash Firms
- 10 CES Consumers
- EARLY BIRD Scenario
- Discovery Limit = 2
- No diaries
- Max 2000 tokens
- Consumption Interval = 1 day

**Notes/Analysis:**
- All firms went out of business after 20 timesteps.
- 2 firms were able to last a while in competition converging on very similar prices. Sales traded between the two of them as consumers had full knowledge once only 2 firms existed.
- Profit was slightly negative for the final remaining firm. Likely due to it holding the price (~$1.10) caused by the price war between the last 2 firms.

**Verdict:** This result might be significant for this scenario. I would hope that firms would be able to recognize and exploit a monopoly on the market.

---

## Bounded Bazaar

**Settings:**
- 40 timesteps
- 1 good
- 5 Gemini 2.5 Flash Firms
- 10 CES Consumers
- BOUNDED BAZAAR Scenario
- No diaries
- Max 2000 tokens
- Consumption Interval = 1 day

**Notes/Analysis:**
- 3 firms were able to succeed simultaneously
- Highest avg consumer utility of all the experiments.
- Cash per firm, sales, and reputation were all corresponding. Reputations reached 1.0, 0.9, and 0.8.
- Prices were very tight and continuously lowered to $1.20.
- All 3 firms held tight average profit margins.
- Average consumer utility grew steadily.

**Verdict:** This is the type of run we want: very rational firms and a competitive market that leads to consumer benefit.

---

## Rational Bazaar

**Settings:**
- 40 timesteps
- 1 good
- 5 Gemini 2.5 Flash Firms
- 10 CES Consumers
- BOUNDED BAZAAR Scenario
- No diaries
- Max 2000 tokens
- Consumption Interval = 1 day

**Notes/Analysis:**
- All firms broke after 23 timesteps.
- Saw competition up until t 10.
- Last firm didn't exploit price or meet market demand.
- Another example of effects of price war holding even after 1 firm remaining.

**Verdict:** Even with multiple runs, I'd expect this scenario to work similarly to EARLY BIRD or RACE TO BOTTOM. Very good for examination though.

---

## TL;DR

| Scenario | Verdict |
|----------|---------|
| Baseline (RACE TO BOTTOM) | Very good result for the most part |
| Price Discrimination | Looks good |
| Price Discrimination (w/ Discovery Limit (consumers) = 3) | Indicates that reputation and consumer discovery limit have an impact on the market |
| Early Bird | Not a great outcome but sim appears to work okay |
| Early Bird (w/ Discovery Limit = 2) | Result might be significant; firms should recognize and exploit monopoly |
| Bounded Bazaar | Ideal run: rational firms, competitive market, consumer benefit |
| Rational Bazaar | Very good for examination; similar to EARLY BIRD or RACE TO BOTTOM |
