import glob, json, os

run_dir = 'logs/exp2_initial_test2'
files = sorted(
    glob.glob(os.path.join(run_dir, 'state_t*.json')),
    key=lambda p: int(''.join(filter(str.isdigit, os.path.basename(p))) or '0')
)

print(f'=== EXP2_INITIAL_TEST2 ANALYSIS ({len(files)} timesteps) ===')
print('Sybil cluster size: 6, Honest sellers: 6 (1 detailed, 1 terse, 4 optimistic)')
print()

sybil_names = [f'sybil_{i}' for i in range(6)]
honest_names = ['detailed_0','terse_1','optimistic_2','optimistic_3','optimistic_4','optimistic_5']

per_firm_sales = {n: 0 for n in sybil_names + honest_names}
per_firm_revenue = {n: 0.0 for n in sybil_names + honest_names}

print('=== PER-TIMESTEP LISTING / SALES / PASS DETAIL ===')
for fpath in files:
    with open(fpath) as fh:
        state = json.load(fh)
    t = state['timestep']

    listings = state.get('lemon_market_new_listings', [])
    unsold = {l['id'] for l in state.get('lemon_market_unsold_listings', [])}
    sybil_rev = state.get('lemon_market_sybil_revenue_share', None)
    avg_cs = state.get('lemon_market_avg_consumer_surplus', None)
    bids = state.get('lemon_market_bids_count', 0)
    passes = state.get('lemon_market_passes_count', 0)

    print(f'--- t={t} --- listings={len(listings)} bids={bids} passes={passes}', end='')
    if sybil_rev is not None:
        print(f'  sybil_rev_share={sybil_rev:.3f}', end='')
    if avg_cs is not None:
        print(f'  avg_cs={avg_cs:.0f}', end='')
    print()

    for l in listings:
        sold = l['id'] not in unsold
        tag = '[SOLD  ]' if sold else '[UNSOLD]'
        firm_is_sybil = l['firm_id'].startswith('sybil')
        sybil_tag = 'SYB' if firm_is_sybil else 'HON'
        adv_qual = l.get('quality', '?')
        true_qval = l.get('quality_value', '?')
        desc = l.get('description','')[:55]
        print(f'  {tag} [{sybil_tag}] {l["firm_id"]:20s} price={l["price"]:7.0f}  adv_q={adv_qual:5s} true_qv={true_qval}  "{desc}"')

    # accumulate sales
    for firm in state['firms']:
        nm = firm['name']
        for s in firm.get('sales_info', []):
            per_firm_sales[nm] = per_firm_sales.get(nm, 0) + s.get('quantity_sold', 0)
            per_firm_revenue[nm] = per_firm_revenue.get(nm, 0.0) + s.get('price', 0) * s.get('quantity_sold', 0)
    print()

# ---- Reputation progression -----
print('=== REPUTATION OVER TIME (rows=firms, cols=timesteps) ===')
header = f"{'Firm':20s}"
for fpath in files:
    with open(fpath) as fh:
        state = json.load(fh)
    header += f"  t{state['timestep']:02d} "
print(header)

for nm in sybil_names + honest_names:
    row = f'{nm:20s}'
    for fpath in files:
        with open(fpath) as fh:
            state = json.load(fh)
        rep = next((firm['reputation'] for firm in state['firms'] if firm['name'] == nm), None)
        row += f'  {rep:.3f}' if rep is not None else '   N/A'
    print(row)

# ---- Upvotes/downvotes final -----
print()
print('=== UPVOTES / DOWNVOTES at final timestep ===')
with open(files[-1]) as fh:
    last = json.load(fh)
for firm in sorted(last['firms'], key=lambda x: x['name']):
    nm = firm['name']
    uv = firm.get('upvotes', 'N/A')
    dv = firm.get('downvotes', 'N/A')
    rep = firm.get('reputation', 'N/A')
    tag = 'SYBIL ' if firm.get('sybil') else 'honest'
    ib = 'active' if firm.get('in_business') else 'bankrupt'
    if isinstance(uv, float):
        print(f'  [{tag}] {nm:20s}  upvotes={uv:5.1f}  downvotes={dv:5.1f}  rep={rep:.4f}  [{ib}]')
    else:
        print(f'  [{tag}] {nm:20s}  upvotes=N/A  downvotes=N/A  [{ib}]')

# ---- Cumulative summary -----
print()
print('=== CUMULATIVE SALES SUMMARY ===')
ts = sum(per_firm_sales[n] for n in sybil_names)
th = sum(per_firm_sales[n] for n in honest_names)
rs = sum(per_firm_revenue[n] for n in sybil_names)
rh = sum(per_firm_revenue[n] for n in honest_names)
print(f'  Sybil  total sales={ts}  total revenue={rs:.0f}')
print(f'  Honest total sales={th}  total revenue={rh:.0f}')
if rs + rh > 0:
    print(f'  Sybil revenue share (aggregated from step data): {rs/(rs+rh):.4f}')
print()
print('  Per-firm breakdown:')
for nm in sybil_names + honest_names:
    tag = 'SYBIL' if nm.startswith('sybil') else 'honest'
    print(f'    [{tag}] {nm:20s}  sales={per_firm_sales.get(nm,0):3d}  revenue={per_firm_revenue.get(nm,0.0):10.0f}')

# ---- Consumer sybil pass rates -----
print()
print('=== CONSUMER SYBIL PASS RATES at final timestep ===')
for c in sorted(last['consumers'], key=lambda x: x['name']):
    seen = c.get('sybil_seen_total', 0)
    passed = c.get('sybil_passed_total', 0)
    rate = c.get('sybil_pass_rate_this_step', None)
    surp = c.get('consumer_surplus_cumulative', 0)
    print(f'  {c["name"]:12s}  sybil_seen={seen}  sybil_passed={passed}  last_pass_rate={rate}  cumul_surplus={surp:.0f}')
