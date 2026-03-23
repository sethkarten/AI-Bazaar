import os, glob

logs_dir = "logs/"

def resolve_run_dir(logs_dir, dlc, n_stab, seed):
    if n_stab == 0:
        if dlc == 3 and seed == 8:
            path = os.path.join(logs_dir, "exp1_baseline")
            return path if os.path.isdir(path) else None
        return None
    if n_stab == 5:
        path = os.path.join(logs_dir, f"exp1_stab_5_dlc{dlc}_seed{seed}")
        return path if os.path.isdir(path) else None
    path = os.path.join(logs_dir, f"exp1_stab_{n_stab}_dlc{dlc}_seed{seed}")
    return path if os.path.isdir(path) else None

# List actual contents of logs/
print("=== logs/ contents (exp1* only) ===")
try:
    entries = [e for e in os.listdir(logs_dir) if e.startswith("exp1")]
    for e in sorted(entries):
        print(" ", e)
except Exception as ex:
    print("ERROR listing logs/:", ex)

print()
print("=== resolve_run_dir tests ===")
for n_stab in [0, 1, 2, 4, 5]:
    for dlc in [1, 3, 5]:
        for seed in [8, 16, 64]:
            r = resolve_run_dir(logs_dir, dlc, n_stab, seed)
            if r:
                token_files = glob.glob(os.path.join(r, "*_token_usage.json"))
                print(f"  n_stab={n_stab} dlc={dlc} seed={seed}: {r} -> tokens: {token_files}")
            # only print found ones
