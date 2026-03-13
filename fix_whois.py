# Run this to fix whois_features.csv with proper flagging
# even when API calls failed
# Run with: python3 fix_whois.py

import pandas as pd
import os

INPUT  = "data/whois_features.csv"
OUTPUT = "data/whois_features.csv"

print("Fixing whois_features.csv...")

df = pd.read_csv(INPUT)
print(f"Loaded {len(df)} rows")

# For rows where domain_age_days = -1 (API failed),
# we set whois_flagged = 0 (unknown = not suspicious)
# For rows where we have real data, apply proper flagging

def flag_domain(row):
    age = row['domain_age_days']
    reg = str(row['registrar'])

    # Unknown data — can't flag
    if age == -1 or reg == 'unknown':
        return 0

    # Young domain = registered less than 1 year ago
    if 0 <= age < 365:
        return 1

    # Very cheap bulk registrars commonly used for fake domains
    suspicious_registrars = [
        'bizcn', 'publicdomainregistry', 'pdr ltd',
        'namecheap', 'enom', 'dynadot'
    ]
    for s in suspicious_registrars:
        if s.lower() in reg.lower():
            return 1

    return 0

df['whois_flagged'] = df.apply(flag_domain, axis=1)

# Save
df.to_csv(OUTPUT, index=False)

print(f"✅ Fixed! Flagged: {df['whois_flagged'].sum()} domains")
print(f"✅ Saved to {OUTPUT}")
print()
print("Sample:")
print(df[['domain','domain_age_days','registrar','whois_flagged']].head(5).to_string(index=False))
print()
print("🎉 whois_features.csv is ready for Day 9!")