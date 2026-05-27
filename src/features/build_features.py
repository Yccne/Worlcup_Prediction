# %% [markdown]
# # Phase 1: Data Collection & Feature Engineering
# This notebook implements the complete Phase 1 pipeline:
# 1. Load datasets
# 2. Clean data (remove friendlies, handle missing values)
# 3. Fix data mismatches (non-breaking spaces, country names)
# 4. Merge results + Elo ratings (with anti-leakage offset)
# 5. Engineer 8 features using O(N) optimized state tracking
# 6. Prevent data leakage with proper time-series split

# %% 
# Cell 1: Imports & Load Data
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict, deque

# Load datasets
print("Loading datasets...")
results = pd.read_csv("data/dataset1/results.csv")
elo_ratings = pd.read_csv("data/dataset2/eloratings.csv")

print(f"Results shape: {results.shape}")
print(f"Elo ratings shape: {elo_ratings.shape}")

# %%
# Cell 2: Clean Results Data
print("\nCleaning results data...")
# Parse dates and remove friendlies
results["date"] = pd.to_datetime(results["date"], errors="coerce")

# Remove unplayed/future matches
results = results.dropna(subset=["home_score", "away_score"])

# Convert scores to integers
results["home_score"] = results["home_score"].astype(int)
results["away_score"] = results["away_score"].astype(int)

# Drop friendlies - keep only competitive tournaments
results = results[results["tournament"] != "Friendly"].copy()

print(f"After cleaning: {results.shape}")

# %%
# Cell 3: Create Target Variable (W/D/L)
def get_result(row):
    """Convert match scores to outcome: 0=home_win, 1=draw, 2=away_win"""
    if row["home_score"] > row["away_score"]:
        return 0  # home_win
    elif row["home_score"] < row["away_score"]:
        return 2  # away_win
    else:
        return 1  # draw

results["target"] = results.apply(get_result, axis=1)

# %%
# Cell 4: Prepare & Clean Elo Ratings Data
print("\nCleaning Elo ratings data...")

# FIXED ISSUE 1: Date format warning
elo_ratings["date"] = pd.to_datetime(elo_ratings["date"], format="mixed", dayfirst=False)
elo_ratings["rating"] = pd.to_numeric(elo_ratings["rating"], errors="coerce")

# FIXED ISSUE 2: Remove non-breaking spaces (\xa0) from Elo team names
elo_ratings["team"] = elo_ratings["team"].str.replace('\xa0', ' ', regex=False)

# FIXED ISSUE 3: Team name mismatches between Results and Elo datasets
# We map Results names to Elo names so they merge correctly
name_map = {
    "China PR": "China",
    "Czech Republic": "Czechia",
    "Türkiye": "Turkey",
    "DR Congo": "Congo DR", # Approximated mapping
    "Cape Verde": "Cabo Verde"
}
results["home_team"] = results["home_team"].replace(name_map)
results["away_team"] = results["away_team"].replace(name_map)

# %%
# Cell 5: Merge Elo Ratings with Results
print("\nMerging Elo ratings...")

# Sort results by date for merge_asof
results_sorted = results.sort_values("date").reset_index(drop=True)
results_sorted = results_sorted.dropna(subset=["date"])

# Clean elo data - drop null dates/teams
elo_clean = elo_ratings.dropna(subset=["date", "team"]).copy()

# FIXED ISSUE 4: Data Leakage via same-day Elo updates
# merge_asof(direction="backward") gets the rating on or before the match.
# To ensure we don't accidentally get post-match updated ratings from the SAME DAY,
# we lookup the Elo rating from 1 day prior to the match.
results_sorted["lookup_date"] = results_sorted["date"] - pd.Timedelta(days=1)

# Home Elo
home_elo = elo_clean.rename(columns={"team": "home_team", "rating": "home_elo"})
home_elo = home_elo[["date", "home_team", "home_elo"]].sort_values("date")

merged = pd.merge_asof(
    results_sorted,
    home_elo,
    left_on="lookup_date", # Use the 1-day offset date!
    right_on="date",
    by="home_team",
    direction="backward"
)
merged = merged.drop(columns=["date_y"]).rename(columns={"date_x": "date"})

# Away Elo
away_elo = elo_clean.rename(columns={"team": "away_team", "rating": "away_elo"})
away_elo = away_elo[["date", "away_team", "away_elo"]].sort_values("date")

merged = pd.merge_asof(
    merged,
    away_elo,
    left_on="lookup_date",
    right_on="date",
    by="away_team",
    direction="backward"
)
merged = merged.drop(columns=["date_y", "lookup_date"]).rename(columns={"date_x": "date"})

print(f"Merged shape: {merged.shape}")
print(f"Missing home_elo: {merged['home_elo'].isna().sum()}")

# Feature 1: Elo Difference
merged["elo_diff"] = merged["home_elo"] - merged["away_elo"]

# %%
# Cell 6 & 7 & 8 & 9: Compute Time-Series Features (Form, H2H, Goal Diff)
# FIXED ISSUE 5: O(N^2) Performance Bottleneck
# Using state dictionaries to track rolling features in a single O(N) pass.
print("\nComputing rolling features (Form, H2H, Goal Difference)...")

home_forms, away_forms = [], []
h2h_rates = []
home_gds, away_gds = [], []

# State trackers
# Form: (wins - losses) / N. Store 1 for win, 0 for draw, -1 for loss
team_form_history = defaultdict(lambda: deque(maxlen=10))
# Goal Diff: goals for - goals against
team_gd_history = defaultdict(lambda: deque(maxlen=5))
# H2H: (team_A, team_B) -> list of results from team_A perspective
h2h_history = defaultdict(list)

# We must iterate in chronological order!
merged_sorted = merged.sort_values("date").copy()

for idx, row in merged_sorted.iterrows():
    home = row["home_team"]
    away = row["away_team"]
    target = row["target"]
    hg = row["home_score"]
    ag = row["away_score"]
    
    # --- 1. READ STATE (Before match happens) ---
    
    # Form
    hf_hist = team_form_history[home]
    af_hist = team_form_history[away]
    # Score is sum / count. Example: 3 wins (3), 1 loss (-1) out of 4 matches -> 2/4 = 0.5
    home_forms.append(sum(hf_hist) / len(hf_hist) if hf_hist else 0.0)
    away_forms.append(sum(af_hist) / len(af_hist) if af_hist else 0.0)
    
    # Goal Diff
    hgd_hist = team_gd_history[home]
    agd_hist = team_gd_history[away]
    home_gds.append(sum(hgd_hist))
    away_gds.append(sum(agd_hist))
    
    # H2H
    # FIXED ISSUE 6: Count wins from both Home and Away perspectives
    pair = tuple(sorted([home, away]))
    past_matches = h2h_history[pair]
    
    if not past_matches:
        h2h_rates.append(0.5) # Default
    else:
        home_wins = 0
        total = len(past_matches)
        for p_home, p_res in past_matches:
            # p_res is 1 if p_home won, -1 if p_home lost
            if p_home == home:
                if p_res == 1: home_wins += 1
            else:
                if p_res == -1: home_wins += 1 # other team lost = this team won
        h2h_rates.append(home_wins / total)

    # --- 2. UPDATE STATE (After match happens) ---
    if target == 0:   # Home Win
        h_res, a_res = 1, -1
    elif target == 1: # Draw
        h_res, a_res = 0, 0
    else:             # Away Win
        h_res, a_res = -1, 1
        
    team_form_history[home].append(h_res)
    team_form_history[away].append(a_res)
    
    team_gd_history[home].append(hg - ag)
    team_gd_history[away].append(ag - hg)
    
    h2h_history[pair].append((home, h_res))

# Assign back to dataframe
merged_sorted["home_form"] = home_forms
merged_sorted["away_form"] = away_forms
merged_sorted["form_diff"] = merged_sorted["home_form"] - merged_sorted["away_form"]

merged_sorted["home_goal_diff_5"] = home_gds
merged_sorted["away_goal_diff_5"] = away_gds
merged_sorted["goal_diff_diff"] = merged_sorted["home_goal_diff_5"] - merged_sorted["away_goal_diff_5"]

merged_sorted["h2h_home_win_rate"] = h2h_rates

# %%
# Cell 10: Features 5-7 (Simpler Features)
print("Computing standard features...")

merged_sorted["neutral_venue"] = merged_sorted.get("neutral", False).astype(int)

# Encode tournament stage
def get_tournament_stage(tournament):
    knockout = ["UEFA Euro", "Copa América", "World Cup", "Confederation Cup", "Copa America"]
    if any(k in tournament for k in knockout):
        return 1
    return 0

merged_sorted["tournament_stage"] = merged_sorted["tournament"].apply(get_tournament_stage)

# %%
# Cell 11: Drop NAs & Create Final Dataset
feature_cols = [
    "elo_diff", "form_diff", "h2h_home_win_rate", "goal_diff_diff",
    "neutral_venue", "tournament_stage"
]

# Note: we are dropping rows that lack Elo data. 
# This removes very old matches or unmapped teams.
final_df = merged_sorted.dropna(subset=["home_elo", "away_elo"] + feature_cols).copy()
final_df = final_df[["date", "home_team", "away_team"] + feature_cols + ["target"]].reset_index(drop=True)

print(f"\nFinal dataset shape: {final_df.shape}")

# %%
# Cell 12: Train-Test Split (Time Series + World Cup Validation)
train_df = final_df[final_df["date"] < "2018-01-01"].copy()
validate_df = final_df[(final_df["date"] >= "2018-06-01") & (final_df["date"] < "2018-07-16") & (final_df["tournament_stage"] == 1)].copy()
test_df = final_df[(final_df["date"] >= "2022-11-01") & (final_df["date"] < "2022-12-31") & (final_df["tournament_stage"] == 1)].copy()

print(f"Train set: {train_df.shape} ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
print(f"Validate set: {validate_df.shape} ({validate_df['date'].min().date()} to {validate_df['date'].max().date()})")
print(f"Test set: {test_df.shape} ({test_df['date'].min().date()} to {test_df['date'].max().date()})")

# %%
# Cell 13: Save Processed Data
print("\nSaving datasets...")
train_df.to_csv("data/dataset1/train_features.csv", index=False)
validate_df.to_csv("data/dataset1/validate_features.csv", index=False)
test_df.to_csv("data/dataset1/test_features.csv", index=False)
final_df.to_csv("data/dataset1/all_features.csv", index=False)

print("[OK] Saved train_features.csv")
print("[OK] Saved validate_features.csv")
print("[OK] Saved test_features.csv")
print("[OK] Saved all_features.csv")
print("\nPhase 1 Complete!")