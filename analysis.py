import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from mplsoccer import Pitch

# =========================================================
# KANSO ANALYTICS COMPETITION - MATCH EVENT ANALYSIS SCRIPT
# =========================================================
# WHAT THIS SCRIPT DOES:
# 1. Loads the raw event data
# 2. Preserves event order
# 3. Cleans text and coordinate fields
# 4. Standardises event names
# 5. Assigns attacking direction by team
# 6. Creates football logic columns
# 7. Builds summary tables
# 8. Exports Power BI-ready CSV files
# 9. Creates football visuals using mplsoccer
#
# FOLDER SETUP:
# kanso_analytics_project/
# ├── data/
# │   └── final_events_half_1.csv
# ├── outputs/
# └── analysis.py
# =========================================================


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def draw_mplsoccer_pitch(figsize=(10, 7), title=None):
    """
    Creates an mplsoccer pitch and returns fig, ax, pitch.
    Uses pitch_type='opta' which works well with 0-100 event coordinates.
    """
    pitch = Pitch(
        pitch_type="opta",
        pitch_color="#2c7a2c",   # dark green base
        line_color="white",
        linewidth=2,
        stripe=True,
        stripe_color="#2f8f2f"
    )

    fig, ax = pitch.draw(figsize=figsize)

    if title:
        ax.set_title(title, fontsize=14, color="black", pad=20)

    return fig, ax, pitch


def add_black_outline(artist, width=3):
    """
    Adds a black outline to mpl artists such as arrows and lines.
    """
    artist.set_path_effects([
        pe.Stroke(linewidth=width, foreground="black"),
        pe.Normal()
    ])


# Consistent color palette
LIGHT_GREEN = "#00ff88"
LIGHT_BLUE = "#66d9ff"
PURPLE = "#d966ff"
ORANGE = "#ffb000"
RED = "#ff4d4d"
YELLOW = "#f5e663"
CYAN = "#00ffff"
LIGHT_GREY = "#cfcfcf"
WHITE = "white"
BLACK = "black"

# Attacking Events Color Coding
CROSS_COLOR = "#C77DFF"      # violet
DRIBBLE_COLOR = "#39FF8F"    # mint green
TURNOVER_COLOR = "#FF5A5F"   # red
BOX_ENTRY_COLOR = "#FFB703"  # amber


# -----------------------------
# 1. LOAD DATA
# -----------------------------
file_path = "data/final_events_half_1.csv"
df = pd.read_csv(file_path)

print("Initial shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nUnique teams:")
print(df["Team"].unique())

print("\nRaw event counts:")
print(df["Event"].value_counts())


# -----------------------------
# 2. PRESERVE EVENT ORDER
# -----------------------------
df = df.reset_index(drop=True)
df["EventOrder"] = np.arange(1, len(df) + 1)

cols = ["EventOrder"] + [col for col in df.columns if col != "EventOrder"]
df = df[cols]

print("\nData with EventOrder added:")
print(df.head())


# -----------------------------
# 3. CLEAN DATA
# -----------------------------
df["Team"] = df["Team"].astype(str).str.strip()
df["Event"] = df["Event"].astype(str).str.strip()

coord_cols = ["X", "Y", "X2", "Y2"]
for col in coord_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("\nMissing values before dropping invalid rows:")
print(df.isnull().sum())

# Keep rows that at least have a starting location
df = df.dropna(subset=["X", "Y"]).copy()

print("\nShape after cleaning:", df.shape)


# -----------------------------
# 4. STANDARDISE EVENT NAMES
# -----------------------------
event_map = {
    "Pass": "Pass",
    "Cross": "Cross",
    "Shot": "Shot",
    "Shot Off Target": "Shot Off Target",
    "Corner Kick": "Corner Kick",
    "Dribble": "Dribble",
    "Ball Recovery": "Ball Recovery",
    "Turnover/Loss": "Turnover/Loss",
    "Clearance": "Clearance"
}

df["Event"] = df["Event"].replace(event_map)

print("\nStandardised event counts:")
print(df["Event"].value_counts())


# -----------------------------
# 5. TEAM ATTACKING DIRECTION
# -----------------------------
team_direction = {
    "Reading": "left",   # toward X = 0
    "Windsor": "right"   # toward X = 100
}

df["AttackDirection"] = df["Team"].map(team_direction)

print("\nAttack direction by team:")
print(df[["Team", "AttackDirection"]].drop_duplicates())


# -----------------------------
# 6. BASIC MOVEMENT METRICS
# -----------------------------
df["DeltaX"] = df["X2"] - df["X"]
df["DeltaY"] = df["Y2"] - df["Y"]
df["Distance"] = np.sqrt((df["DeltaX"] ** 2) + (df["DeltaY"] ** 2))

def forward_distance(row):
    if pd.isna(row["X2"]):
        return np.nan

    if row["AttackDirection"] == "left":
        return row["X"] - row["X2"]
    elif row["AttackDirection"] == "right":
        return row["X2"] - row["X"]
    return np.nan

df["ForwardDistance"] = df.apply(forward_distance, axis=1)
df["ProgressiveAction"] = np.where(df["ForwardDistance"] > 10, 1, 0)


# -----------------------------
# 7. PITCH THIRDS
# -----------------------------
def get_start_third(row):
    x = row["X"]

    if row["AttackDirection"] == "left":
        if x > 66.67:
            return "Defensive Third"
        elif x > 33.33:
            return "Middle Third"
        else:
            return "Final Third"

    elif row["AttackDirection"] == "right":
        if x < 33.33:
            return "Defensive Third"
        elif x < 66.67:
            return "Middle Third"
        else:
            return "Final Third"

    return np.nan


def get_end_third(row):
    x = row["X2"]

    if pd.isna(x):
        return np.nan

    if row["AttackDirection"] == "left":
        if x > 66.67:
            return "Defensive Third"
        elif x > 33.33:
            return "Middle Third"
        else:
            return "Final Third"

    elif row["AttackDirection"] == "right":
        if x < 33.33:
            return "Defensive Third"
        elif x < 66.67:
            return "Middle Third"
        else:
            return "Final Third"

    return np.nan


df["StartThird"] = df.apply(get_start_third, axis=1)
df["EndThird"] = df.apply(get_end_third, axis=1)

print("\nStart third counts:")
print(df["StartThird"].value_counts())


# -----------------------------
# 8. FINAL THIRD ENTRIES
# -----------------------------
df["FinalThirdEntry"] = np.where(
    (df["StartThird"] != "Final Third") & (df["EndThird"] == "Final Third"),
    1,
    0
)


# -----------------------------
# 9. BOX ENTRIES
# -----------------------------
def is_box_entry(row):
    if pd.isna(row["X2"]) or pd.isna(row["Y2"]):
        return 0

    if row["AttackDirection"] == "left":
        starts_outside = row["X"] > 18
        ends_inside = (row["X2"] <= 18) and (21 <= row["Y2"] <= 79)
        return 1 if starts_outside and ends_inside else 0

    elif row["AttackDirection"] == "right":
        starts_outside = row["X"] < 82
        ends_inside = (row["X2"] >= 82) and (21 <= row["Y2"] <= 79)
        return 1 if starts_outside and ends_inside else 0

    return 0

df["BoxEntry"] = df.apply(is_box_entry, axis=1)


# -----------------------------
# 10. EVENT CATEGORY FLAGS
# -----------------------------
attacking_events = ["Pass", "Cross", "Dribble", "Shot", "Shot Off Target", "Corner Kick"]
defensive_events = ["Ball Recovery", "Clearance"]
turnover_events = ["Turnover/Loss"]
regain_events = ["Ball Recovery"]

df["IsAttackingEvent"] = df["Event"].isin(attacking_events).astype(int)
df["IsDefensiveEvent"] = df["Event"].isin(defensive_events).astype(int)
df["IsTurnover"] = df["Event"].isin(turnover_events).astype(int)
df["IsRegain"] = df["Event"].isin(regain_events).astype(int)
df["IsShot"] = df["Event"].isin(["Shot", "Shot Off Target"]).astype(int)
df["IsCross"] = (df["Event"] == "Cross").astype(int)
df["IsPass"] = (df["Event"] == "Pass").astype(int)
df["IsDribble"] = (df["Event"] == "Dribble").astype(int)
df["IsClearance"] = (df["Event"] == "Clearance").astype(int)
df["IsCorner"] = (df["Event"] == "Corner Kick").astype(int)


# -----------------------------
# 11. NEXT-EVENT LOGIC
# -----------------------------
df["NextTeam"] = df["Team"].shift(-1)
df["NextEvent"] = df["Event"].shift(-1)
df["NextX"] = df["X"].shift(-1)
df["NextY"] = df["Y"].shift(-1)

df["SameTeamNext"] = np.where(df["Team"] == df["NextTeam"], 1, 0)

def regain_outcome(row):
    if row["Event"] != "Ball Recovery":
        return np.nan

    if pd.isna(row["NextEvent"]):
        return "No Next Event"

    if row["NextTeam"] != row["Team"]:
        return "Lost Immediately"

    if row["NextEvent"] in ["Pass", "Dribble", "Cross"]:
        return "Retained and On-Ball Action"

    if row["NextEvent"] in ["Shot", "Shot Off Target"]:
        return "Retained and Shot"

    return "Retained Other"

df["RegainOutcome"] = df.apply(regain_outcome, axis=1)


# -----------------------------
# 12. SIMPLE TEAM-POSSESSION CHAINS
# -----------------------------
df["PrevTeam"] = df["Team"].shift(1)
df["NewChainFlag"] = np.where(df["Team"] != df["PrevTeam"], 1, 0)
df["ChainID"] = df["NewChainFlag"].cumsum()

chain_summary = (
    df.groupby(["ChainID", "Team"])
      .agg(
          ChainStartEventOrder=("EventOrder", "min"),
          ChainEndEventOrder=("EventOrder", "max"),
          ActionsInChain=("EventOrder", "count"),
          ProgressiveActions=("ProgressiveAction", "sum"),
          FinalThirdEntries=("FinalThirdEntry", "sum"),
          BoxEntries=("BoxEntry", "sum"),
          Shots=("IsShot", "sum"),
          Crosses=("IsCross", "sum"),
          Turnovers=("IsTurnover", "sum")
      )
      .reset_index()
)

chain_summary["ChainOutcome"] = np.select(
    [
        chain_summary["Shots"] > 0,
        chain_summary["BoxEntries"] > 0,
        chain_summary["FinalThirdEntries"] > 0,
        chain_summary["Turnovers"] > 0
    ],
    [
        "Shot",
        "Box Entry",
        "Final Third Entry",
        "Turnover"
    ],
    default="Other"
)


# -----------------------------
# 13. SUMMARY TABLES
# -----------------------------
team_summary = (
    df.groupby("Team")
      .agg(
          TotalEvents=("EventOrder", "count"),
          Passes=("IsPass", "sum"),
          Crosses=("IsCross", "sum"),
          Dribbles=("IsDribble", "sum"),
          Shots=("IsShot", "sum"),
          Corners=("IsCorner", "sum"),
          Clearances=("IsClearance", "sum"),
          BallRecoveries=("IsRegain", "sum"),
          Turnovers=("IsTurnover", "sum"),
          ProgressiveActions=("ProgressiveAction", "sum"),
          FinalThirdEntries=("FinalThirdEntry", "sum"),
          BoxEntries=("BoxEntry", "sum")
      )
      .reset_index()
)

team_summary["ProgressiveActionRate"] = (
    team_summary["ProgressiveActions"] / team_summary["TotalEvents"]
).round(3)

team_summary["FinalThirdEntryRate"] = (
    team_summary["FinalThirdEntries"] / team_summary["TotalEvents"]
).round(3)

team_summary["BoxEntryRate"] = (
    team_summary["BoxEntries"] / team_summary["TotalEvents"]
).round(3)

event_summary = (
    df.groupby(["Team", "Event"])
      .size()
      .reset_index(name="Count")
      .sort_values(["Team", "Count"], ascending=[True, False])
)

zone_summary = (
    df.groupby(["Team", "StartThird"])
      .size()
      .reset_index(name="Count")
      .sort_values(["Team", "Count"], ascending=[True, False])
)

regain_summary = (
    df[df["Event"] == "Ball Recovery"]
    .groupby(["Team", "RegainOutcome"])
    .size()
    .reset_index(name="Count")
    .sort_values(["Team", "Count"], ascending=[True, False])
)

reading_df = df[df["Team"] == "Reading"].copy()

reading_kpis = pd.DataFrame({
    "Metric": [
        "Total Events",
        "Passes",
        "Crosses",
        "Dribbles",
        "Shots",
        "Corners",
        "Ball Recoveries",
        "Turnovers",
        "Progressive Actions",
        "Final Third Entries",
        "Box Entries"
    ],
    "Value": [
        reading_df["EventOrder"].count(),
        reading_df["IsPass"].sum(),
        reading_df["IsCross"].sum(),
        reading_df["IsDribble"].sum(),
        reading_df["IsShot"].sum(),
        reading_df["IsCorner"].sum(),
        reading_df["IsRegain"].sum(),
        reading_df["IsTurnover"].sum(),
        reading_df["ProgressiveAction"].sum(),
        reading_df["FinalThirdEntry"].sum(),
        reading_df["BoxEntry"].sum()
    ]
})


# -----------------------------
# 14. EXPORT OUTPUTS
# -----------------------------
os.makedirs("outputs", exist_ok=True)

df.to_csv("outputs/events_enriched.csv", index=False)
reading_df.to_csv("outputs/reading_events.csv", index=False)
team_summary.to_csv("outputs/team_summary.csv", index=False)
event_summary.to_csv("outputs/event_summary.csv", index=False)
zone_summary.to_csv("outputs/zone_summary.csv", index=False)
regain_summary.to_csv("outputs/regain_summary.csv", index=False)
chain_summary.to_csv("outputs/chain_summary.csv", index=False)
reading_kpis.to_csv("outputs/reading_kpis.csv", index=False)

print("\nExports completed:")
print("- outputs/events_enriched.csv")
print("- outputs/reading_events.csv")
print("- outputs/team_summary.csv")
print("- outputs/event_summary.csv")
print("- outputs/zone_summary.csv")
print("- outputs/regain_summary.csv")
print("- outputs/chain_summary.csv")
print("- outputs/reading_kpis.csv")


# -----------------------------
# 15. READING ACTION START LOCATIONS
# -----------------------------
plot_df = df[df["Team"] == "Reading"].copy()

fig, ax, pitch = draw_mplsoccer_pitch(title="Reading Action Start Locations")

sc = pitch.scatter(
    plot_df["X"],
    plot_df["Y"],
    ax=ax,
    s=40,
    alpha=0.85,
    color=LIGHT_GREEN,
    edgecolors=BLACK,
    linewidths=1.2
)
add_black_outline(sc, width=2)

plt.savefig("outputs/reading_action_starts.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nPitch plot saved:")
print("- outputs/reading_action_starts.png")


# -----------------------------
# 16. QUICK PRINT CHECKS
# -----------------------------
print("\nTeam Summary:")
print(team_summary)

print("\nReading KPIs:")
print(reading_kpis)

print("\nTop 10 Reading chains by length:")
print(
    chain_summary[chain_summary["Team"] == "Reading"]
    .sort_values("ActionsInChain", ascending=False)
    .head(10)
)


# -----------------------------
# 17. PROGRESSIVE ACTION MAP (READING)
# -----------------------------
prog_df = df[
    (df["Team"] == "Reading") &
    (df["ProgressiveAction"] == 1) &
    (df["X2"].notna()) &
    (df["Y2"].notna())
].copy()

fig, ax, pitch = draw_mplsoccer_pitch(title="Reading Progressive Actions")

arr = pitch.arrows(
    prog_df["X"],
    prog_df["Y"],
    prog_df["X2"],
    prog_df["Y2"],
    ax=ax,
    width=2,
    headwidth=5,
    headlength=5,
    alpha=0.95,
    color=LIGHT_GREEN
)
add_black_outline(arr, width=3.5)

plt.savefig("outputs/reading_progressive_actions.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nSaved progressive action map:")
print("- outputs/reading_progressive_actions.png")


# -----------------------------
# 18. REGAIN MAP (READING)
# -----------------------------
regain_df = df[
    (df["Team"] == "Reading") &
    (df["IsRegain"] == 1)
].copy()

print("\nUnique Reading events:")
print(df[df["Team"] == "Reading"]["Event"].value_counts())

print("\nRegain events being counted:")
print(regain_events)

print("\nNumber of Reading regains plotted:", len(regain_df))
print(regain_df[["EventOrder", "Team", "Event", "X", "Y"]])

fig, ax, pitch = draw_mplsoccer_pitch(title="Reading Ball Recoveries")

sc = pitch.scatter(
    regain_df["X"],
    regain_df["Y"],
    ax=ax,
    s=90,
    alpha=0.9,
    color=LIGHT_GREEN,
    edgecolors=BLACK,
    linewidths=1.2
)
add_black_outline(sc, width=2)

plt.savefig("outputs/reading_ball_recoveries.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nSaved regain map:")
print("- outputs/reading_ball_recoveries.png")


# -----------------------------
# 19. READING REGAIN OUTCOMES SUMMARY
# -----------------------------
reading_regain_outcomes = (
    df[
        (df["Team"] == "Reading") &
        (df["IsRegain"] == 1)
    ]
    .groupby("RegainOutcome")
    .size()
    .reset_index(name="Count")
    .sort_values("Count", ascending=False)
)

print("\nReading regain outcomes:")
print(reading_regain_outcomes)

reading_regain_outcomes.to_csv("outputs/reading_regain_outcomes.csv", index=False)
print("- outputs/reading_regain_outcomes.csv")


# -----------------------------
# 20. FINAL THIRD ENTRY MAP (READING)
# -----------------------------
fte_df = df[
    (df["Team"] == "Reading") &
    (df["FinalThirdEntry"] == 1) &
    (df["X2"].notna()) &
    (df["Y2"].notna())
].copy()

print("\nNumber of final third entries:", len(fte_df))
print("\nFinal third entry event counts:")
print(fte_df["Event"].value_counts())

event_colors = {
    "Pass": "blue",        
    "Dribble": "green",       
    "Cross": "purple",         
    "Shot": "orange",          
    "Shot Off Target": "red", 
    "Corner Kick": "yellow",  
    "Clearance": "#F8F9FA",    
    "Ball Recovery": "black", 
    "Turnover/Loss": "grey"  
}

fig, ax, pitch = draw_mplsoccer_pitch(title="Reading Final Third Entries")

for event_name in fte_df["Event"].dropna().unique():
    subset = fte_df[fte_df["Event"] == event_name]
    color = event_colors.get(event_name, LIGHT_GREY)

    arr = pitch.arrows(
        subset["X"],
        subset["Y"],
        subset["X2"],
        subset["Y2"],
        ax=ax,
        width=2,
        headwidth=5,
        headlength=5,
        color=color,
        alpha=0.95,
        label=event_name
    )
    add_black_outline(arr, width=3.5)

ax.legend(loc="upper right")

plt.savefig("outputs/reading_final_third_entries.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nSaved final third entry map:")
print("- outputs/reading_final_third_entries.png")


# -----------------------------
# 21. SHOT MAP WITH NORMALIZED END LOCATIONS (READING)
# -----------------------------
def normalize_shot_end(row):
    """
    Normalize shot end locations for cleaner visualization.

    Assumptions:
    - Reading attacks toward X = 0
    - Windsor attacks toward X = 100
    - Goal mouth is approximated as Y between 44 and 56
    """
    x2 = row["X2"]
    y2 = row["Y2"]

    if pd.isna(x2) or pd.isna(y2):
        return pd.Series([x2, y2])

    # On-target shots -> force to goal line and rescale into goal mouth
    if row["Event"] == "Shot":
        if row["AttackDirection"] == "left":
            new_x2 = 0
        else:
            new_x2 = 100

        new_y2 = 44 + ((min(max(y2, 0), 100) / 100) * 12)
        return pd.Series([new_x2, new_y2])

    # Off-target shots -> leave as tagged
    return pd.Series([x2, y2])


df[["ShotPlotX2", "ShotPlotY2"]] = df.apply(normalize_shot_end, axis=1)

shot_df = df[
    (df["Team"] == "Reading") &
    (df["Event"].isin(["Shot", "Shot Off Target"])) &
    (df["ShotPlotX2"].notna()) &
    (df["ShotPlotY2"].notna())
].copy()

print("\nTotal Reading shots:", len(shot_df))
print(shot_df["Event"].value_counts())

fig, ax, pitch = draw_mplsoccer_pitch(title="Reading Shot Map")

for _, row in shot_df.iterrows():
    color = LIGHT_GREEN if row["Event"] == "Shot" else RED

    ln = pitch.lines(
        row["X"],
        row["Y"],
        row["ShotPlotX2"],
        row["ShotPlotY2"],
        ax=ax,
        color=color,
        lw=2.5,
        alpha=0.95
    )
    add_black_outline(ln, width=4)

    sc_start = pitch.scatter(
        row["X"],
        row["Y"],
        ax=ax,
        s=40,
        color=color,
        edgecolors=BLACK,
        linewidths=1.2,
        alpha=0.9
    )
    add_black_outline(sc_start, width=2)

    sc_end = pitch.scatter(
        row["ShotPlotX2"],
        row["ShotPlotY2"],
        ax=ax,
        s=120,
        color=color,
        edgecolors=BLACK,
        linewidths=1.4,
        alpha=0.95
    )
    add_black_outline(sc_end, width=2)

ax.scatter([], [], color=LIGHT_GREEN, edgecolors=BLACK, s=80, label="Shot")
ax.scatter([], [], color=RED, edgecolors=BLACK, s=80, label="Shot Off Target")
ax.legend(loc="upper right")

plt.savefig("outputs/reading_shot_map_normalized.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nSaved normalized shot map:")
print("- outputs/reading_shot_map_normalized.png")



# -----------------------------
# 22. CROSSES MAP (READING)
# -----------------------------
cross_df = df[
    (df["Team"] == "Reading") &
    (df["IsCross"] == 1) &
    (df["X2"].notna()) &
    (df["Y2"].notna())
].copy()

fig, ax, pitch = draw_mplsoccer_pitch(title="Reading Crosses")

arr = pitch.arrows(
    cross_df["X"],
    cross_df["Y"],
    cross_df["X2"],
    cross_df["Y2"],
    ax=ax,
    width=2,
    headwidth=5,
    headlength=5,
    color=CROSS_COLOR,
    alpha=0.95
)
add_black_outline(arr, width=3.5)

plt.savefig("outputs/reading_crosses.png", dpi=300, bbox_inches="tight")
plt.show()

print("- outputs/reading_crosses.png")


# -----------------------------
# 23. DRIBBLES MAP (READING)
# -----------------------------
dribble_df = df[
    (df["Team"] == "Reading") &
    (df["IsDribble"] == 1) &
    (df["X2"].notna()) &
    (df["Y2"].notna())
].copy()

fig, ax, pitch = draw_mplsoccer_pitch(title="Reading Dribbles")

arr = pitch.arrows(
    dribble_df["X"],
    dribble_df["Y"],
    dribble_df["X2"],
    dribble_df["Y2"],
    ax=ax,
    width=2,
    headwidth=5,
    headlength=5,
    color=DRIBBLE_COLOR,
    alpha=0.95
)
add_black_outline(arr, width=3.5)

plt.savefig("outputs/reading_dribbles.png", dpi=300, bbox_inches="tight")
plt.show()

print("- outputs/reading_dribbles.png")

# -----------------------------
# 24. TURNOVERS MAP (READING)
# -----------------------------
turnover_df = df[
    (df["Team"] == "Reading") &
    (df["IsTurnover"] == 1)
].copy()

fig, ax, pitch = draw_mplsoccer_pitch(title="Reading Turnovers")

sc = pitch.scatter(
    turnover_df["X"],
    turnover_df["Y"],
    ax=ax,
    s=80,
    color=TURNOVER_COLOR,
    edgecolors="black",
    linewidths=1.2,
    alpha=0.95
)
add_black_outline(sc, width=2)

plt.savefig("outputs/reading_turnovers.png", dpi=300, bbox_inches="tight")
plt.show()

print("- outputs/reading_turnovers.png")

# -----------------------------
# 25. BOX ENTRIES MAP (READING)
# -----------------------------
box_df = df[
    (df["Team"] == "Reading") &
    (df["BoxEntry"] == 1) &
    (df["X2"].notna()) &
    (df["Y2"].notna())
].copy()

fig, ax, pitch = draw_mplsoccer_pitch(title="Reading Box Entries")

arr = pitch.arrows(
    box_df["X"],
    box_df["Y"],
    box_df["X2"],
    box_df["Y2"],
    ax=ax,
    width=2,
    headwidth=5,
    headlength=5,
    color=BOX_ENTRY_COLOR,
    alpha=0.95
)
add_black_outline(arr, width=3.5)

plt.savefig("outputs/reading_box_entries.png", dpi=300, bbox_inches="tight")
plt.show()

print("- outputs/reading_box_entries.png")