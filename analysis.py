import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from mplsoccer import Pitch

# =========================================================
# KANSO ANALYTICS COMPETITION - MATCH EVENT ANALYSIS SCRIPT
# (All events colored navy blue with yellow outline)
# =========================================================


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def draw_mplsoccer_pitch(figsize=(12, 8), title=None):
    pitch = Pitch(
        pitch_type="opta",
        pitch_color="#133617",  # Lighter green color for better visibility
        line_color="white",
        linewidth=2,
        stripe=False
    )
    fig, ax = pitch.draw(figsize=figsize)
    if title:
        ax.set_title(title, fontsize=20, color="#ffffff", pad=20, fontweight='bold')
    return fig, ax, pitch


def add_yellow_outline(artist, width=2):
    artist.set_path_effects([
        pe.Stroke(linewidth=width, foreground=EVENT_OUTLINE),
        pe.Normal()
    ])
    
def add_black_outline(artist, width=2):
    artist.set_path_effects([
        pe.Stroke(linewidth=width, foreground="#000000"),
        pe.Normal()
    ])


# -----------------------------
# Color scheme
# -----------------------------
EVENT_COLOR = "#F5E700"   # yellow
EVENT_OUTLINE = "#F5E700" # yellow


# =========================================================
# 1. LOAD DATA
# =========================================================
file_path = "data/final_events_half_1.csv"
df = pd.read_csv(file_path)

print("Initial shape:", df.shape)
print(df.columns.tolist())


# =========================================================
# 2. PRESERVE EVENT ORDER
# =========================================================
df = df.reset_index(drop=True)
df["EventOrder"] = np.arange(1, len(df) + 1)
df = df[["EventOrder"] + [c for c in df.columns if c != "EventOrder"]]


# =========================================================
# 3. CLEAN DATA
# =========================================================
df["Team"] = df["Team"].astype(str).str.strip()
df["Event"] = df["Event"].astype(str).str.strip()
for c in ["X", "Y", "X2", "Y2"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["X", "Y"]).copy()


# =========================================================
# 4. STANDARDISE EVENT NAMES
# =========================================================
event_map = {
    "Pass": "Pass", "Cross": "Cross", "Shot": "Shot",
    "Shot Off Target": "Shot Off Target", "Corner Kick": "Corner Kick",
    "Dribble": "Dribble", "Ball Recovery": "Ball Recovery",
    "Turnover/Loss": "Turnover/Loss", "Clearance": "Clearance"
}
df["Event"] = df["Event"].replace(event_map)


# =========================================================
# 5. TEAM ATTACKING DIRECTION
# =========================================================
team_direction = {"Reading": "left", "Windsor": "right"}
df["AttackDirection"] = df["Team"].map(team_direction)


# =========================================================
# 6. BASIC MOVEMENT METRICS
# =========================================================
df["DeltaX"] = df["X2"] - df["X"]
df["DeltaY"] = df["Y2"] - df["Y"]
df["Distance"] = np.sqrt(df["DeltaX"] ** 2 + df["DeltaY"] ** 2)

def forward_distance(r):
    if pd.isna(r["X2"]):
        return np.nan
    if r["AttackDirection"] == "left":
        return r["X"] - r["X2"]
    elif r["AttackDirection"] == "right":
        return r["X2"] - r["X"]
    return np.nan

df["ForwardDistance"] = df.apply(forward_distance, axis=1)
df["ProgressiveAction"] = np.where(df["ForwardDistance"] > 10, 1, 0)


# =========================================================
# 7. PITCH THIRDS
# =========================================================
def get_start_third(r):
    x = r["X"]
    if r["AttackDirection"] == "left":
        if x > 66.67: return "Defensive Third"
        elif x > 33.33: return "Middle Third"
        else: return "Final Third"
    elif r["AttackDirection"] == "right":
        if x < 33.33: return "Defensive Third"
        elif x < 66.67: return "Middle Third"
        else: return "Final Third"
    return np.nan

def get_end_third(r):
    x = r["X2"]
    if pd.isna(x): return np.nan
    if r["AttackDirection"] == "left":
        if x > 66.67: return "Defensive Third"
        elif x > 33.33: return "Middle Third"
        else: return "Final Third"
    elif r["AttackDirection"] == "right":
        if x < 33.33: return "Defensive Third"
        elif x < 66.67: return "Middle Third"
        else: return "Final Third"
    return np.nan

df["StartThird"] = df.apply(get_start_third, axis=1)
df["EndThird"] = df.apply(get_end_third, axis=1)


# =========================================================
# 8. FINAL THIRD ENTRIES
# =========================================================
df["FinalThirdEntry"] = np.where(
    (df["StartThird"] != "Final Third") & (df["EndThird"] == "Final Third"),
    1, 0
)
# =========================================================
# 9. BOX ENTRIES
# =========================================================
def is_box_entry(r):
    if pd.isna(r["X2"]) or pd.isna(r["Y2"]):
        return 0
    if r["AttackDirection"] == "left":
        return 1 if (r["X"] > 18 and r["X2"] <= 18 and 21 <= r["Y2"] <= 79) else 0
    elif r["AttackDirection"] == "right":
        return 1 if (r["X"] < 82 and r["X2"] >= 82 and 21 <= r["Y2"] <= 79) else 0
    return 0

df["BoxEntry"] = df.apply(is_box_entry, axis=1)


# =========================================================
# 10. EVENT CATEGORY FLAGS
# =========================================================
attacking_events = ["Pass","Cross","Dribble","Shot","Shot Off Target","Corner Kick"]
defensive_events = ["Ball Recovery","Clearance"]
turnover_events = ["Turnover/Loss"]
regain_events = ["Ball Recovery"]

df["IsAttackingEvent"] = df["Event"].isin(attacking_events).astype(int)
df["IsDefensiveEvent"] = df["Event"].isin(defensive_events).astype(int)
df["IsTurnover"] = df["Event"].isin(turnover_events).astype(int)
df["IsRegain"] = df["Event"].isin(regain_events).astype(int)
df["IsShot"] = df["Event"].isin(["Shot","Shot Off Target"]).astype(int)
df["IsCross"] = (df["Event"]=="Cross").astype(int)
df["IsPass"] = (df["Event"]=="Pass").astype(int)
df["IsDribble"] = (df["Event"]=="Dribble").astype(int)
df["IsClearance"] = (df["Event"]=="Clearance").astype(int)
df["IsCorner"] = (df["Event"]=="Corner Kick").astype(int)


# =========================================================
# 11. NEXT-EVENT LOGIC
# =========================================================
df["NextTeam"] = df["Team"].shift(-1)
df["NextEvent"] = df["Event"].shift(-1)
df["NextX"] = df["X"].shift(-1)
df["NextY"] = df["Y"].shift(-1)
df["SameTeamNext"] = np.where(df["Team"] == df["NextTeam"], 1, 0)

def regain_outcome(r):
    if r["Event"] != "Ball Recovery":
        return np.nan
    if pd.isna(r["NextEvent"]): return "No Next Event"
    if r["NextTeam"] != r["Team"]: return "Lost Immediately"
    if r["NextEvent"] in ["Pass","Dribble","Cross"]: return "Retained and On-Ball Action"
    if r["NextEvent"] in ["Shot","Shot Off Target"]: return "Retained and Shot"
    return "Retained Other"

df["RegainOutcome"] = df.apply(regain_outcome, axis=1)


# =========================================================
# 12. SIMPLE TEAM-POSSESSION CHAINS
# =========================================================
df["PrevTeam"] = df["Team"].shift(1)
df["NewChainFlag"] = np.where(df["Team"] != df["PrevTeam"], 1, 0)
df["ChainID"] = df["NewChainFlag"].cumsum()

chain_summary = (
    df.groupby(["ChainID","Team"])
      .agg(
          ChainStartEventOrder=("EventOrder","min"),
          ChainEndEventOrder=("EventOrder","max"),
          ActionsInChain=("EventOrder","count"),
          ProgressiveActions=("ProgressiveAction","sum"),
          FinalThirdEntries=("FinalThirdEntry","sum"),
          BoxEntries=("BoxEntry","sum"),
          Shots=("IsShot","sum"),
          Crosses=("IsCross","sum"),
          Turnovers=("IsTurnover","sum")
      ).reset_index()
)

chain_summary["ChainOutcome"] = np.select(
    [
        chain_summary["Shots"]>0,
        chain_summary["BoxEntries"]>0,
        chain_summary["FinalThirdEntries"]>0,
        chain_summary["Turnovers"]>0
    ],
    ["Shot","Box Entry","Final Third Entry","Turnover"],
    default="Other"
)


# =========================================================
# 13. SUMMARY TABLES
# =========================================================
team_summary = (
    df.groupby("Team")
      .agg(
          TotalEvents=("EventOrder","count"),
          Passes=("IsPass","sum"),
          Crosses=("IsCross","sum"),
          Dribbles=("IsDribble","sum"),
          Shots=("IsShot","sum"),
          Corners=("IsCorner","sum"),
          Clearances=("IsClearance","sum"),
          BallRecoveries=("IsRegain","sum"),
          Turnovers=("IsTurnover","sum"),
          ProgressiveActions=("ProgressiveAction","sum"),
          FinalThirdEntries=("FinalThirdEntry","sum"),
          BoxEntries=("BoxEntry","sum")
      ).reset_index()
)

team_summary["ProgressiveActionRate"] = (
    team_summary["ProgressiveActions"]/team_summary["TotalEvents"]).round(3)
team_summary["FinalThirdEntryRate"] = (
    team_summary["FinalThirdEntries"]/team_summary["TotalEvents"]).round(3)
team_summary["BoxEntryRate"] = (
    team_summary["BoxEntries"]/team_summary["TotalEvents"]).round(3)

reading_df = df[df["Team"]=="Reading"].copy()
reading_kpis = pd.DataFrame({
    "Metric": ["Total Events","Passes","Crosses","Dribbles","Shots",
               "Corners","Ball Recoveries","Turnovers",
               "Progressive Actions","Final Third Entries","Box Entries"],
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


# =========================================================
# 14. EXPORT OUTPUTS
# =========================================================
os.makedirs("outputs", exist_ok=True)
df.to_csv("outputs/events_enriched.csv",index=False)
reading_df.to_csv("outputs/reading_events.csv",index=False)
team_summary.to_csv("outputs/team_summary.csv",index=False)
chain_summary.to_csv("outputs/chain_summary.csv",index=False)
reading_kpis.to_csv("outputs/reading_kpis.csv",index=False)
print("Data exports completed.")


# =========================================================
# 15. READING ACTION START LOCATIONS
# =========================================================
plot_df = df[df["Team"]=="Reading"].copy()

fig, ax, pitch = draw_mplsoccer_pitch(title="Reading Action Start Locations")
fig.patch.set_facecolor("#133617")

sc = pitch.scatter(plot_df["X"],plot_df["Y"],ax=ax,
                   s=40,alpha=0.9,
                   color=EVENT_COLOR,edgecolors=EVENT_OUTLINE,linewidths=2)
add_yellow_outline(sc,width=2)


plt.savefig("assets/reading_action_starts.png",dpi=300,bbox_inches="tight")
plt.show()


# =========================================================
# 17. PROGRESSIVE ACTION MAP (READING)
# =========================================================
prog_df = df[(df["Team"]=="Reading")&
             (df["ProgressiveAction"]==1)&
             (df["X2"].notna())&(df["Y2"].notna())].copy()

fig, ax, pitch = draw_mplsoccer_pitch(title="Reading Progressive Actions")
fig.patch.set_facecolor("#133617")

arr = pitch.arrows(prog_df["X"],prog_df["Y"],prog_df["X2"],prog_df["Y2"],
                   ax=ax,width=2,headwidth=5,headlength=5,
                   alpha=0.95,color=EVENT_COLOR)
add_yellow_outline(arr,width=2)

plt.savefig("assets/reading_progressive_actions.png",dpi=300,bbox_inches="tight")
plt.show()

# =========================================================
# 18. REGAIN MAP (READING)
# =========================================================
regain_df = df[(df["Team"]=="Reading") & (df["IsRegain"]==1)].copy()

fig, ax, pitch = draw_mplsoccer_pitch(title="Reading Ball Recoveries")
fig.patch.set_facecolor("#133617")

sc = pitch.scatter(regain_df["X"],regain_df["Y"],ax=ax,
                   s=90,alpha=0.9,
                   color=EVENT_COLOR,edgecolors=EVENT_OUTLINE,linewidths=2)
add_yellow_outline(sc,width=2)

plt.savefig("assets/reading_ball_recoveries.png",dpi=300,bbox_inches="tight")
plt.show()


# =========================================================
# 19. READING REGAIN OUTCOMES SUMMARY
# =========================================================
reading_regain_outcomes = (
    df[(df["Team"]=="Reading") & (df["IsRegain"]==1)]
      .groupby("RegainOutcome").size().reset_index(name="Count")
      .sort_values("Count",ascending=False)
)
reading_regain_outcomes.to_csv("outputs/reading_regain_outcomes.csv",index=False)


# =========================================================
# 20. FINAL THIRD ENTRY MAP (READING) – MATCHING SAMPLE STYLE
# =========================================================
fte_df = df[
    (df["Team"] == "Reading")
    & (df["FinalThirdEntry"] == 1)
    & (df["X2"].notna()) & (df["Y2"].notna())
].copy()

# Color palette (same feel as your image)
event_colors = {
    "Cross":        "#C77DFF",  # violet‑purple
    "Turnover/Loss":"#B0B0B0",  # grey
    "Pass":         "#1A2E6F",  # navy
    "Dribble":      "#39FF8F",  # green
    "Clearance":    "#F8F9FA"   # light white‑grey
}

fig, ax, pitch = draw_mplsoccer_pitch(title="Reading Final Third Entries")
fig.patch.set_facecolor("#133617")

for event_name, color in event_colors.items():
    subset = fte_df[fte_df["Event"] == event_name]
    if subset.empty:
        continue

    arr = pitch.arrows(
        subset["X"], subset["Y"], subset["X2"], subset["Y2"],
        ax=ax, width=2, headwidth=5, headlength=5,
        color=color, alpha=0.95, label=event_name
    )
    add_black_outline(arr, width=2)

# Legend
ax.legend(loc="upper right", fontsize=9, title="Event Type")

plt.savefig("assets/reading_final_third_entries_colored.png", dpi=300, bbox_inches="tight")
plt.show()


# =========================================================
# 21. SHOT MAP WITH COLOR CODING (READING)
# =========================================================
def normalize_shot_end(r):
    x2, y2 = r["X2"], r["Y2"]
    if pd.isna(x2) or pd.isna(y2):
        return pd.Series([x2, y2])
    if r["Event"] == "Shot":  # on target
        new_x2 = 0 if r["AttackDirection"] == "left" else 100
        new_y2 = 44 + ((min(max(y2, 0), 100) / 100) * 12)
        return pd.Series([new_x2, new_y2])
    return pd.Series([x2, y2])

df[["ShotPlotX2", "ShotPlotY2"]] = df.apply(normalize_shot_end, axis=1)

shot_df = df[
    (df["Team"] == "Reading")
    & (df["Event"].isin(["Shot", "Shot Off Target"]))
    & (df["ShotPlotX2"].notna())
    & (df["ShotPlotY2"].notna())
].copy()

fig, ax, pitch = draw_mplsoccer_pitch(title="Reading Shot Map (On vs Off Target)")
fig.patch.set_facecolor("#133617")

for _, r in shot_df.iterrows():
    # Set color based on outcome
    color = "#4CAF50" if r["Event"] == "Shot" else "#FF4D4D"  # green / red

    # Draw arrow
    ln = pitch.lines(r["X"], r["Y"], r["ShotPlotX2"], r["ShotPlotY2"], ax=ax,
                     color=color, lw=2, alpha=0.95)
    add_black_outline(ln, width=2)

    # Start dot
    sc_start = pitch.scatter(r["X"], r["Y"], ax=ax, s=40,
                             color=color, edgecolors='#000000',
                             linewidths=2, alpha=0.9)
    add_black_outline(sc_start, width=2)

    # End dot (goal or direction)
    sc_end = pitch.scatter(r["ShotPlotX2"], r["ShotPlotY2"], ax=ax, s=120,
                           color=color, edgecolors='#000000',
                           linewidths=2, alpha=0.95)
    add_black_outline(sc_end, width=2)

# Legend (green = on target, red = off target)
ax.scatter([], [], color="#4CAF50", edgecolors='#000000', s=80, label="Shot (On Target)")
ax.scatter([], [], color="#FF4D4D", edgecolors='#000000', s=80, label="Shot Off Target")
ax.legend(loc="upper right")

plt.savefig("assets/reading_shot_map_colored.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================================================
# 22. CROSSES MAP (READING)
# =========================================================
cross_df = df[(df["Team"]=="Reading") &
              (df["IsCross"]==1) &
              (df["X2"].notna()) & (df["Y2"].notna())].copy()

fig, ax, pitch = draw_mplsoccer_pitch(title="Reading Crosses")
fig.patch.set_facecolor("#133617")

arr = pitch.arrows(cross_df["X"],cross_df["Y"],cross_df["X2"],cross_df["Y2"],
                   ax=ax,width=2,headwidth=5,headlength=5,
                   color=EVENT_COLOR,alpha=0.95)
add_yellow_outline(arr,width=2)
plt.savefig("assets/reading_crosses.png",dpi=300,bbox_inches="tight")
plt.show()


# =========================================================
# 23. DRIBBLES MAP (READING)
# =========================================================
dribble_df = df[(df["Team"]=="Reading") &
                (df["IsDribble"]==1) &
                (df["X2"].notna()) & (df["Y2"].notna())].copy()

fig, ax, pitch = draw_mplsoccer_pitch(title="Reading Dribbles")
fig.patch.set_facecolor("#133617")

arr = pitch.arrows(dribble_df["X"],dribble_df["Y"],dribble_df["X2"],dribble_df["Y2"],
                   ax=ax,width=2,headwidth=5,headlength=5,
                   color=EVENT_COLOR,alpha=0.95)
add_yellow_outline(arr,width=2)
plt.savefig("assets/reading_dribbles.png",dpi=300,bbox_inches="tight")
plt.show()


# =========================================================
# 24. TURNOVERS MAP (READING) — DANGER COLOR-CODED
# =========================================================
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patheffects as pe

turnover_df = df[
    (df["Team"] == "Reading") &
    (df["IsTurnover"] == 1)
].copy()

# ── Danger Score ──────────────────────────────────────────────────────────────
# Right post in Opta coords: (100, 50)
# Closer = higher danger → invert distance so high score = more dangerous
RIGHT_POST_X = 100
RIGHT_POST_Y = 50

turnover_df["DistToRightPost"] = np.sqrt(
    (turnover_df["X"] - RIGHT_POST_X) ** 2 +
    (turnover_df["Y"] - RIGHT_POST_Y) ** 2
)

# Normalize: closer = score→1 (critical), farther = score→0 (safe)
max_dist = turnover_df["DistToRightPost"].max()
min_dist = turnover_df["DistToRightPost"].min()

turnover_df["DangerScore"] = 1 - (
    (turnover_df["DistToRightPost"] - min_dist) /
    (max_dist - min_dist + 1e-9)  # avoid div-by-zero
)

# ── Custom Colormap: green → yellow → red ────────────────────────────────────
danger_cmap = LinearSegmentedColormap.from_list(
    "turnover_danger",
    ["#2DC653", "#F5E700", "#FF2D2D"],  # safe → warning → critical
    N=256
)

norm = Normalize(vmin=0, vmax=1)

# ── Plot ──────────────────────────────────────────────────────────────────────
pitch = Pitch(
    pitch_type="opta",
    pitch_color="#133617",       # deep dark green — more premium feel
    line_color="#c8d6c8",
    linewidth=1.8,
    stripe=False,
    goal_type="box"
)

fig, ax = pitch.draw(figsize=(13, 9))
fig.patch.set_facecolor("#133617")

# ── Draw each turnover dot, colored by danger score ───────────────────────────
for _, row in turnover_df.iterrows():
    danger  = row["DangerScore"]
    color   = danger_cmap(norm(danger))
    size    = 80 + danger * 220   # critical = bigger dot

    sc = pitch.scatter(
        row["X"], row["Y"],
        ax=ax,
        s=size,
        color=color,
        edgecolors="#133617",
        linewidths=1.4,
        alpha=0.92,
        zorder=4
    )
    sc.set_path_effects([
        pe.Stroke(linewidth=2.5, foreground="#000000"),
        pe.Normal()
    ])

# ── Colorbar ──────────────────────────────────────────────────────────────────
sm = ScalarMappable(cmap=danger_cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=ax,
    orientation="vertical",
    fraction=0.025,
    pad=0.02,
    shrink=0.6
)
cbar.set_label(
    "← Safe          Critical →",
    fontsize=10,
    color="white",
    labelpad=12
)
cbar.ax.yaxis.set_tick_params(color="white")
cbar.outline.set_edgecolor("white")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])

# ── Danger zone shading (right penalty box area) ──────────────────────────────
danger_zone = plt.Rectangle(
    (82, 21), 18, 58,
    linewidth=1.5,
    edgecolor="#FF2D2D",
    facecolor="#FF2D2D",
    alpha=0.07,
    zorder=1
)
ax.add_patch(danger_zone)

# ── Annotate total critical turnovers (DangerScore > 0.65) ───────────────────
critical_count = (turnover_df["DangerScore"] > 0.65).sum()
total_count    = len(turnover_df)

ax.text(
    1, 103,
    f"⚠  Critical Turnovers (top-third danger zone): {critical_count} / {total_count}",
    fontsize=10,
    color="#FF2D2D",
    fontweight="bold",
    va="bottom"
)

# ── Title & subtitle ──────────────────────────────────────────────────────────
ax.set_title(
    "Reading — Turnover Danger Map",
    fontsize=18,
    color="white",
    fontweight="bold",
    pad=18
)

os.makedirs("assets", exist_ok=True)
plt.savefig("assets/reading_turnovers_danger.png", dpi=300,
            bbox_inches="tight", facecolor="#0a1a0c")
plt.show()
print(f"Turnover danger map saved. Critical: {critical_count}/{total_count}")



# =========================================================
# 25. BOX ENTRIES MAP (READING)
# =========================================================
box_df = df[(df["Team"]=="Reading") & 
            (df["BoxEntry"]==1) &
            (df["X2"].notna()) & (df["Y2"].notna())].copy()

fig, ax, pitch = draw_mplsoccer_pitch(title="Reading Box Entries")
fig.patch.set_facecolor("#133617")

arr = pitch.arrows(box_df["X"],box_df["Y"],box_df["X2"],box_df["Y2"],
                   ax=ax,width=2,headwidth=5,headlength=5,
                   color=EVENT_COLOR,alpha=0.95)
add_yellow_outline(arr,width=2)
plt.savefig("assets/reading_box_entries.png",dpi=300,bbox_inches="tight")
plt.show()

print("\nAll Reading event maps rendered with navy-blue and yellow outlines.")

# =========================================================
# 26. SHOT LOCATION HEATMAP (READING)
# =========================================================
from matplotlib.colors import LinearSegmentedColormap

# Filter to Reading shots
# heat_df = df[
#     (df["Team"] == "Reading") &
#     (df["Event"].isin(["Shot", "Shot Off Target"]))
# ].copy()

heat_df = df[df["Team"]=="Reading"].copy()

fig, ax, pitch = draw_mplsoccer_pitch(title="Reading Action Start Heatmap")
fig.patch.set_facecolor("#133617")

# Custom colormap – blue→yellow gradient for clarity on green pitch
colors = ["blue", "cyan", "limegreen", "yellow", "orange", "red"]
custom_cmap = LinearSegmentedColormap.from_list("reading_heat", colors)

# Draw density
pitch.kdeplot(
    x=heat_df["X"],
    y=heat_df["Y"],
    ax=ax,
    fill=True,
    levels=100,
    cmap=custom_cmap,
    shade_lowest=False,
    alpha=0.9,
    thresh=0.05
)

# Optional: overlay actual shot points
pitch.scatter(
    heat_df["X"], heat_df["Y"],
    ax=ax,
    s=20, c='blue', edgecolors="white", linewidths=0.4, alpha=0.6
)

plt.savefig("assets/reading_action_start_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

