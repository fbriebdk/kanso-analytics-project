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

lc = pitch.lines(prog_df['X'], prog_df['Y'],
                 prog_df['X2'], prog_df['Y2'],
                 lw=4,
                 transparent=True,
                 comet=True,
                 color=EVENT_COLOR,
                 ax=ax)

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
# 20. READING FINAL THIRD ENTRIES — COLOUR-CODED LINES
# =========================================================
from mplsoccer import Pitch
from scipy.ndimage import gaussian_filter  # (not required but in imports already)

# --- Filter and clean
fte_df = df[
    (df["Team"] == "Reading") &
    (df["FinalThirdEntry"] == 1) &
    (df["X2"].notna()) & (df["Y2"].notna())
].copy()

fte_df["X"]  = fte_df["X"].clip(0, 100)
fte_df["Y"]  = fte_df["Y"].clip(0, 100)
fte_df["X2"] = fte_df["X2"].clip(0, 100)
fte_df["Y2"] = fte_df["Y2"].clip(0, 100)

# --- Colour palette (vivid & consistent)
event_colors = {
    "Pass":         "#1A2E6F",  # navy
    "Cross":        "#C77DFF",  # violet‑purple
    "Dribble":      "#39FF8F",  # neon green
    "Clearance":    "#F8F9FA",  # light grey
    "Turnover/Loss":"#B0B0B0"   # grey
}

# --- Pitch setup
fig, ax, pitch = draw_mplsoccer_pitch(title="Reading Final Third Entries (By Action Type)")
fig.patch.set_facecolor("#133617")

# --- Plot each action type as comet-style lines
for event_name, color in event_colors.items():
    subset = fte_df[fte_df["Event"] == event_name]
    if subset.empty:
        continue

    pitch.lines(
        subset["X"], subset["Y"],
        subset["X2"], subset["Y2"],
        ax=ax,
        lw=4,
        color=color,
        transparent=True,
        comet=True,
        alpha=0.9,
        label=event_name
    )

ax.legend(loc="upper right")

# --- Save
plt.savefig("assets/reading_final_third_entries_lines_colored.png",
            dpi=300, bbox_inches="tight", facecolor="#133617")
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

lc = pitch.lines(cross_df['X'], cross_df['Y'],
                 cross_df['X2'], cross_df['Y2'],
                 lw=4,
                 transparent=True,
                 comet=True,
                 color=EVENT_COLOR,
                 ax=ax)

plt.savefig("assets/reading_crosses.png",dpi=300,bbox_inches="tight")
plt.show()


# =========================================================
# 23. DRIBBLES MAP (READING)
# =========================================================
dribble_df = df[(df["Team"] == "Reading") &
                (df["IsDribble"] == 1) &
                (df["X2"].notna()) & (df["Y2"].notna())].copy()

fig, ax, pitch = draw_mplsoccer_pitch(title="Reading Dribbles")
fig.patch.set_facecolor("#133617")

lc = pitch.lines(dribble_df['X'], dribble_df['Y'],
                 dribble_df['X2'], dribble_df['Y2'],
                 lw=4,
                 transparent=True,
                 comet=True,
                 color=EVENT_COLOR,
                 ax=ax)

# (optional) Legend
# ax.legend(['dribbles'], facecolor='#133617', edgecolor='none', fontsize=14, loc='upper left')

plt.savefig("assets/reading_dribbles.png", dpi=300, bbox_inches="tight")
plt.show()



# =========================================================
# 24. TURNOVERS MAP (READING) — DANGER COLOR-CODED (NO BOX)
# =========================================================
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patheffects as pe

turnover_df = df[
    (df["Team"] == "Reading") &
    (df["IsTurnover"] == 1)
].copy()

# ── Danger Score ──────────────────────────────────────────────────────────────
RIGHT_POST_X = 100
RIGHT_POST_Y = 50

turnover_df["DistToRightPost"] = np.sqrt(
    (turnover_df["X"] - RIGHT_POST_X) ** 2 +
    (turnover_df["Y"] - RIGHT_POST_Y) ** 2
)

max_dist = turnover_df["DistToRightPost"].max()
min_dist = turnover_df["DistToRightPost"].min()

turnover_df["DangerScore"] = 1 - (
    (turnover_df["DistToRightPost"] - min_dist) /
    (max_dist - min_dist + 1e-9)
)

# ── Custom Colormap: green → yellow → red ────────────────────────────────────
danger_cmap = LinearSegmentedColormap.from_list(
    "turnover_danger",
    ["#2DC653", "#F5E700", "#FF2D2D"],
    N=256
)
norm = Normalize(vmin=0, vmax=1)

# ── Plot ──────────────────────────────────────────────────────────────────────
pitch = Pitch(
    pitch_type="opta",
    pitch_color="#133617",
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
    size    = 80 + danger * 220

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
        pe.Stroke(linewidth=2.5, foreground="#133617"),
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

# ── Title ────────────────────────────────────────────────────────────────────
ax.set_title(
    "Reading — Turnover Danger Map",
    fontsize=18,
    color="white",
    fontweight="bold",
    pad=18
)

os.makedirs("assets", exist_ok=True)
plt.savefig("assets/reading_turnovers_danger.png", dpi=300,
            bbox_inches="tight", facecolor="#133617")
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

lc = pitch.lines(box_df['X'], box_df['Y'],
                 box_df['X2'], box_df['Y2'],
                 lw=4,
                 transparent=True,
                 comet=True,
                 color=EVENT_COLOR,
                 ax=ax)

plt.savefig("assets/reading_box_entries.png",dpi=300,bbox_inches="tight")
plt.show()

print("\nAll Reading event maps rendered with navy-blue and yellow outlines.")

# =========================================================
# 27. READING ACTION START HEATMAP
# =========================================================
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap

# Filter + clean
action_df = df[df["Team"] == "Reading"].copy()
action_df["X"] = action_df["X"].clip(0, 100)
action_df["Y"] = action_df["Y"].clip(0, 100)
action_df = action_df.dropna(subset=["X", "Y"])

# Draw pitch
fig, ax, pitch = draw_mplsoccer_pitch(title="Reading Action Start Heatmap")
fig.patch.set_facecolor("#133617")

# Bin + smooth
bin_stat = pitch.bin_statistic(
    action_df["X"], action_df["Y"], statistic="count", bins=(50, 50)
)
bin_stat["statistic"] = gaussian_filter(bin_stat["statistic"], sigma=2)

# Custom cmap (dark → yellow → red)
cmap = LinearSegmentedColormap.from_list(
    "reading_heat", ["#133617", "#F5E700", "#FF6B00", "#FF0000"]
)

# Plot smooth heatmap
pcm = pitch.heatmap(bin_stat, ax=ax, cmap=cmap, edgecolors="none", alpha=0.9, zorder=1)

# Re‑draw pitch lines on top
pitch.draw(ax=ax)

# Optional light scatter overlay
pitch.scatter(
    action_df["X"], action_df["Y"], ax=ax, s=10, color="#efefef", alpha=0.15, zorder=11
)

# Colorbar
cbar = fig.colorbar(pcm, ax=ax, shrink=0.6, pad=0.02)
cbar.outline.set_edgecolor("#efefef")
cbar.ax.yaxis.set_tick_params(color="#efefef")
plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#efefef", fontsize=8)
cbar.set_label("Action Density", color="#efefef", fontsize=9)

# Save
plt.savefig("assets/reading_action_start_heatmap.png", dpi=300,
            bbox_inches="tight", facecolor="#133617")
plt.show()

# =========================================================
# 28. READING BALL RECOVERIES HEATMAP
# =========================================================
recover_df = df[(df["Team"] == "Reading") & (df["IsRegain"] == 1)].copy()
recover_df["X"] = recover_df["X"].clip(0, 100)
recover_df["Y"] = recover_df["Y"].clip(0, 100)
recover_df = recover_df.dropna(subset=["X", "Y"])

fig, ax, pitch = draw_mplsoccer_pitch(title="Reading Ball Recoveries Heatmap")
fig.patch.set_facecolor("#133617")

# Bin + smooth
bin_stat = pitch.bin_statistic(
    recover_df["X"], recover_df["Y"], statistic="count", bins=(50, 50)
)
bin_stat["statistic"] = gaussian_filter(bin_stat["statistic"], sigma=2)

# Colormap: use yellow tone for recoveries
cmap = LinearSegmentedColormap.from_list(
    "recover_heat", ["#133617", "#F5E700", "#FFB000", "#FF5000"]
)

pcm = pitch.heatmap(bin_stat, ax=ax, cmap=cmap, edgecolors="none", alpha=0.9, zorder=1)

# Redraw lines
pitch.draw(ax=ax)

# Optional scatter overlay
pitch.scatter(
    recover_df["X"], recover_df["Y"], ax=ax, s=10, color="#efefef", alpha=0.15, zorder=11
)

# Colorbar
cbar = fig.colorbar(pcm, ax=ax, shrink=0.6, pad=0.02)
cbar.outline.set_edgecolor("#efefef")
cbar.ax.yaxis.set_tick_params(color="#efefef")
plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#efefef", fontsize=8)
cbar.set_label("Recovery Density", color="#efefef", fontsize=9)

plt.savefig("assets/reading_ball_recoveries_heatmap.png", dpi=300,
            bbox_inches="tight", facecolor="#133617")
plt.show()

# =========================================================
# 29. READING TURNOVERS HEATMAP
# =========================================================
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap

# --- Filter team turnovers
turnover_df = df[(df["Team"] == "Reading") & (df["IsTurnover"] == 1)].copy()

# --- Clip coordinates to pitch frame 0–100 (avoid off‑pitch leaks)
turnover_df["X"] = turnover_df["X"].clip(0, 100)
turnover_df["Y"] = turnover_df["Y"].clip(0, 100)
turnover_df = turnover_df.dropna(subset=["X", "Y"])

# --- Create consistent pitch
fig, ax, pitch = draw_mplsoccer_pitch(title="Reading Turnovers Heatmap")
fig.patch.set_facecolor("#133617")

# --- Bin + smooth events
bin_stat = pitch.bin_statistic(
    turnover_df["X"], turnover_df["Y"],
    statistic='count',
    bins=(50, 50)
)
bin_stat['statistic'] = gaussian_filter(bin_stat['statistic'], sigma=2)

# --- Custom colormap (safe → warning → critical)
cmap = LinearSegmentedColormap.from_list(
    "turnover_heat",
    ["#133617", "#F5E700", "#FF6B00", "#FF0000"]
)

# --- Draw smooth heatmap
pcm = pitch.heatmap(
    bin_stat,
    ax=ax,
    cmap=cmap,
    edgecolors='none',
    alpha=0.9,
    zorder=1       # allow lines later to sit on top
)

# --- Redraw crisp white pitch lines over the heatmap
pitch.draw(ax=ax)

# --- Optional subtle event markers
pitch.scatter(
    turnover_df["X"], turnover_df["Y"],
    ax=ax,
    s=8, color="#efefef",
    alpha=0.15, zorder=11
)

# --- Colorbar styling
cbar = fig.colorbar(pcm, ax=ax, shrink=0.6, pad=0.02)
cbar.outline.set_edgecolor("#efefef")
cbar.ax.yaxis.set_tick_params(color="#efefef")
plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#efefef", fontsize=8)
cbar.set_label("Turnover Density", color="#efefef", fontsize=9)

# --- Save final figure
plt.savefig("assets/reading_turnovers_heatmap.png", dpi=300,
            bbox_inches="tight", facecolor="#133617")
plt.show()

# =========================================================
# 30. READING PROGRESSIVE ACTION START HEATMAP
# =========================================================
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap

# --- Filter data
prog_df = df[(df["Team"] == "Reading") & (df["ProgressiveAction"] == 1)].copy()

# --- Clean coordinates
prog_df["X"] = prog_df["X"].clip(0, 100)
prog_df["Y"] = prog_df["Y"].clip(0, 100)
prog_df = prog_df.dropna(subset=["X", "Y"])

# --- Standard pitch
fig, ax, pitch = draw_mplsoccer_pitch(title="Reading Progressive Action Start Heatmap")
fig.patch.set_facecolor("#133617")

# --- Bin + smooth starting points
bin_stat = pitch.bin_statistic(
    prog_df["X"], prog_df["Y"], statistic='count', bins=(50, 50)
)
bin_stat["statistic"] = gaussian_filter(bin_stat["statistic"], sigma=2)

# --- Colormap (bright yellow → red, fits brand)
cmap = LinearSegmentedColormap.from_list(
    "prog_heat", ["#133617", "#F5E700", "#FF6B00", "#FF0000"]
)

# --- Draw heatmap and re‑apply white lines
pcm = pitch.heatmap(bin_stat, ax=ax, cmap=cmap, edgecolors='none', alpha=0.9, zorder=1)
pitch.draw(ax=ax)

# --- Optional scatter overlay
pitch.scatter(prog_df["X"], prog_df["Y"], ax=ax,
              s=10, color="#efefef", alpha=0.15, zorder=11)

# --- Colorbar
cbar = fig.colorbar(pcm, ax=ax, shrink=0.6, pad=0.02)
cbar.outline.set_edgecolor('#efefef')
cbar.ax.yaxis.set_tick_params(color='#efefef')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#efefef', fontsize=8)
cbar.set_label("Action Start Density", color='#efefef', fontsize=9)

plt.savefig("assets/reading_progressive_action_starts_heatmap.png",
            dpi=300, bbox_inches="tight", facecolor="#133617")
plt.show()
