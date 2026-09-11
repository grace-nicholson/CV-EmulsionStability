import pandas as pd
import matplotlib.pyplot as plt
import os


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "LF_RS_pH5_detection4.csv"

OUTPUT_FOLDER = "phase_graphs_LF"


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# LOAD CSV
# ============================================================

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df)} detection rows")


# ============================================================
# MAKE SURE TIME IS NUMERIC
# ============================================================

df["time_s"] = pd.to_numeric(
    df["time_s"],
    errors="coerce"
)


# Remove rows where time is missing

df = df.dropna(
    subset=["time_s"]
)


# ============================================================
# CALCULATE VIAL HEIGHT
# ============================================================

df["vial_height"] = (
    df["vial_y2"] -
    df["vial_y1"]
)


# ============================================================
# CALCULATE PHASE HEIGHT
# ============================================================

df["phase_height"] = (
    df["phase_y2"] -
    df["phase_y1"]
)


# ============================================================
# NORMALISE PHASE HEIGHT
# ============================================================

df["normalised_phase_height"] = (
    df["phase_height"] /
    df["vial_height"]
)


# ============================================================
# PHASES TO PLOT
# ============================================================

phases_to_plot = [
    "oil",
    "surfactant",
    "separating",
    "emulsion",
    "creaming",
    "continuous",
    "sedimentation"
]


# ============================================================
# CREATE ONE GRAPH FOR EACH VIAL / SAMPLE
# ============================================================

# Group by sample_id and vial_ind

for (sample_id, vial_ind), vial_data in df.groupby(
    ["sample_id", "vial_ind"]
):

    print(
        f"Creating graph for {sample_id}, "
        f"vial {vial_ind}"
    )


    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )


    # --------------------------------------------------------
    # Plot each phase
    # --------------------------------------------------------

    for phase_name in phases_to_plot:

        phase_data = vial_data[
            vial_data["phase_name"].str.lower()
            == phase_name
        ].copy()


        # Don't plot phases that were never detected

        if phase_data.empty:
            continue


        # Sort by time

        phase_data = phase_data.sort_values(
            "time_s"
        )


        # Plot phase

        ax.plot(
            phase_data["time_s"],
            phase_data["normalised_phase_height"],
            linewidth=2,
            marker="o",
            markersize=3,
            label=phase_name
        )


    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax.set_xlabel(
        "Time (s)",
        fontsize=14
    )

    ax.set_ylabel(
        "Normalised phase height",
        fontsize=14
    )


    ax.set_title(
        f"Vial {sample_id}",
        fontsize=18
    )


    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    ax.grid(
        True,
        alpha=0.3
    )


    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    ax.legend(
        fontsize=12
    )


    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    plt.tight_layout()


    # --------------------------------------------------------
    # Save graph
    # --------------------------------------------------------

    filename = (
        f"{sample_id}_vial_{vial_ind}"
        "_phase_height.png"
    )


    filepath = os.path.join(
        OUTPUT_FOLDER,
        filename
    )


    plt.savefig(
        filepath,
        dpi=300
    )


    plt.close()


print()
print("=" * 60)
print("DONE")
print("=" * 60)

print(
    f"Graphs saved in: {OUTPUT_FOLDER}"
)