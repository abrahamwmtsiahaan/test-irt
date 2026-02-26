import os

import dotenv
import matplotlib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

matplotlib.use("Agg")  # Headless mode for server/terminal execution
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from matplotlib.ticker import MultipleLocator

# Load environment variables
dotenv.load_dotenv()

DB_URL = os.getenv("DB_IRT_GENERATE_URL_PY")
TARGET_SAMPLES = 20000


def get_db_engine():
    return create_engine(DB_URL)


def main():
    kode_tob = 114399

    csv_file = "irt_114399.csv"
    if not os.path.exists(csv_file):
        print(f"❌ File '{csv_file}' tidak ditemukan.")
        return

    print(f"📂 Membaca {csv_file}...")
    try:
        df_excel = pd.read_csv(csv_file)
    except Exception as e:
        print(f"❌ Gagal membaca file CSV: {e}")
        return

    # Cleaning Column Names
    df_excel.columns = [str(c).upper().strip() for c in df_excel.columns]

    # Cleaning Data (Numeric & Non-Zero)
    subjects = ["PBM", "PPU", "PM", "LBIND", "LBING", "KK", "KPU"]

    for subj in subjects:
        if subj in df_excel.columns:
            df_excel[subj] = pd.to_numeric(df_excel[subj], errors="coerce")
            df_excel = df_excel[df_excel[subj].notna() & (df_excel[subj] != 0)]

    # Ensure NO_REGISTER is string & Sanitize (Prevent SQL Injection)
    if "NO_REGISTER" in df_excel.columns:
        df_excel["NO_REGISTER"] = df_excel["NO_REGISTER"].astype(str)
        df_excel["NO_REGISTER"] = df_excel["NO_REGISTER"].str.replace(
            r"[^a-zA-Z0-9]", "", regex=True
        )

    total_rows = len(df_excel)
    print(f"ℹ️ Total data valid Xcalibre: {total_rows}")

    # --- SAMPLING STRATEGY ---
    pool_target_size = int(TARGET_SAMPLES * 1.5)
    candidates_idx = set()

    # Fill sisa kuota pool dengan Random
    remaining_idx = df_excel.index.difference(candidates_idx)
    current_count = len(candidates_idx)

    if current_count < pool_target_size and len(remaining_idx) > 0:
        needed = pool_target_size - current_count
        sample_size = min(needed, len(remaining_idx))
        random_samples = np.random.choice(
            remaining_idx.values, sample_size, replace=False
        )
        candidates_idx.update(random_samples)

    df_pool = df_excel.loc[list(candidates_idx)]
    print(f"🔍 Mengirim {len(df_pool)} kandidat (Pool Awal) untuk dicek ke database...")

    engine = get_db_engine()

    try:
        with engine.connect() as conn:
            regs = df_pool["NO_REGISTER"].tolist()
            if not regs:
                print("❌ Tidak ada data register yang valid di pool.")
                return

            placeholders = ",".join([f"'{str(r)}'" for r in regs])

            # Check existence in DB
            sql_check = text(
                f"SELECT no_register FROM irt_nilai_to WHERE kode_tob = :tob AND no_register IN ({placeholders}) AND PBM > 200 AND PPU > 200 AND PM > 200 AND LBIND > 200 AND LBING > 200 AND KK > 200 AND KPU > 200"
            )
            existing = pd.read_sql(sql_check, conn, params={"tob": kode_tob})
            existing_regs = set(existing["no_register"].astype(str).tolist())

            # Filter pool to valid only
            df_valid_pool = df_pool[df_pool["NO_REGISTER"].isin(existing_regs)]
            print(f"✅ Ditemukan {len(df_valid_pool)} data yang valid di Database.")

            # --- FINAL SELECTION ---
            final_indices = set()
            for col in subjects:
                if col in df_valid_pool.columns:
                    df_v_sorted = df_valid_pool.sort_values(by=col)
                    if not df_v_sorted.empty:
                        final_indices.add(df_v_sorted.index[0])
                        final_indices.add(df_v_sorted.index[-1])
                        final_indices.add(df_v_sorted.index[len(df_v_sorted) // 2])

            if len(final_indices) < TARGET_SAMPLES:
                remaining = df_valid_pool.index.difference(final_indices)
                needed = TARGET_SAMPLES - len(final_indices)
                if len(remaining) > 0:
                    extras = np.random.choice(
                        remaining.values,
                        size=min(len(remaining), needed),
                        replace=False,
                    )
                    final_indices.update(extras)

            df_sample = df_excel.loc[list(final_indices)]

            if not df_sample.empty:
                print(
                    f"🚀 Mengambil data detail skor untuk {len(df_sample)} siswa terpilih..."
                )
                registers = df_sample["NO_REGISTER"].tolist()
                placeholders = ",".join([f"'{str(r)}'" for r in registers])

                # Get Scores
                db_cols = ", ".join(sorted(list(set([s.lower() for s in subjects]))))
                sql = text(
                    f"SELECT no_register, {db_cols} FROM irt_nilai_to WHERE no_register IN ({placeholders}) AND PBM IS NOT NULL AND PPU IS NOT NULL AND PM IS NOT NULL AND LBIND IS NOT NULL AND LBING IS NOT NULL AND KK IS NOT NULL AND KPU IS NOT NULL AND kode_tob = {kode_tob}"
                )
                df_db = pd.read_sql(sql, conn)

                # SAFETY: Deduplicate columns immediately
                df_db = df_db.loc[:, ~df_db.columns.duplicated()]

                df_db.columns = [c.upper() for c in df_db.columns]
                df_db["NO_REGISTER"] = df_db["NO_REGISTER"].astype(str)

                # Get Total Correct & Theta
                sql_tc = text(f"""
                    SELECT no_register, nama_kelompok_ujian, total_correct, theta
                    FROM analisis_nilai_irt
                    WHERE kode_tob = :tob AND no_register IN ({placeholders})
                """)
                df_tc = pd.read_sql(sql_tc, conn, params={"tob": kode_tob})
                df_tc.columns = [c.upper() for c in df_tc.columns]
                df_tc["NO_REGISTER"] = df_tc["NO_REGISTER"].astype(str)

                if not df_tc.empty:
                    # 1. Standarisasi/Mapping Nama Kelompok Ujian
                    map_mapel = {"LBI": "LBIND", "PK": "KK", "PU": "KPU"}
                    df_tc["NAMA_KELOMPOK_UJIAN"] = df_tc["NAMA_KELOMPOK_UJIAN"].replace(
                        map_mapel
                    )

                    # 2. Pivot Multi-Value (TOTAL_CORRECT & THETA)
                    df_tc_pivot = df_tc.pivot(
                        index="NO_REGISTER",
                        columns="NAMA_KELOMPOK_UJIAN",
                        values=["TOTAL_CORRECT", "THETA"],
                    )

                    # 3. Flatten MultiIndex Columns
                    df_tc_pivot.columns = [
                        f"TC_{col[1]}"
                        if col[0] == "TOTAL_CORRECT"
                        else f"THETA_{col[1]}"
                        for col in df_tc_pivot.columns
                    ]
                    df_tc_pivot = df_tc_pivot.reset_index()

                    # Merge ke database dataframe utama
                    df_db = pd.merge(df_db, df_tc_pivot, on="NO_REGISTER", how="left")

                merged = pd.merge(
                    df_sample,
                    df_db,
                    on="NO_REGISTER",
                    suffixes=("_Xcalibre", "_Generate"),
                )

                # Save Raw Data
                base_cols = ["NO_REGISTER"] + [
                    f"{s}_{suffix}"
                    for s in subjects
                    for suffix in ["Xcalibre", "Generate"]
                ]
                cols_to_save = [c for c in base_cols if c in merged.columns]
                cols_to_save += [
                    c
                    for c in merged.columns
                    if c.startswith("TC_") or c.startswith("THETA_")
                ]

                merged.to_csv("hasil_korelasi.csv", index=False, columns=cols_to_save)
                print(
                    f"📂 File 'hasil_korelasi.csv' berhasil dibuat ({len(merged)} baris)."
                )

                print("\n=== VISUALISASI JOINT PLOT & DISTRIBUSI ===")

                # Filter subjects that actually exist in both
                actual_available_subjects = [
                    subj
                    for subj in subjects
                    if f"{subj}_Xcalibre" in merged.columns
                    and f"{subj}_Generate" in merged.columns
                ]

                sns.set_theme(style="white", color_codes=True)
                print(f"Subjects to process: {actual_available_subjects}")

                for subj in actual_available_subjects:
                    col_x = f"{subj}_Xcalibre"
                    col_y = f"{subj}_Generate"

                    if col_x not in merged.columns or col_y not in merged.columns:
                        continue

                    try:
                        # ---------------------------------------------------------
                        # 1. GENERATE GRAFIK DISTRIBUSI NORMAL (GELOMBANG) TERPISAH
                        # ---------------------------------------------------------
                        plt.figure(figsize=(10, 6))
                        sns.kdeplot(
                            data=merged,
                            x=col_x,
                            fill=True,
                            color="#4C72B0",
                            label="Xcalibre",
                            alpha=0.5,
                        )
                        sns.kdeplot(
                            data=merged,
                            x=col_y,
                            fill=True,
                            color="#DD8452",
                            label="Generate",
                            alpha=0.5,
                        )

                        plt.title(
                            f"Distribusi Skor {subj} - Xcalibre vs Generate",
                            fontweight="bold",
                        )
                        plt.xlabel("Skor")
                        plt.ylabel("Densitas")
                        plt.legend()

                        dist_filename = f"distribusi_{subj}.png"
                        plt.savefig(dist_filename, dpi=300, bbox_inches="tight")
                        print(f"🌊 Grafik distribusi disimpan: {dist_filename}")
                        plt.close()

                        # ---------------------------------------------------------
                        # 2. GENERATE JOINT PLOT UTAMA
                        # ---------------------------------------------------------
                        r, p = stats.pearsonr(merged[col_x], merged[col_y])
                        jitter_amount = 2.0

                        plot_data = merged[[col_x, col_y]].copy()
                        plot_data[col_x] += np.random.uniform(
                            -jitter_amount, jitter_amount, size=len(plot_data)
                        )
                        plot_data[col_y] += np.random.uniform(
                            -jitter_amount, jitter_amount, size=len(plot_data)
                        )

                        g = sns.jointplot(
                            data=plot_data,
                            x=col_x,
                            y=col_y,
                            kind="reg",
                            truncate=False,
                            color="#4C72B0",
                            height=7,
                            scatter_kws={"s": 15, "alpha": 0.6},
                            line_kws={"color": "red", "linewidth": 1},
                        )

                        g.set_axis_labels(
                            f"Xcalibre Score ({subj})",
                            f"Generate Score ({subj})",
                            fontsize=12,
                            fontweight="bold",
                        )
                        g.ax_joint.xaxis.set_major_locator(MultipleLocator(50))
                        g.ax_joint.yaxis.set_major_locator(MultipleLocator(50))

                        g.ax_joint.text(
                            0.05,
                            0.95,
                            f"Pearson r = {r:.4f}\np = {p:.2e}",
                            transform=g.ax_joint.transAxes,
                            fontsize=12,
                            verticalalignment="top",
                            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                        )

                        joint_filename = f"korelasi_{subj}.png"
                        plt.savefig(joint_filename, dpi=300, bbox_inches="tight")
                        print(f"📈 Grafik korelasi disimpan: {joint_filename}")
                        plt.close()

                    except Exception as plot_e:
                        print(f"⚠️ Gagal me-render plot untuk {subj}: {plot_e}")
                        plt.close("all")

    except Exception as e:
        print(f"❌ Error Critical: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
