#!/usr/bin/env python3
import pandas as pd
import argparse
import os
import numpy as np


# ----------------------------
# Fixed settings
# ----------------------------
WINDOW_SIZE = 1  # fixed: do not expose as CLI option


############################################################
# Argument parsing
############################################################

def GetArguments():
    parser = argparse.ArgumentParser(
        description=(
            "Detect introgressed blocks per individual using per-site parental alleles p1/p2 (0/1), "
            "then MASK the FULL input SNP table (all rows) so that for each individual, genotypes "
            "OUTSIDE that individual's blocks are overwritten to the expected background homozygote.\n\n"
            "Important: Missing genotypes (.|.) are NEVER overwritten (inside or outside blocks)."
        )
    )

    parser.add_argument(
        "-c", "--input_file_name",
        type=str, required=True,
        help="Input TSV with columns: Chromosome, Position, p1, p2, and individual genotype columns."
    )

    parser.add_argument(
        "-i", "--individuals",
        type=str, required=True,
        help="Comma-separated list of individual IDs (column names) to process."
    )

    parser.add_argument(
        "--background_map",
        type=str, required=True,
        help="TSV/whitespace file with columns: Individual Background (p1 or p2). Extra columns are ignored."
    )

    parser.add_argument(
        "--max_distance",
        type=int,
        default=50000,
        help="Maximum bp distance between consecutive introgressed SNPs inside a block (default: 50000)."
    )

    parser.add_argument(
        "--min_snps",
        type=int,
        default=3,
        help="Minimum number of SNP rows (total, incl. non-intro/missing) in a block to keep it (default: 3)."
    )

    parser.add_argument(
        "--max_missing_prop",
        type=float,
        default=0.5,
        help="Discard blocks with missing / N_SNPs > this (default: 0.5)."
    )

    parser.add_argument(
        "--max_nonintro_prop",
        type=float,
        default=0.5,
        help=(
            "Discard blocks with non_intro / N_SNPs > this (default: 0.5). "
            "Non-intro = expected background homozygote per site (based on p1/p2 at that site)."
        )
    )

    parser.add_argument(
        "--count_hetero_as_introgressed",
        action="store_true",
        help=(
            "If set, heterozygotes containing the foreign allele (0|1 or 1|0) count as introgressed. "
            "If not set, only foreign homozygotes count."
        )
    )

    parser.add_argument(
        "--missing_breaks_streak",
        action="store_true",
        help=(
            "If set, missing genotypes (.|.) count toward the 'consecutive non-introgressed' streak break rule. "
            "If NOT set (default), missing does NOT increase the non-intro streak (more tolerant)."
        )
    )

    parser.add_argument(
        "--max_consecutive_nonintro",
        type=int,
        default=1,
        help=(
            "How many consecutive NON-introgressed (and possibly missing, depending on --missing_breaks_streak) "
            "SNPs are allowed while extending a block. "
            "Set 0 for strict: any first non-intro breaks immediately. Default: 1."
        )
    )

    parser.add_argument(
        "--masked_snps_out",
        type=str,
        required=True,
        help="Output TSV: full SNP table (same number of rows as input) with per-individual masking applied."
    )

    parser.add_argument(
        "--reuse_existing_blocks",
        action="store_true",
        help=(
            "If set: when blocks_<IND>_w1.tsv exists in the current working directory, "
            "reuse it instead of recomputing blocks for that individual."
        )
    )

    return parser.parse_args()


############################################################
# Helper functions
############################################################

def parse_alleles(genotype):
    """Return (a,b) from '0|1'. Return None for missing/malformed."""
    if not isinstance(genotype, str):
        return None
    if genotype == ".|.":
        return None
    if "|" not in genotype:
        return None
    a, b = genotype.split("|", 1)
    return (a, b)


def is_introgressed_foreign(genotype, foreign_allele, count_hetero):
    """
    True if genotype contains foreign allele (and not missing).
    If count_hetero=False -> only homozygous foreign counts.
    """
    alleles = parse_alleles(genotype)
    if alleles is None:
        return False
    a, b = alleles
    if count_hetero:
        return (a == foreign_allele) or (b == foreign_allele)
    return (a == foreign_allele) and (b == foreign_allele)


def expected_background_homozygote(bg_allele_int):
    """0/1 -> '0|0' or '1|1'."""
    return f"{bg_allele_int}|{bg_allele_int}"


def merge_intervals_touching(intervals):
    """
    Merge overlapping OR directly-touching intervals: s <= prev_end + 1.
    (No 'gap merging' beyond 1 bp.)
    """
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(a, b) for a, b in merged]


def build_keep_mask_for_intervals_positions(pos_array, intervals):
    """
    pos_array: 1D numpy array of Positions for a single chromosome, sorted like input rows
    intervals: merged list of (start,end) sorted, disjoint
    returns boolean array same length as pos_array: True if Position inside any interval
    """
    if len(pos_array) == 0 or not intervals:
        return np.zeros(len(pos_array), dtype=bool)

    starts = np.array([x[0] for x in intervals], dtype=int)
    ends = np.array([x[1] for x in intervals], dtype=int)

    k = np.searchsorted(starts, pos_array, side="right") - 1
    inside = (k >= 0) & (pos_array <= ends[k])
    return inside


############################################################
# Block-building function
############################################################

def sliding_window(
    df,
    individual_col,
    background_parent,
    window_size,
    max_distance,
    min_snps,
    max_missing_prop,
    max_nonintro_prop,
    count_hetero_as_introgressed,
    missing_breaks_streak,
    max_consecutive_nonintro
):
    """
    Build blocks for a single individual.
    df must be sorted by Chromosome, Position and reset_index(drop=True).
    background_parent must be 'p1' or 'p2'.
    """
    blocks = []
    n = len(df)
    ind_idx = df.columns.get_loc(individual_col)

    i = 0
    while i < n - window_size + 1:
        window = df.iloc[i:i + window_size]
        introgressed_indices = []

        # Find introgressed SNPs inside this window
        for idx_in_window, row_index in enumerate(window.index):
            geno = df.iat[row_index, ind_idx]
            foreign = str(df.at[row_index, "p2"]) if background_parent == "p1" else str(df.at[row_index, "p1"])
            if is_introgressed_foreign(geno, foreign, count_hetero_as_introgressed):
                introgressed_indices.append(idx_in_window)

        if not introgressed_indices:
            i += 1
            continue

        # Seed: first introgressed SNP in the window
        first_in_window = introgressed_indices[0]
        start_idx = window.index[first_in_window]
        start_pos = int(df.at[start_idx, "Position"])
        start_chr = df.at[start_idx, "Chromosome"]

        block_positions = [start_pos]
        block_indices = [start_idx]

        non_intro_streak = 0
        j = start_idx + 1

        # Extend block forward
        while j < n:
            if df.at[j, "Chromosome"] != start_chr:
                break

            pos_j = int(df.at[j, "Position"])
            geno_j = df.iat[j, ind_idx]
            is_missing = (geno_j == ".|.")

            foreign_j = str(df.at[j, "p2"]) if background_parent == "p1" else str(df.at[j, "p1"])

            if is_introgressed_foreign(geno_j, foreign_j, count_hetero_as_introgressed):
                # distance between consecutive introgressed SNPs
                if abs(pos_j - block_positions[-1]) > max_distance:
                    break
                non_intro_streak = 0
                block_positions.append(pos_j)
                block_indices.append(j)
            else:
                # not introgressed
                if is_missing and (not missing_breaks_streak):
                    # tolerate missing: do not increment streak
                    pass
                else:
                    non_intro_streak += 1
                    if non_intro_streak > max_consecutive_nonintro:
                        break

            j += 1

        end_idx = block_indices[-1]
        end_pos = block_positions[-1]

        block_df = df.iloc[start_idx:end_idx + 1]
        N_snps = len(block_df)

        if N_snps < min_snps:
            i = end_idx + 1
            continue

        geno_series = block_df.iloc[:, ind_idx]
        homo00 = (geno_series == "0|0").sum()
        homo11 = (geno_series == "1|1").sum()
        hetero = ((geno_series == "0|1") | (geno_series == "1|0")).sum()
        missing = (geno_series == ".|.").sum()

        # Non-introgressed = expected background homozygote per site
        if background_parent == "p1":
            bg_alleles = block_df["p1"].astype(int).to_numpy()
        else:
            bg_alleles = block_df["p2"].astype(int).to_numpy()

        genos = geno_series.to_numpy()
        non_intro = 0
        for g, bg_a in zip(genos, bg_alleles):
            if g == ".|.":
                continue
            if g == expected_background_homozygote(int(bg_a)):
                non_intro += 1

        missing_prop = missing / float(N_snps)
        if missing_prop > max_missing_prop:
            i = end_idx + 1
            continue

        nonintro_prop = non_intro / float(N_snps)
        if nonintro_prop > max_nonintro_prop:
            i = end_idx + 1
            continue

        block_length = end_pos - start_pos
        if block_length > 0:
            blocks.append((
                start_chr,
                individual_col,
                start_pos,
                end_pos,
                block_length,
                int(homo00),
                int(homo11),
                int(hetero),
                int(missing),
                int(N_snps),
                int(non_intro)
            ))

        i = end_idx + 1

    return blocks


def read_blocks_file(path):
    """
    Read an existing blocks TSV and return list of tuples compatible with blocks format:
    (Chromosome, Individual, Start, End, ...)
    Only Chromosome/Individual/Start/End are needed for masking.
    """
    dfb = pd.read_csv(path, sep="\t")
    needed = {"Chromosome", "Individual", "Start", "End"}
    if not needed.issubset(set(dfb.columns)):
        raise ValueError(f"Existing blocks file {path} missing columns: {needed - set(dfb.columns)}")
    out = []
    for _, r in dfb.iterrows():
        out.append((str(r["Chromosome"]), str(r["Individual"]), int(r["Start"]), int(r["End"]),
                    None, None, None, None, None, None, None))
    return out


############################################################
# Main
############################################################

def main():
    arg = GetArguments()

    print("Reading:", arg.input_file_name)
    df = pd.read_csv(arg.input_file_name, sep="\t")

    for col in ["Chromosome", "Position", "p1", "p2"]:
        if col not in df.columns:
            raise ValueError(f"Input file missing required column: {col}")

    # Ensure p1/p2 are int 0/1
    df["p1"] = pd.to_numeric(df["p1"], errors="raise").astype(int)
    df["p2"] = pd.to_numeric(df["p2"], errors="raise").astype(int)

    # Sort & reset index 0..n-1 (important for indexing masks)
    df = df.sort_values(["Chromosome", "Position"]).reset_index(drop=True)

    # Background map
    bg = pd.read_csv(
        arg.background_map,
        sep=r"\s+",
        engine="python",
        comment="#",
        usecols=[0, 1]
    )
    bg.columns = ["Individual", "Background"]
    bg["Individual"] = bg["Individual"].astype(str).str.strip()
    bg["Background"] = bg["Background"].astype(str).str.strip().str.lower()
    bg = bg[bg["Background"].isin(["p1", "p2"])]
    bg_map = dict(zip(bg["Individual"], bg["Background"]))

    individuals = [x.strip() for x in arg.individuals.split(",")]

    # Collect per-individual intervals for masking
    intervals_by_ind = {}  # ind -> chrom -> list of (start,end)

    # Also write blocks files (w1) for reproducibility/reuse
    for individual in individuals:
        if individual not in df.columns:
            print(f"Skipping {individual}: not found as a column in input.")
            continue
        if individual not in bg_map:
            print(f"Skipping {individual}: not found in background_map.")
            continue

        background_parent = bg_map[individual]
        if background_parent not in {"p1", "p2"}:
            print(f"Skipping {individual}: Background must be p1 or p2, got '{background_parent}'.")
            continue

        outname = f"blocks_{individual}_w{WINDOW_SIZE}.tsv"

        if arg.reuse_existing_blocks and os.path.isfile(outname):
            print(f"Reusing existing blocks: {outname}")
            blocks = read_blocks_file(outname)
        else:
            print(f"Processing {individual} (background={background_parent})")
            blocks = sliding_window(
                df=df,
                individual_col=individual,
                background_parent=background_parent,
                window_size=WINDOW_SIZE,
                max_distance=arg.max_distance,
                min_snps=arg.min_snps,
                max_missing_prop=arg.max_missing_prop,
                max_nonintro_prop=arg.max_nonintro_prop,
                count_hetero_as_introgressed=arg.count_hetero_as_introgressed,
                missing_breaks_streak=arg.missing_breaks_streak,
                max_consecutive_nonintro=arg.max_consecutive_nonintro
            )

            print(f"Writing {len(blocks)} blocks → {outname}")
            with open(outname, "w") as outfile:
                outfile.write(
                    "Chromosome\tIndividual\tStart\tEnd\tLenght\t"
                    "Homo_00_count\tHomo_11_count\tHeterozygote\tMissing\tN_SNPs\tNon_introgressed\n"
                )
                for b in blocks:
                    # blocks contain all fields in this order already
                    outfile.write(
                        f"{b[0]}\t{b[1]}\t{b[2]}\t{b[3]}\t{b[4]}"
                        f"\t{b[5]}\t{b[6]}\t{b[7]}\t{b[8]}"
                        f"\t{b[9]}\t{b[10]}\n"
                    )

        # store intervals
        intervals_by_ind.setdefault(individual, {})
        for b in blocks:
            chrom = str(b[0])
            start = int(b[2])
            end = int(b[3])
            intervals_by_ind[individual].setdefault(chrom, []).append((start, end))

    # Merge touching/overlapping intervals per individual+chrom (for clean masks)
    merged_intervals_by_ind = {}
    for ind, chrom_map in intervals_by_ind.items():
        merged_intervals_by_ind[ind] = {
            chrom: merge_intervals_touching(iv)
            for chrom, iv in chrom_map.items()
        }

    # ----------------------------
    # MASK FULL TABLE (same rows as input)
    # ----------------------------
    print("Masking full SNP table (all rows) outside each individual's blocks...")
    df_masked = df.copy()

    # Precompute per-chrom row indices and positions once (speed)
    chrom_groups = df_masked.groupby("Chromosome").groups  # chrom -> index labels (0..n-1)
    pos_all = df_masked["Position"].astype(int).to_numpy()

    for ind in individuals:
        if ind not in df_masked.columns:
            continue
        if ind not in bg_map:
            continue

        bg_parent = bg_map[ind]  # "p1" or "p2"
        if bg_parent not in {"p1", "p2"}:
            continue

        # Background homozygote per row based on p1/p2 at that site
        bg_allele = df_masked[bg_parent].astype(int).to_numpy()
        bg_homo = np.where(bg_allele == 0, "0|0", "1|1")

        keep_ind = np.zeros(len(df_masked), dtype=bool)

        ind_intervals = merged_intervals_by_ind.get(ind, {})

        for chrom, idx in chrom_groups.items():
            intervals = ind_intervals.get(chrom, [])
            if not intervals:
                continue
            idx_arr = np.fromiter(idx, dtype=int)
            pos_chr = pos_all[idx_arr]
            inside_chr = build_keep_mask_for_intervals_positions(pos_chr, intervals)
            keep_ind[idx_arr] = inside_chr

        geno = df_masked[ind].astype(str).to_numpy()
        is_missing = (geno == ".|.")

        outside = ~keep_ind

        # Missing is NEVER overwritten, anywhere
        to_overwrite = outside & (~is_missing)
        geno[to_overwrite] = bg_homo[to_overwrite]

        df_masked[ind] = geno

    # write full masked output
    df_masked.to_csv(arg.masked_snps_out, sep="\t", index=False)
    print("Wrote full masked SNP table:", arg.masked_snps_out)


if __name__ == "__main__":
    main()
