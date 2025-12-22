#!/usr/bin/env python3
import pandas as pd
import argparse
import os
import numpy as np


############################################################
# Argument parsing
############################################################

def GetArguments():
    parser = argparse.ArgumentParser(
        description=(
            "Detect introgressed blocks for individuals using per-site parental alleles p1/p2 (0/1). "
            "Introgressed sites are those where an individual's genotype contains the FOREIGN parental allele "
            "relative to its background (p1 or p2). Optionally outputs a filtered input-like SNP table "
            "containing only SNPs that fall inside ANY kept block across all processed individuals.\n\n"
            "NEW: Can additionally 'mask' genotypes OUTSIDE each individual's own blocks to the expected "
            "background homozygote (based on per-site p1/p2 + individual's background parent)."
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
        "-w", "--window_size",
        type=int, required=True,
        help="Window size (number of SNPs) used to find an initial introgressed site."
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
            "If set, missing genotypes (.|.) count toward the 'two consecutive non-introgressed' break rule. "
            "If NOT set (default), missing does NOT increase the non-intro streak (more tolerant)."
        )
    )

    parser.add_argument(
        "--filtered_snps_out",
        type=str,
        default=None,
        help=(
            "If set, write an input-like SNP table filtered to positions within ANY kept block across ALL "
            "processed individuals."
        )
    )

    # NEW masking options
    parser.add_argument(
        "--mask_outside_blocks",
        action="store_true",
        help=(
            "If set AND --filtered_snps_out is used: for each individual column, overwrite genotypes OUTSIDE that "
            "individual's own blocks with the expected background homozygote for that site (based on p1/p2 and background_map)."
        )
    )

    parser.add_argument(
        "--mask_missing_too",
        action="store_true",
        help=(
            "If set: also overwrite missing genotypes (.|.) outside blocks to the background homozygote. "
            "If not set (default): keep missing as .|. even outside blocks."
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


def merge_intervals(intervals):
    """
    intervals: list of (start, end)
    returns merged list of (start, end) non-overlapping, sorted.
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


def build_global_keep_mask(df, merged_by_chr):
    """
    df must have Chromosome and Position.
    merged_by_chr: dict chrom -> list of merged (start,end)
    returns boolean numpy mask of rows to keep.
    """
    keep = np.zeros(len(df), dtype=bool)
    pos = df["Position"].astype(int).to_numpy()

    for chrom, sub_idx in df.groupby("Chromosome").groups.items():
        if chrom not in merged_by_chr:
            continue

        intervals = merged_by_chr[chrom]
        starts = np.array([x[0] for x in intervals], dtype=int)
        ends = np.array([x[1] for x in intervals], dtype=int)

        idx_arr = np.fromiter(sub_idx, dtype=int)
        p = pos[idx_arr]

        k = np.searchsorted(starts, p, side="right") - 1
        inside = (k >= 0) & (p <= ends[k])
        keep[idx_arr] = inside

    return keep


# NEW helpers for per-individual masking
def merge_intervals_touching(intervals):
    """
    Union of overlapping OR directly-touching intervals: (s <= prev_end + 1).
    This is NOT 'merge by gap'. It does not bridge gaps > 1 bp.
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


def build_keep_mask_for_intervals(df_sub, intervals):
    """
    df_sub: dataframe already filtered to one chromosome, must have Position int
    intervals: merged list of (start,end) sorted, disjoint
    returns boolean numpy array: True if Position inside any interval
    """
    if not intervals or len(df_sub) == 0:
        return np.zeros(len(df_sub), dtype=bool)

    starts = np.array([x[0] for x in intervals], dtype=int)
    ends = np.array([x[1] for x in intervals], dtype=int)
    p = df_sub["Position"].astype(int).to_numpy()

    k = np.searchsorted(starts, p, side="right") - 1
    inside = (k >= 0) & (p <= ends[k])
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
    missing_breaks_streak
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

        # Identify introgressed SNPs in this window
        for idx_in_window, row_index in enumerate(window.index):
            geno = df.iat[row_index, ind_idx]
            if background_parent == "p1":
                foreign = str(df.at[row_index, "p2"])
            else:
                foreign = str(df.at[row_index, "p1"])

            if is_introgressed_foreign(geno, foreign, count_hetero_as_introgressed):
                introgressed_indices.append(idx_in_window)

        if not introgressed_indices:
            i += 1
            continue

        # Block start = first introgressed SNP in this window
        first_in_window = introgressed_indices[0]
        start_idx = window.index[first_in_window]
        start_pos = int(df.at[start_idx, "Position"])
        start_chr = df.at[start_idx, "Chromosome"]

        block_positions = [start_pos]
        block_indices = [start_idx]

        non_intro_streak = 0
        j = start_idx + 1

        # Extend block
        while j < n:
            if df.at[j, "Chromosome"] != start_chr:
                break

            pos_j = int(df.at[j, "Position"])
            geno_j = df.iat[j, ind_idx]
            is_missing = (geno_j == ".|.")

            if background_parent == "p1":
                foreign_j = str(df.at[j, "p2"])
            else:
                foreign_j = str(df.at[j, "p1"])

            if is_introgressed_foreign(geno_j, foreign_j, count_hetero_as_introgressed):
                non_intro_streak = 0

                # Distance check between consecutive introgressed SNPs
                if abs(pos_j - block_positions[-1]) > max_distance:
                    break

                block_positions.append(pos_j)
                block_indices.append(j)
            else:
                if is_missing and (not missing_breaks_streak):
                    pass
                else:
                    non_intro_streak += 1
                    if non_intro_streak >= 2:
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
        non_intro = 0
        if background_parent == "p1":
            bg_alleles = block_df["p1"].astype(int).to_numpy()
        else:
            bg_alleles = block_df["p2"].astype(int).to_numpy()

        genos = geno_series.to_numpy()
        for g, bg_a in zip(genos, bg_alleles):
            if g == ".|.":
                continue
            if g == expected_background_homozygote(int(bg_a)):
                non_intro += 1

        # Filters
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

    # Sort & reset index 0..n-1
    df = df.sort_values(["Chromosome", "Position"]).reset_index(drop=True)

    # Read background map robustly (tabs OR spaces, ignore extra columns, allow comments)
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
    bg_map = dict(zip(bg["Individual"], bg["Background"]))

    individuals = [x.strip() for x in arg.individuals.split(",")]

    all_blocks = []        # blocks across all individuals for global SNP filtering
    intervals_by_ind = {}  # individual -> chrom -> list of (start,end) for masking

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

        print(f"Processing {individual} (background={background_parent})")

        blocks = sliding_window(
            df=df,
            individual_col=individual,
            background_parent=background_parent,
            window_size=arg.window_size,
            max_distance=arg.max_distance,
            min_snps=arg.min_snps,
            max_missing_prop=arg.max_missing_prop,
            max_nonintro_prop=arg.max_nonintro_prop,
            count_hetero_as_introgressed=arg.count_hetero_as_introgressed,
            missing_breaks_streak=arg.missing_breaks_streak
        )

        all_blocks.extend(blocks)

        # store per-individual intervals for optional masking later
        intervals_by_ind.setdefault(individual, {})
        for b in blocks:
            chrom = b[0]
            start = int(b[2])
            end = int(b[3])
            intervals_by_ind[individual].setdefault(chrom, []).append((start, end))

        outname = f"blocks_{individual}_w{arg.window_size}.tsv"
        if os.path.isfile(outname):
            print(f"File {outname} exists. Remove it first.")
            continue

        print(f"Writing {len(blocks)} blocks → {outname}")
        with open(outname, "w") as outfile:
            outfile.write(
                "Chromosome\tIndividual\tStart\tEnd\tLenght\t"
                "Homo_00_count\tHomo_11_count\tHeterozygote\tMissing\tN_SNPs\tNon_introgressed\n"
            )
            for block in blocks:
                outfile.write(
                    f"{block[0]}\t{block[1]}\t{block[2]}\t{block[3]}\t{block[4]}"
                    f"\t{block[5]}\t{block[6]}\t{block[7]}\t{block[8]}"
                    f"\t{block[9]}\t{block[10]}\n"
                )

    # Optional: write a filtered input-like SNP file containing only SNPs within ANY block
    if arg.filtered_snps_out is not None:
        if len(all_blocks) == 0:
            print("No blocks found across individuals; not writing filtered SNP file.")
            return

        intervals_by_chr = {}
        for b in all_blocks:
            chrom = b[0]
            start = int(b[2])
            end = int(b[3])
            intervals_by_chr.setdefault(chrom, []).append((start, end))

        merged_by_chr = {chrom: merge_intervals(iv) for chrom, iv in intervals_by_chr.items()}
        total_merged = sum(len(v) for v in merged_by_chr.values())
        print(f"Merged to {total_merged} non-overlapping intervals across chromosomes.")

        keep_mask = build_global_keep_mask(df, merged_by_chr)
        kept = int(keep_mask.sum())
        print(f"Writing filtered SNP table: keeping {kept}/{len(df)} rows ({kept/len(df)*100:.2f}%).")

        df_filtered = df.loc[keep_mask].copy()
        # CRITICAL FIX: reset index so groupby().groups gives 0..n-1 indices
        df_filtered = df_filtered.reset_index(drop=True)

        # NEW: mask outside each individual's own blocks to background homozygote
        if arg.mask_outside_blocks:
            print("Masking genotypes outside each individual's blocks to background homozygote...")

            # merge touching/overlapping intervals for each individual+chrom
            merged_intervals_by_ind = {}
            for ind, chrom_map in intervals_by_ind.items():
                merged_intervals_by_ind[ind] = {
                    chrom: merge_intervals_touching(iv)
                    for chrom, iv in chrom_map.items()
                }

            # apply masking per individual column
            for ind in individuals:
                if ind not in df_filtered.columns:
                    continue
                if ind not in bg_map:
                    continue

                bg_parent = bg_map[ind]  # "p1" or "p2"
                if bg_parent not in {"p1", "p2"}:
                    continue

                # expected background allele per row (0/1) and homozygote string
                bg_allele = df_filtered[bg_parent].astype(int).to_numpy()
                bg_homo = np.where(bg_allele == 0, "0|0", "1|1")

                keep_ind = np.zeros(len(df_filtered), dtype=bool)

                # chromosome-by-chromosome membership test
                for chrom, idx in df_filtered.groupby("Chromosome").groups.items():
                    idx_arr = np.fromiter(idx, dtype=int)

                    intervals = merged_intervals_by_ind.get(ind, {}).get(chrom, [])
                    if not intervals:
                        continue

                    sub = df_filtered.loc[idx_arr, ["Position"]]
                    inside = build_keep_mask_for_intervals(sub, intervals)
                    keep_ind[idx_arr] = inside

                geno = df_filtered[ind].astype(str).to_numpy()
                outside = ~keep_ind

                if arg.mask_missing_too:
                    geno[outside] = bg_homo[outside]
                else:
                    is_missing = (geno == ".|.")
                    to_overwrite = outside & (~is_missing)
                    geno[to_overwrite] = bg_homo[to_overwrite]

                df_filtered[ind] = geno

        df_filtered.to_csv(arg.filtered_snps_out, sep="\t", index=False)
        print("Wrote:", arg.filtered_snps_out)


if __name__ == "__main__":
    main()
