"""Join single-position sylva shots files (already in one frame) into one plot shots file, row group by row group.

usage: merge_shots.py OUT.parquet IN1.parquet IN2.parquet ...
Needs pyarrow (run with a Python that has it; sylva's venv does not). Each input must hold exactly one scanner
position; input i becomes scan i of the output, whose metadata lists the positions in input order, the union of the
echo bounds and the echo total, as Shots.save writes them for a concatenated Shots. Memory stays at one row group.
"""
import sys
import pyarrow as pa, pyarrow.parquet as pq


def merge(out, inputs):
    """Write the shots files inputs (paths, in order) as one shots file out; returns (number of inputs, total echoes).

    The sylva shots format is a Parquet file with one row per pulse; its first column, "scan", indexes the scanner positions
    listed in the file metadata (sylva.scans, "x y z;x y z;..."). Each input has one position, so its rows all hold 0; in
    the output they are set to the input's index i. The inputs must have the same columns and the single-precision format.
    """
    # lo, hi: running min and max of the echo bounds (x, y, z) over the inputs, starting from +-1e300
    files, scans, lo, hi, n_echoes = [pq.ParquetFile(p) for p in inputs], [], [1e300] * 3, [-1e300] * 3, 0
    schema = files[0].schema_arrow.remove_metadata()          # the columns alone; the metadata is rebuilt below
    for p, f in zip(inputs, files):                       # footers first: the output metadata is known before writing
        md = f.metadata.metadata; pos = md[b"sylva.scans"].decode().split(";")
        if len(pos) != 1: raise ValueError(f"{p}: {len(pos)} scanner positions, expected 1")
        if md[b"sylva.shots"] != b"1" or md[b"sylva.precision"] != b"single": raise ValueError(f"{p}: unexpected shots format")
        if f.schema_arrow.remove_metadata() != schema: raise ValueError(f"{p}: columns differ from the first input")
        scans.append(pos[0]); n_echoes += int(md[b"sylva.n_echoes"])
        b = [float(v) for v in md[b"sylva.bounds"].split()]   # "xmin ymin zmin xmax ymax zmax" of the echoes
        lo = [min(x, y) for x, y in zip(lo, b[:3])]; hi = [max(x, y) for x, y in zip(hi, b[3:])]
    # the output metadata: the first input's, with every position listed (scan i = input i), the origin offset at the first
    # position (as Shots.save sets it, the first pulse's origin), the echo total and the union of the bounds
    md = dict(files[0].metadata.metadata); md[b"sylva.scans"] = ";".join(scans).encode(); md[b"sylva.origin_offset"] = scans[0].encode()
    md[b"sylva.n_echoes"] = str(n_echoes).encode(); md[b"sylva.bounds"] = " ".join(repr(v) for v in lo + hi).encode()
    schema = schema.with_metadata(md)
    with pq.ParquetWriter(out, schema, compression="zstd", compression_level=3) as w:
        # copy row group by row group (keeping the input's group sizes), so memory stays at one group
        for i, f in enumerate(files):
            for g in range(f.metadata.num_row_groups):
                t = f.read_row_group(g); cols = t.columns
                cols[0] = pa.array([i] * t.num_rows, pa.int32())   # column 0 is "scan": point every pulse at position i
                w.write_table(pa.Table.from_arrays(cols, schema=schema), row_group_size=t.num_rows)
    return len(inputs), n_echoes


if __name__ == "__main__":
    print(merge(sys.argv[1], sys.argv[2:]))
