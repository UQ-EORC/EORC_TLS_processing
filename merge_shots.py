"""Join single-position sylva shots files (already in one frame) into one plot shots file, row group by row group.

usage: merge_shots.py OUT.parquet IN1.parquet IN2.parquet ...
Needs pyarrow (run with a Python that has it; sylva's venv does not). Each input must hold exactly one scanner
position; input i becomes scan i of the output, whose metadata lists the positions in input order, the union of the
echo bounds and the echo total, as Shots.save writes them for a concatenated Shots. Memory stays at one row group.
"""
import sys
import pyarrow as pa, pyarrow.parquet as pq


def merge(out, inputs):
    files, scans, lo, hi, n_echoes = [pq.ParquetFile(p) for p in inputs], [], [1e300] * 3, [-1e300] * 3, 0
    schema = files[0].schema_arrow.remove_metadata()
    for p, f in zip(inputs, files):                       # footers first: the output metadata is known before writing
        md = f.metadata.metadata; pos = md[b"sylva.scans"].decode().split(";")
        if len(pos) != 1: raise ValueError(f"{p}: {len(pos)} scanner positions, expected 1")
        if md[b"sylva.shots"] != b"1" or md[b"sylva.precision"] != b"single": raise ValueError(f"{p}: unexpected shots format")
        if f.schema_arrow.remove_metadata() != schema: raise ValueError(f"{p}: columns differ from the first input")
        scans.append(pos[0]); n_echoes += int(md[b"sylva.n_echoes"])
        b = [float(v) for v in md[b"sylva.bounds"].split()]
        lo = [min(x, y) for x, y in zip(lo, b[:3])]; hi = [max(x, y) for x, y in zip(hi, b[3:])]
    md = dict(files[0].metadata.metadata); md[b"sylva.scans"] = ";".join(scans).encode(); md[b"sylva.origin_offset"] = scans[0].encode()
    md[b"sylva.n_echoes"] = str(n_echoes).encode(); md[b"sylva.bounds"] = " ".join(repr(v) for v in lo + hi).encode()
    schema = schema.with_metadata(md)
    with pq.ParquetWriter(out, schema, compression="zstd", compression_level=3) as w:
        for i, f in enumerate(files):
            for g in range(f.metadata.num_row_groups):
                t = f.read_row_group(g); cols = t.columns
                cols[0] = pa.array([i] * t.num_rows, pa.int32())
                w.write_table(pa.Table.from_arrays(cols, schema=schema), row_group_size=t.num_rows)
    return len(inputs), n_echoes


if __name__ == "__main__":
    print(merge(sys.argv[1], sys.argv[2:]))
