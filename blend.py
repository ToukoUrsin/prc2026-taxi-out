"""Average two prediction files that share the same template order (simple equal-weight blend)."""
import sys
import pyarrow as pa, pyarrow.parquet as pq

a, b, out = sys.argv[1:4]
ta, tb = pq.read_table(a), pq.read_table(b)
assert ta['MVT_ID_mvt'].equals(tb['MVT_ID_mvt'])
v = (ta['TAXITIME_SEC_mvt'].to_numpy() + tb['TAXITIME_SEC_mvt'].to_numpy()) / 2
pq.write_table(pa.Table.from_arrays([ta['MVT_ID_mvt'], pa.array(v, type=pa.float64())],
                                    names=['MVT_ID_mvt', 'TAXITIME_SEC_mvt']), out)
