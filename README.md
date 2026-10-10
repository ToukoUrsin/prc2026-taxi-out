# PRC Data Challenge 2026 — taxi-out time (team `unbelievable-quartz`)

Gradient-boosted (LightGBM) taxi-out model for the [PRC Data Challenge 2026](https://prc-data-challenge-2026.netlify.app/).
Licence: GPL-3.0.

## Method

`train_nm.py` uses only columns present in the official ranking inputs (never `BLOCK_TIME_UTC_mvt` or the target):

- time differences between the takeoff time `MVT_TIME_UTC_mvt` and the schedule / NM times
  (`AOBT_3_flt`, `EOBT_1_flt`, `LOBT_flt`, `IOBT_flt`, `SCHED_TIME_UTC_mvt`, `ARVT_*`);
- airport traffic context computed from all supplied movements: departures in the 10/30 min before and 10 min after,
  arrivals around takeoff, NM off-blocks in the last 10/30 min, aircraft between off-block and takeoff, same-runway departures;
- categories: airport, destination, runway, stand, aircraft type, operator, market segment, wake, flight type/rule;
- hour, day of week, month, diversion flag.

Jan + Jul 2025 are held out for early stopping; the model is then refit on all 2025 departures with 1.1x the best
iteration count. Two targets are trained:

- `--target y`: taxi-out seconds directly (no clipping);
- `--target resid`: taxi-out minus a base (takeoff minus `AOBT_3_flt` when present, else the airport median).

The submitted predictions are the equal-weight average of the two (`blend.py`).

## Reproduce

Download the official files from the competition bucket (`prc-2026-datasets`) into `data/`
(`training/training_2025-*.parquet`, `ranking.parquet`, `submitting.parquet`, `final_ranking.parquet`,
`final_submitting.parquet`), then:

```bash
pip install -r requirements.txt
for t in y resid; do
  python train_nm.py --training 'data/training/*.parquet' --out out-$t --threads 4 --target $t --clip $([ $t = y ] && echo 1000000 || echo 20000) \
    --predict data/ranking.parquet:data/submitting.parquet:ranking.parquet \
              data/final_ranking.parquet:data/final_submitting.parquet:final.parquet
done
python blend.py out-y/ranking.parquet out-resid/ranking.parquet unbelievable-quartz_vN.parquet
python blend.py out-y/final.parquet out-resid/final.parquet unbelievable-quartz_final.parquet
```

Each run takes about 15 minutes on 4 vCPUs / 8 GiB. `out-*/log.json` records hold-out scores, feature importance and file hashes.

## Results (official leaderboard RMSE, seconds, Jan + Jul 2026)

| Submission | Model | RMSE |
|---|---|---|
| v1 | earlier entry without NM timing fields | 478.74 |
| v2 | `--target y`, labels clipped at 7,200 s | 495.00 |
| v3 | `--target resid` | 415.39 |
| v4 | `--target y`, unclipped | 392.37 |
| v5 | average of v3 and v4 | **389.12** |

Hold-out (Jan + Jul 2025) RMSE: 529.5 (v2 model), 488.1 (resid), 463.6 (y unclipped). Roughly half of the squared
error comes from rare departures with taxi-out over one hour, many without an NM flight match.
`unbelievable-quartz_final.parquet` (Jan, Feb, Jun, Jul 2026) uses the same blend.
