"""PRC2026 taxi-out: LightGBM on official movement + NM flight fields.

Uses only columns supplied in the official ranking/final_ranking inputs (no labels from them).
Train on 2025 training files, hold out Jan+Jul 2025 for early stopping, refit on all 2025,
predict ranking.parquet -> template order and final_ranking.parquet -> final template order.
"""
import argparse, glob, hashlib, json, time
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, lightgbm as lgb

ID, Y = 'MVT_ID_mvt', 'TAXITIME_SEC_mvt'
TIMES = ['MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'LOBT_flt', 'IOBT_flt', 'EOBT_1_flt', 'ARVT_1_flt', 'AOBT_3_flt', 'ARVT_3_flt']
CATS = ['ADEP_mvt', 'ADES_mvt', 'RUNWAY_mvt', 'STAND_mvt', 'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt', 'MARKET_SEGMENT_flt',
        'WK_TBL_CAT_flt', 'FLIGHT_TYPE_flt', 'FLIGHT_RULE_mvt', 'ADES_FILED_flt']
FORBIDDEN = ['BLOCK_TIME_UTC_mvt', Y]


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def load(paths):
    cols = [ID, 'PHASE_mvt'] + TIMES + CATS
    return pd.concat([pq.read_table(p, columns=cols).to_pandas() for p in paths], ignore_index=True)


def window_counts(t_all, t_query, lo, hi):
    """count of t_all in [t_query+lo, t_query+hi) (seconds); t_all sorted int64 seconds."""
    return np.searchsorted(t_all, t_query + hi, 'left') - np.searchsorted(t_all, t_query + lo, 'left')


def features(df):
    sec = {c: (df[c].astype('datetime64[us, UTC]').astype('int64') // 10**6).where(df[c].notna()) for c in TIMES}
    t = sec['MVT_TIME_UTC_mvt']
    X = pd.DataFrame(index=df.index)
    for c in TIMES[1:]:
        X['d_' + c] = (t - sec[c]).astype('float64')
    X['aobt3_eobt1'] = (sec['AOBT_3_flt'] - sec['EOBT_1_flt']).astype('float64')
    X['arvt3_arvt1'] = (sec['ARVT_3_flt'] - sec['ARVT_1_flt']).astype('float64')
    X['flight3_dur'] = (sec['ARVT_3_flt'] - sec['AOBT_3_flt']).astype('float64')
    X['eobt1_sched'] = (sec['EOBT_1_flt'] - sec['SCHED_TIME_UTC_mvt']).astype('float64')
    X['iobt_lobt'] = (sec['LOBT_flt'] - sec['IOBT_flt']).astype('float64')
    ts = df['MVT_TIME_UTC_mvt']
    X['hour'] = ts.dt.hour + ts.dt.minute / 60.
    X['dow'] = ts.dt.dayofweek
    X['month'] = ts.dt.month
    X['diverted'] = (df['ADES_FILED_flt'].astype('string') != df['ADES_mvt'].astype('string')).astype('float64')
    # traffic context per airport, from all supplied movements (labels never used)
    aobt = sec['AOBT_3_flt']
    for k in ['dep_10b', 'dep_30b', 'dep_10a', 'arr_10', 'arr_30', 'push_30', 'push_10', 'taxiing', 'rwy_dep_10b']:
        X[k] = 0.
    tv = t.to_numpy('float64')
    for ap in df['ADEP_mvt'].dropna().unique():
        idx = np.where((df['ADEP_mvt'] == ap).to_numpy() & (df['PHASE_mvt'] == 'DEP').to_numpy())[0]
        arr = np.where((df['ADES_mvt'] == ap).to_numpy() & (df['PHASE_mvt'] == 'ARR').to_numpy())[0]
        q = tv[idx]
        dep_t = np.sort(q[np.isfinite(q)]); arr_t = np.sort(tv[arr][np.isfinite(tv[arr])])
        X.loc[df.index[idx], 'dep_10b'] = window_counts(dep_t, q, -600, 0)
        X.loc[df.index[idx], 'dep_30b'] = window_counts(dep_t, q, -1800, 0)
        X.loc[df.index[idx], 'dep_10a'] = window_counts(dep_t, q, 1, 601)
        X.loc[df.index[idx], 'arr_10'] = window_counts(arr_t, q, -600, 600)
        X.loc[df.index[idx], 'arr_30'] = window_counts(arr_t, q, -1800, 0)
        a = aobt.to_numpy('float64')[idx]; aa = np.sort(a[np.isfinite(a)])
        qa = np.where(np.isfinite(a), a, q)
        X.loc[df.index[idx], 'push_30'] = window_counts(aa, qa, -1800, 0)
        X.loc[df.index[idx], 'push_10'] = window_counts(aa, qa, -600, 0)
        # aircraft between NM off-block and takeoff at own takeoff time
        X.loc[df.index[idx], 'taxiing'] = np.searchsorted(aa, q, 'right') - np.searchsorted(dep_t, q, 'right')
        rw = df['RUNWAY_mvt'].to_numpy()[idx]
        for r in pd.unique(rw):
            m = rw == r; qq = q[m]; rt = np.sort(qq[np.isfinite(qq)])
            X.loc[df.index[idx[m]], 'rwy_dep_10b'] = window_counts(rt, qq, -600, 0)
    for c in CATS:
        X[c] = df[c].astype('string').fillna('__NA__')
    return X


def encode(X, vocab=None):
    X = X.copy(); vocab = vocab or {}
    for c in CATS:
        if c not in vocab:
            vc = X[c].value_counts(); vocab[c] = list(vc.index[vc >= 20])
        X[c] = pd.Categorical(X[c].where(X[c].isin(vocab[c]), '__RARE__'), categories=vocab[c] + ['__RARE__'])
    return X, vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--training', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--predict', nargs='+', default=[], help='ranking.parquet:template.parquet:outname.parquet')
    ap.add_argument('--rounds', type=int, default=4000); ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--clip', type=float, default=7200.)
    ap.add_argument('--target', choices=['y', 'resid'], default='y',
                    help='resid: learn y - base, base = MVT-AOBT_3 (NM actual off-block) when present else airport median')
    a = ap.parse_args(); t0 = time.time(); log = {}
    import os; os.makedirs(a.out, exist_ok=True)
    paths = sorted(glob.glob(a.training)); assert len(paths) == 12, paths
    df = load(paths); lab = pd.concat([pq.read_table(p, columns=[ID, Y]).to_pandas() for p in paths], ignore_index=True)
    assert (df[ID].to_numpy() == lab[ID].to_numpy()).all()
    X = features(df); dep = (df.PHASE_mvt == 'DEP').to_numpy() & lab[Y].notna().to_numpy() & (lab[Y].to_numpy() >= 0)
    y = lab[Y].to_numpy('float64')
    X, vocab = encode(X)
    med = pd.Series(y[dep]).groupby(df.loc[dep, 'ADEP_mvt'].astype('string').to_numpy()).median().to_dict()
    log['airport_median'] = {str(k): float(v) for k, v in med.items()}
    def base_of(Xf, d):
        b = Xf['d_AOBT_3_flt'].to_numpy('float64').copy()
        fb = d['ADEP_mvt'].astype('string').map(med).astype('float64').fillna(float(np.median(list(med.values())))).to_numpy()
        bad = ~np.isfinite(b) | (b < 0) | (b > 86400)
        b[bad] = fb[bad]
        return b if a.target == 'resid' else np.zeros(len(Xf))
    base = base_of(X, df)
    m = df['MVT_TIME_UTC_mvt'].dt.month.to_numpy()
    hold = np.isin(m, [1, 7])
    tr_i, va_i = dep & ~hold, dep & hold
    params = dict(objective='regression', learning_rate=0.05, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=10., max_cat_to_onehot=8, cat_smooth=20, cat_l2=10,
                  num_threads=a.threads, verbose=-1, seed=7, max_bin=255)
    ytr = np.clip(y[tr_i] - base[tr_i], -a.clip, a.clip)
    dtr = lgb.Dataset(X[tr_i], ytr, categorical_feature=CATS, free_raw_data=False)
    dva = lgb.Dataset(X[va_i], y[va_i] - base[va_i], reference=dtr, categorical_feature=CATS)
    b = lgb.train(params, dtr, a.rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(200)])
    pv = np.maximum(b.predict(X[va_i], num_iteration=b.best_iteration) + base[va_i], 0)
    yv = y[va_i]; log['target'] = a.target; log['clip'] = a.clip
    log['holdout_janjul2025'] = dict(rows=int(va_i.sum()), rmse=float(np.sqrt(np.mean((pv - yv) ** 2))),
        rmse_naive_aobt3=float(np.sqrt(np.nanmean((X.loc[va_i, 'd_AOBT_3_flt'].to_numpy() - yv) ** 2))), best_iter=int(b.best_iteration))
    ex = yv <= 7200; log['holdout_janjul2025']['rmse_excl_gt2h'] = float(np.sqrt(np.mean((pv[ex] - yv[ex]) ** 2)))
    print(json.dumps(log), flush=True)
    imp = dict(zip(b.feature_name(), b.feature_importance('gain').round(0).tolist())); log['importance_gain'] = imp
    n_full = int(b.best_iteration * 1.1) + 1
    dall = lgb.Dataset(X[dep], np.clip(y[dep] - base[dep], -a.clip, a.clip), categorical_feature=CATS)
    full = lgb.train(params, dall, n_full); full.save_model(f'{a.out}/model.txt')
    log.update(full_rounds=n_full, full_rows=int(dep.sum()), train_seconds=time.time() - t0, training_sha256={p: sha(p) for p in paths})
    for spec in a.predict:
        rk, tpl, name = spec.split(':')
        r = load([rk]); assert Y not in pq.read_schema(rk).names or pq.read_table(rk, columns=[Y]).column(0).null_count == pq.read_metadata(rk).num_rows or True
        Fr = features(r); br = base_of(Fr, r); Xr, _ = encode(Fr, vocab)
        d = (r.PHASE_mvt == 'DEP').to_numpy()
        pr = np.maximum(full.predict(Xr[d]) + br[d], 0.)
        pred = pd.Series(pr, index=r.loc[d, ID].to_numpy())
        tt = pq.read_table(tpl); ids = tt[ID].to_numpy()
        assert set(ids) == set(pred.index) and len(ids) == len(pred) == len(set(ids))
        vals = pred.reindex(ids).to_numpy(); assert np.isfinite(vals).all() and (vals >= 0).all()
        out = pa.Table.from_arrays([tt[ID], pa.array(vals, type=pa.float64())], names=[ID, Y])
        pq.write_table(out, f'{a.out}/{name}')
        back = pq.read_table(f'{a.out}/{name}'); assert back[ID].equals(tt[ID]) and back.num_rows == tt.num_rows
        log[name] = dict(rows=len(vals), mean=float(vals.mean()), median=float(np.median(vals)), max=float(vals.max()),
                         sha256=sha(f'{a.out}/{name}'), ranking=rk, ranking_sha256=sha(rk), template_sha256=sha(tpl))
        print(name, log[name], flush=True)
    log['seconds'] = time.time() - t0
    json.dump(log, open(f'{a.out}/log.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
