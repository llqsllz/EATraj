"""Fit empirical development envelopes using training HISTORY endpoints only."""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from trajectory_prediction.artifacts import start_run,finish_run,write_json,sha256_file
from trajectory_prediction.packed_data import PackedSceneDataset

def weighted_quantile(values,weights,q):
    order=np.argsort(values,kind="stable");v=np.asarray(values)[order];w=np.asarray(weights)[order]
    return float(v[min(np.searchsorted(np.cumsum(w),q*w.sum(),side="left"),len(v)-1)])

def main():
    p=argparse.ArgumentParser();p.add_argument("--data",default="data/trajectory_joint_v3/dev_20260921")
    p.add_argument("--quantile",type=float,default=.999);p.add_argument("--margin",type=float,default=1.1)
    args=p.parse_args()
    if not 0<args.quantile<1 or args.margin<1:raise ValueError("Invalid empirical envelope settings")
    ds=PackedSceneDataset(ROOT/args.data,"train",20)
    config=dict(vars(args),fit_split="train",
                fit_observations="history endpoint values only; future values are not used in fitting; cache integrity checks read target file bytes",
                fit_scope={"split":"train","observation":"history_endpoints","uses_future_values":False},
                dataset_config_sha256=sha256_file(ROOT/args.data/"dataset_config.json"),
                training_scene_index_sha256=sha256_file(ROOT/args.data/"scene_index.jsonl"),
                split_manifest_sha256=sha256_file(ROOT/args.data/"split_manifest.csv"),
                cache_manifest_sha256=sha256_file(ds.root/"manifest.json"),array_integrity=ds.integrity,
                interpretation="empirical development envelope, not a certified aircraft flight envelope",
                altitude_unit_and_datum_verified=False,independent_generation_families_verified=False,
                acceleration_estimator="least-squares slope of causal historical velocity over final 1s",
                aggregation="equal candidate-prefix groups / replays / windows / eligible actors")
    run,record=start_run("training-history-motion-bounds",config,[Path(__file__)])
    try:
        group_replays=defaultdict(set);replay_windows=Counter()
        for row in ds.rows:group_replays[row["family_id"]].add(row["replay_id"]);replay_windows[row["replay_id"]]+=1
        values={k:[] for k in ("Vmax","Wmax","Amax")};weights={k:[] for k in values};skips=Counter()
        # Only history arrays are read below. Dataset targets are never indexed.
        for row in ds.rows:
            lo,n=row["actor_offset"],row["actor_count"]
            f=np.asarray(ds.arrays["history_features"][lo:lo+n],dtype=np.float64)
            times=f[0,:,-1]*10;recent=times>=times[-1]-1.-1e-6
            base=1/(len(group_replays)*len(group_replays[row["family_id"]])*replay_windows[row["replay_id"]])
            endpoint=f[:,-1,7]>.5
            v=f[:,-1,3:6]*100
            for key,x in (("Vmax",np.linalg.norm(v,axis=1)),("Wmax",np.abs(v[:,2]))):
                if endpoint.any():values[key].extend(x[endpoint].tolist());weights[key].extend([base/endpoint.sum()]*int(endpoint.sum()))
                else:skips[key]+=1
            accel=[]
            for actor in range(n):
                keep=recent&(f[actor,:,7]>.5);t=times[keep]
                if len(t)<3 or t[-1]-t[0]<.4:continue
                centered=t-t.mean();slope=(centered[:,None]*(f[actor,keep,3:6]*100)).sum(axis=0)/(centered@centered)
                accel.append(float(np.linalg.norm(slope)))
            if accel:values["Amax"].extend(accel);weights["Amax"].extend([base/len(accel)]*len(accel))
            else:skips["Amax"]+=1
        bounds={k:args.margin*weighted_quantile(values[k],weights[k],args.quantile) for k in values}
        output=dict(config,bounds=bounds,statistics={k:dict(n_actor_endpoints=len(values[k]),
                    maximum=float(np.max(values[k])),weighted_quantiles={str(q):weighted_quantile(values[k],weights[k],q) for q in (.5,.95,.99,.999)},
                    training_weight_above_bound=float(np.sum(np.asarray(weights[k])[np.asarray(values[k])>bounds[k]])/sum(weights[k]))) for k in values},
                    skipped_windows=dict(skips),n_replays=len(replay_windows),n_candidate_prefix_groups=len(group_replays))
        write_json(run/"motion_bounds.json",output)
        finish_run(run,record,"completed",bounds=bounds,output=str((run/"motion_bounds.json").relative_to(ROOT)))
        print(json.dumps(dict(run_id=record["run_id"],bounds=bounds)),flush=True)
    except Exception as exc:finish_run(run,record,"failed",error=str(exc));raise

if __name__=="__main__":main()
