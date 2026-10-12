"""Build exploratory windows; quarantined source partitions remain unbuilt."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from trajectory_prediction.artifacts import start_run,finish_run,write_json,sha256_file
from trajectory_prediction.geometry import geodetic_to_enu
from trajectory_prediction.data import history_actors


def clean_track(df,origin,quality):
    ids=sorted(df.aircraft_id.astype(str).unique())
    fmin,fmax=int(df.frame_idx.min()),int(df.frame_idx.max())
    frames=np.arange(fmin,fmax+1,dtype=np.int64)
    pos=np.zeros((len(frames),len(ids),3),dtype=np.float64)
    valid=np.zeros((len(frames),len(ids)),dtype=bool)
    segment=np.full((len(frames),len(ids)),-1,dtype=np.int32)
    counts={"raw_invalid":0,"quarantined_warmup":0,"jump_resets":0,"gap_resets":0}
    for j,aid in enumerate(ids):
        a=df[df.aircraft_id.astype(str)==aid].sort_values("frame_idx")
        fi=a.frame_idx.to_numpy(dtype=int); ts=fi/10.
        llh=a[["longitude_raw","latitude_raw","altitude_raw"]].to_numpy(dtype=float)
        good=np.isfinite(llh).all(axis=1)&np.any(llh!=0,axis=1)&(np.abs(llh[:,0])<=180)&(np.abs(llh[:,1])<=90)
        xyz=np.zeros_like(llh); xyz[good]=geodetic_to_enu(llh[good],origin)
        sg=-1; start=None; prev=None
        for k in range(len(a)):
            idx=fi[k]-fmin
            if not good[k]:
                counts["raw_invalid"]+=1; start=None; prev=None
                continue
            reset=prev is None
            if prev is not None:
                dt=ts[k]-ts[prev]
                if dt<=0 or dt>quality["max_source_gap_seconds"]:
                    reset=True; counts["gap_resets"]+=1
                elif np.linalg.norm(xyz[k]-xyz[prev])/dt>quality["max_observed_step_speed_mps"]:
                    reset=True; counts["jump_resets"]+=1
            if reset:
                sg+=1; start=ts[k]
            prev=k
            pos[idx,j]=xyz[k]; segment[idx,j]=sg
            if ts[k]-start>=quality["causal_warmup_seconds"]-1e-8:
                valid[idx,j]=True
            else:
                counts["quarantined_warmup"]+=1
    return {"frames":frames,"position":pos,"valid":valid,"segment":segment,
        "actor_ids":np.asarray(ids),"origin_llh":np.asarray(origin)},counts


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/experiments/trajectory_development_v3.json")
    args=parser.parse_args(); cp=ROOT/args.config
    config=json.loads(cp.read_text(encoding="utf-8"))
    out=ROOT/config["output_root"]
    if out.exists():
        raise SystemExit(f"Refusing overwrite: {out}. Choose a new versioned output_root.")
    audit=ROOT/config["source_audit_run"]
    replay=pd.read_csv(audit/"replay_audit.csv")
    exposure=pd.read_csv(ROOT/config["exposure_manifest"])
    replay=replay.merge(exposure,on="replay_id",validate="one_to_one")
    # Conservative grouping of detected repeated prefixes; not certified families.
    replay["split"]=np.where(replay.old_split=="train","train","validation")
    replay.loc[replay.reserve_unverified_old_5v5_test,"split"]="reserved"
    for _,group in replay.groupby("prefix_fingerprint_candidate"):
        role="reserved" if (group.split=="reserved").any() else "validation" if (group.split=="validation").any() else "train"
        replay.loc[group.index,"split"]=role
    replay["development_group_id"]="prefix_"+replay.prefix_fingerprint_candidate.str[:20]
    config["source_audit_sha256"]=sha256_file(audit/"replay_audit.csv")
    config["exposure_manifest_sha256"]=sha256_file(ROOT/config["exposure_manifest"])
    run_dir,record=start_run("trajectory-development-build",config,[Path(__file__),cp,ROOT/config["exposure_manifest"]])
    out.mkdir(parents=True); (out/"tracks").mkdir()
    write_json(out/"dataset_config.json",config)
    replay.to_csv(out/"split_manifest.csv",index=False,encoding="utf-8-sig")
    built=[]; scene_counts={}; total_quality={}; snapshots={}
    try:
        with (out/"scene_index.jsonl").open("w",encoding="utf-8") as index_file:
            for number,(_,row) in enumerate(replay.iterrows(),1):
                if row.split=="reserved":
                    continue
                source=ROOT/"data/parsed/v3/replays"/row.replay_id/"aircraft_states.parquet"
                df=pd.read_parquet(source,columns=["frame_idx","aircraft_id","longitude_raw","latitude_raw","altitude_raw"])
                origin=np.asarray([row.origin_lon,row.origin_lat,row.origin_alt])
                track,quality=clean_track(df,origin,config["quality"])
                destination=out/"tracks"/(row.replay_id+".npz")
                np.savez_compressed(destination,**track)
                snapshots[row.replay_id]={"source_parquet_sha256":row.aircraft_parquet_sha256,"track_sha256":sha256_file(destination)}
                stride=round(config[f"{row.split}_stride_seconds"]*config["source_rate_hz"])
                hist=round(config["history_seconds"]*config["source_rate_hz"])
                step=int(config["source_rate_hz"]/config["sample_rate_hz"])
                count=0
                for end in range(hist,len(track["frames"]),stride):
                    actors=history_actors(track,end,hist,step)
                    if len(actors)<config["minimum_history_actors"]:
                        continue
                    sid=hashlib.sha256(f"{config['version']}|{row.replay_id}|{int(track['frames'][end])}".encode()).hexdigest()[:24]
                    sample={"sample_id":sid,"replay_id":row.replay_id,"family_id":row.development_group_id,
                        "family_status":"unverified_prefix_proxy","split":row.split,"scenario":row.scenario,
                        "endpoint_index":end,"endpoint_frame":int(track["frames"][end]),"actor_indices":actors.tolist()}
                    index_file.write(json.dumps(sample)+"\n"); count+=1
                built.append({"replay_id":row.replay_id,"split":row.split,"scene_count":count,"quality":quality})
                scene_counts[row.split]=scene_counts.get(row.split,0)+count
                for key,value in quality.items(): total_quality[key]=total_quality.get(key,0)+value
                if number%25==0:
                    print(f"source {number}/{len(replay)}; built={len(built)}; scenes={scene_counts}",flush=True)
        summary={"dataset_root":str(out.relative_to(ROOT)),"evidence_status":"exploratory",
            "split_replay_counts":replay.groupby("split").size().to_dict(),"split_scenario_counts":{
                s:g.groupby("scenario").size().to_dict() for s,g in replay.groupby("split")},
            "window_counts":scene_counts,"quality_counts":total_quality,
            "built_replays":len(built),"reserved_replays_not_built":int((replay.split=="reserved").sum()),
            "confirmed_independent_families":False,"final_test_performed":False,
            "required_semantic_annotations":[],"time_protocol_frozen_for_paper":False}
        write_json(out/"build_summary.json",summary); write_json(out/"source_checksums.json",snapshots)
        write_json(run_dir/"build_summary.json",summary); write_json(run_dir/"replay_build_counts.json",built)
        finish_run(run_dir,record,"completed",dataset_config_sha256=sha256_file(out/"dataset_config.json"),
            split_manifest_sha256=sha256_file(out/"split_manifest.csv"),scene_index_sha256=sha256_file(out/"scene_index.jsonl"),summary=summary)
        print(json.dumps({"run_dir":str(run_dir),"summary":summary},ensure_ascii=True,indent=2))
    except Exception:
        finish_run(run_dir,record,"failed",error=traceback.format_exc()); raise


if __name__=="__main__": main()
