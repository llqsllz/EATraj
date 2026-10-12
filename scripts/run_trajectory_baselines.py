"""Tracked, common-cohort CV/CA development evaluation over several horizons."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from trajectory_prediction.artifacts import start_run,finish_run,write_json,sha256_file
from trajectory_prediction.data import SceneDataset
from trajectory_prediction.packed_data import PackedSceneDataset
from trajectory_prediction.geometry import trailing_velocity
from trajectory_prediction.metrics import score_scene,aggregate_scene_records


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",default="data/trajectory_joint_v3/dev_20260921")
    parser.add_argument("--split",default="validation",choices=["train","validation","test_dev"])
    parser.add_argument("--index-name",default="scene_index.jsonl")
    parser.add_argument("--cache-split",default=None,
                        help="feature_cache directory name; source split is unchanged")
    parser.add_argument("--limit",type=int,default=0)
    parser.add_argument("--loader",choices=["auto","scene","packed"],default="auto",
                        help="auto selects PackedSceneDataset when the split has a feature cache")
    parser.add_argument("--skip-array-hash",action="store_true",
                        help="skip NPY byte hashes after schema/provenance checks (development only)")
    args=parser.parse_args()
    dataset_root = ROOT / args.data
    cache_exists = (dataset_root / "feature_cache" / (args.cache_split or args.split) / "manifest.json").exists()
    use_packed = args.loader == "packed" or (args.loader == "auto" and cache_exists and
                                                (args.cache_split is not None or
                                                 not (dataset_root / "tracks").exists()))
    config = json.loads((dataset_root / "dataset_config.json").read_text(encoding="utf-8"))
    horizons_from_config = "evaluation_horizons_seconds" in config
    horizons = config.get("evaluation_horizons_seconds", [5.0, 10.0, 20.0])
    if use_packed:
        # Public raw adapters expose only the packed contract. Load the full
        # stored horizon once, then score exact prefixes below.
        data=PackedSceneDataset(dataset_root,args.split,max(horizons),
                                verify_array_hashes=not args.skip_array_hash,
                                index_name=args.index_name,cache_split=args.cache_split)
        loader_name="packed"
    else:
        data=SceneDataset(dataset_root,args.split,index_name=args.index_name)
        loader_name="scene"
    split_manifest = dataset_root / "split_manifest.csv"
    conf={"data":args.data,"split":args.split,"limit":args.limit,"models":["CV","CA"],
          "index_name":args.index_name,"cache_split":args.cache_split or args.split,
          "loader":loader_name,"array_hashes_verified":getattr(data,"integrity",{}).get("array_hashes_verified"),
          "dataset_config_sha256":sha256_file(dataset_root/"dataset_config.json"),
          "split_manifest_sha256":sha256_file(split_manifest) if split_manifest.exists() else None,
          "scene_index_sha256":sha256_file(dataset_root/args.index_name),
          "horizons_seconds":horizons,
          "horizons_source":"dataset_config" if horizons_from_config else "compatibility_default_5_10_20_s",
          "target_time_alignment":(
              "nearest-unused raw observations within configured tolerance; "
              "target_time_offsets.npy is an audit sidecar, not an input to the common nominal-grid scorer"
              if "target_match_tolerance_s" in config else "canonical grid targets"),
          "CA_acceleration_estimator":"trailing1s_LS_slope_of_history_only_velocity_estimates",
          "source_units_and_independence":"assumed_units_and_unverified_prefix_clusters_exploratory_only"}
    run_dir,record=start_run("cv-ca-development",conf,[Path(__file__)])
    records={}; start=time.perf_counter()
    max_actors=max(row["actor_count"] for row in data.rows)
    prediction_store={"CV":[],"CA":[]}; prediction_metadata=[]; prediction_counts=[]
    try:
        with (run_dir/"scene_scores.jsonl").open("w",encoding="utf-8") as f:
            for i in range(min(len(data),args.limit or len(data))):
                sample=data[i]; meta=sample["metadata"]
                if loader_name == "packed":
                    # Public packed rows retain the canonical 13-D features,
                    # not the live-only derived history fields. Recover the
                    # same physical values using the stored local frame.
                    features=sample["history_features"]
                    context=sample["history_context"]
                    p0=context["p0_shared"]
                    rotation=context["R_local_to_shared"]
                    v_local=features[...,3:6]*100.0
                    v=np.einsum("ntd,nde->nte",v_local,np.swapaxes(rotation,1,2))
                    history_velocity_valid=features[...,7] > 0.5
                    history_times=features[0,:,12].astype(np.float64)*10.0
                else:
                    p_hist=sample["history_position_shared"]
                    p0=p_hist[:,-1]
                    v=sample["history_velocity_shared"]
                    history_velocity_valid=sample["history_velocity_valid"]
                    history_times=sample["history_times"]
                v0=v[:,-1]
                acc=[]
                for vi,mask in zip(v,history_velocity_valid):
                    av,am=trailing_velocity(history_times[mask],vi[mask],window_s=1,min_samples=3,min_span_s=.4)
                    acc.append(av[-1] if am[-1] else np.zeros(3))
                acc=np.asarray(acc)
                prediction_metadata.append(meta)
                prediction_counts.append(v0.shape[0])
                for model in ("CV","CA"):
                    times=sample["forecast_times"]
                    pred=p0[:,None,:]+v0[:,None,:]*times[None,:,None]
                    if model=="CA": pred=pred+.5*acc[:,None,:]*times[None,:,None]**2
                    padded=np.zeros((max_actors,pred.shape[1],3),dtype=np.float32)
                    padded[:pred.shape[0]]=pred
                    prediction_store[model].append(padded)
                    for h in conf["horizons_seconds"]:
                        count=round(h*config["sample_rate_hz"])
                        scored=score_scene(pred[None,:,:count],sample["target"][:,:count],sample["future_valid"][:,:count])
                        scored.update(replay_id=meta["replay_id"],family_id=meta["family_id"],
                                      sample_id=meta.get("sample_id",meta.get("scene_id")),
                                      model=model,horizon_seconds=h,
                                      scenario=meta.get("scenario","trajair_public"))
                        records.setdefault((model,h),[]).append(scored)
                        f.write(json.dumps(scored,allow_nan=False)+"\n")
                if (i+1)%500==0: print(f"scored {i+1}/{len(data)} windows in {time.perf_counter()-start:.1f}s",flush=True)
        summaries=[]
        for (model,h),rows in records.items():
            summary=aggregate_scene_records(rows)
            summary.update(model=model,horizon_seconds=h,evidence_status="exploratory",independence_verified=False)
            summaries.append(summary)
        write_json(run_dir/"summary.json",summaries)
        np.savez_compressed(run_dir/"baseline_predictions.npz",
                            cv=np.asarray(prediction_store["CV"]),
                            ca=np.asarray(prediction_store["CA"]),
                            actor_counts=np.asarray(prediction_counts),
                            sample_ids=np.asarray([m.get("sample_id",m.get("scene_id",m.get("endpoint_index"))) for m in prediction_metadata]))
        write_json(run_dir/"prediction_index.json",prediction_metadata)
        table=[{k:s[k] for k in ("model","horizon_seconds","scene_minADE_m","scene_minFDE_m","top_score_ADE_m","top_score_FDE_m")} | s["counts"] for s in summaries]
        with (run_dir/"baseline_table.csv").open("w",newline="",encoding="utf-8-sig") as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
        finish_run(run_dir,record,"completed",elapsed_seconds=time.perf_counter()-start,summary_table=table,
                   exact_scene_index_sha256=sha256_file(dataset_root/args.index_name))
        print(json.dumps({"run_dir":str(run_dir),"table":table},ensure_ascii=True,indent=2))
    except Exception:
        finish_run(run_dir,record,"failed",error=traceback.format_exc());raise


if __name__=="__main__": main()
