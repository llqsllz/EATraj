"""Build numeric NPY cache once; never pickle targets or model objects."""
from __future__ import annotations
import argparse
import json
import sys
import traceback
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from trajectory_prediction.artifacts import start_run,finish_run,write_json,sha256_file
from trajectory_prediction.data import SceneDataset


def cache_split(root, split, *, index_name="scene_index.jsonl", cache_name=None):
    """Cache one source split without modifying either index or existing caches."""
    cache_name=cache_name or split
    if not cache_name or Path(cache_name).name!=cache_name or cache_name in (".",".."):
        raise ValueError("cache name must be a single directory name")
    if index_name!="scene_index.jsonl" and cache_name==split:
        raise ValueError("A nondefault source index requires a separate cache name")
    dest=root/"feature_cache"/cache_name
    if dest.exists(): raise ValueError(f"Refusing to overwrite feature cache {dest}")
    ds=SceneDataset(root,split,index_name=index_name)
    if not len(ds): raise ValueError(f"No {split} rows in {index_name}")
    first=ds[0];t=first["history_features"].shape[1];h=first["target"].shape[1]
    na=sum(len(r["actor_indices"]) for r in ds.rows)
    ne=sum(len(r["actor_indices"])**2 for r in ds.rows)
    config={"dataset_config_sha256":sha256_file(root/"dataset_config.json"),
            "scene_index_sha256":sha256_file(root/index_name),"index_name":index_name,
            "data":str(root),"cache_split":cache_name,"future_not_in_features":True}
    shapes={"history_features":(na,t,13),"history_edges":(ne,3,13),"target":(na,h,3),
            "future_valid":(na,h),"p0_shared":(na,3),"v0_local":(na,3),"R_local_to_shared":(na,3,3)}
    dest.mkdir(parents=True)
    arrays={k:np.lib.format.open_memmap(dest/(k+".npy"),mode="w+",dtype=bool if k=="future_valid" else np.float32,shape=s) for k,s in shapes.items()}
    ao=eo=0
    with (dest/"rows.jsonl").open("w",encoding="utf-8") as f:
        for i in range(len(ds)):
            s=first if i==0 else ds[i];n=len(s["actor_mask"])
            for k in ("history_features","target","future_valid"):
                arrays[k][ao:ao+n]=s[k]
            for k in ("p0_shared","v0_local","R_local_to_shared"):
                arrays[k][ao:ao+n]=s["history_context"][k]
            arrays["history_edges"][eo:eo+n*n]=s["history_edges"].reshape(n*n,3,13)
            row=dict(s["metadata"],actor_offset=ao,actor_count=n,edge_offset=eo)
            f.write(json.dumps(row)+"\n");ao+=n;eo+=n*n
            if (i+1)%1000==0: print(f"cached {cache_name} {i+1}/{len(ds)}",flush=True)
    for a in arrays.values(): a.flush()
    manifest=dict(config,status="complete",split=split,scenes=len(ds),actors=na,pairs=ne,
                  sample_rate_hz=ds.config["sample_rate_hz"],arrays={k:{"shape":list(s),"sha256":sha256_file(dest/(k+".npy"))} for k,s in shapes.items()})
    write_json(dest/"manifest.json",manifest)
    return {"split":split,"cache_split":cache_name,"index_name":index_name,"scenes":len(ds),"actors":na,"pairs":ne}


def main():
    p=argparse.ArgumentParser();p.add_argument("--data",default="data/trajectory_joint_v3/dev_20260921")
    p.add_argument("--index-name",default="scene_index.jsonl")
    # test_dev is intentionally supported for exploratory evaluation views.
    # Confirmatory test generation remains gated elsewhere and is not implied by
    # building this cache.
    p.add_argument("--splits",nargs="+",choices=["train","validation","test_dev"],default=["train","validation"])
    p.add_argument("--cache-split",default=None,
                   help="separate cache directory; requires exactly one source split")
    args=p.parse_args();root=ROOT/args.data
    if args.cache_split and len(args.splits)!=1:
        p.error("--cache-split requires exactly one --splits value")
    config={"data":args.data,"dataset_config_sha256":sha256_file(root/"dataset_config.json"),
            "scene_index_sha256":sha256_file(root/args.index_name),"index_name":args.index_name,
            "cache_split":args.cache_split,"splits":args.splits,"future_not_in_features":True}
    run_dir,record=start_run("trajectory-feature-cache",config,[Path(__file__)])
    summaries=[]
    try:
        for split in args.splits:
            summaries.append(cache_split(root,split,index_name=args.index_name,cache_name=args.cache_split))
        finish_run(run_dir,record,"completed",summaries=summaries)
        print(json.dumps({"run_dir":str(run_dir),"summaries":summaries},ensure_ascii=True,indent=2))
    except Exception:
        finish_run(run_dir,record,"failed",error=traceback.format_exc());raise


if __name__=="__main__":main()
