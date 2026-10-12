"""Memory-mapped, numeric-only feature cache for repeatable GPU experiments."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .artifacts import sha256_file
from .data import resolve_time_grid


class PackedSceneDataset:
    def __init__(self,root: Path | str,split: str,horizon_seconds: float, *, verify_array_hashes: bool = True,
                 index_name: str = "scene_index.jsonl", cache_split: str | None = None):
        """Verify provenance/schema once, then expose read-only memory maps.

        Array content hashes are checked by default. Explicitly opting out only
        skips the full NPY byte scan, never protocol/index/metadata validation.
        The caller can record self.integrity in its experiment metadata.
        """
        dataset_root=Path(root)
        self.root=dataset_root/"feature_cache"/(cache_split or split)
        self.manifest=json.loads((self.root/"manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("status")!="complete": raise ValueError("Incomplete feature cache")
        if self.manifest.get("split")!=split: raise ValueError("Feature cache split mismatch")
        if self.manifest.get("index_name","scene_index.jsonl")!=index_name:
            raise ValueError("Feature cache source index mismatch")
        if self.manifest.get("cache_split",split)!=(cache_split or split):
            raise ValueError("Feature cache directory identity mismatch")
        for filename,key in (("dataset_config.json","dataset_config_sha256"),(index_name,"scene_index_sha256")):
            if self.manifest.get(key)!=sha256_file(dataset_root/filename):
                raise ValueError(f"Feature cache {key} mismatch; rebuild a versioned cache")
        config=json.loads((dataset_root/"dataset_config.json").read_text(encoding="utf-8"))
        grid=resolve_time_grid(config,horizon_seconds)
        if self.manifest.get("sample_rate_hz")!=grid["sample_rate_hz"]:
            raise ValueError("Feature cache sample_rate_hz mismatch")
        self.rows=[json.loads(x) for x in (self.root/"rows.jsonl").read_text(encoding="utf-8").splitlines() if x]
        source_rows=[json.loads(x) for x in (dataset_root/index_name).read_text(encoding="utf-8").splitlines() if x]
        source_rows=[row for row in source_rows if row["split"]==split]
        if any(row.get("split") != split for row in self.rows):
            raise ValueError("Feature cache rows contain records from another split")
        if len(self.rows)!=len(source_rows) or self.manifest.get("scenes")!=len(self.rows):
            raise ValueError("Feature cache scene count mismatch")
        ao=eo=0
        cache_fields={"actor_offset","actor_count","edge_offset"}
        for row,source in zip(self.rows,source_rows):
            if {k:v for k,v in row.items() if k not in cache_fields}!=source:
                raise ValueError("Feature cache row metadata differs from scene_index")
            n=len(source["actor_indices"])
            if row.get("actor_offset")!=ao or row.get("edge_offset")!=eo or row.get("actor_count")!=n:
                raise ValueError("Feature cache actor/edge offsets mismatch")
            ao+=n;eo+=n*n
        if self.manifest.get("actors")!=ao or self.manifest.get("pairs")!=eo:
            raise ValueError("Feature cache actor/pair totals mismatch")
        t,h=grid["history_steps"],grid["max_future_steps"]
        expected_shapes={"history_features":(ao,t,13),"history_edges":(eo,3,13),"target":(ao,h,3),
                         "future_valid":(ao,h),"p0_shared":(ao,3),"v0_local":(ao,3),"R_local_to_shared":(ao,3,3)}
        if set(self.manifest.get("arrays",{}))!=set(expected_shapes):
            raise ValueError("Feature cache array schema mismatch")
        self.arrays={}
        for name,shape in expected_shapes.items():
            path=self.root/(name+".npy")
            entry=self.manifest["arrays"][name]
            if entry.get("shape")!=list(shape): raise ValueError(f"Feature cache {name} declared shape mismatch")
            if verify_array_hashes and entry.get("sha256")!=sha256_file(path):
                raise ValueError(f"Feature cache {name} SHA256 mismatch")
            array=np.load(path,mmap_mode="r",allow_pickle=False)
            dtype=np.dtype(bool if name=="future_valid" else np.float32)
            if array.shape!=shape or array.dtype!=dtype:
                raise ValueError(f"Feature cache {name} stored shape/dtype mismatch")
            self.arrays[name]=array
        self.integrity={"protocol_and_index_hashes_verified":True,"row_metadata_verified":True,
                        "array_schema_verified":True,"array_hashes_verified":bool(verify_array_hashes),
                        "validation_stage":"initialization_only"}
        self.h=grid["future_steps"]
        self.forecast_times=np.arange(1,self.h+1,dtype=np.float32)/grid["sample_rate_hz"]

    def __len__(self): return len(self.rows)

    def __getitem__(self,i):
        row=self.rows[i]; lo,hi=row["actor_offset"],row["actor_offset"]+row["actor_count"]
        ep=row["edge_offset"]; n=row["actor_count"]
        return {"history_features":self.arrays["history_features"][lo:hi],"actor_mask":np.ones(n,bool),
                "history_edges":self.arrays["history_edges"][ep:ep+n*n].reshape(n,n,3,13),
                "history_context":{key:self.arrays[key][lo:hi] for key in ("p0_shared","v0_local","R_local_to_shared")},
                "target":self.arrays["target"][lo:hi,:self.h],"future_valid":self.arrays["future_valid"][lo:hi,:self.h],
                "forecast_times":self.forecast_times,"metadata":row}


def collate_scenes(samples):
    import torch
    b=len(samples); n=max(len(s["actor_mask"]) for s in samples)
    t=samples[0]["history_features"].shape[1]; h=len(samples[0]["forecast_times"])
    shape={"history_features":(b,n,t,13),"history_edges":(b,n,n,3,13),
           "target":(b,n,h,3),"future_valid":(b,n,h),"actor_mask":(b,n),
           "p0_shared":(b,n,3),"v0_local":(b,n,3),"R_local_to_shared":(b,n,3,3)}
    out={k:np.zeros(s,dtype=bool if k in ("future_valid","actor_mask") else np.float32) for k,s in shape.items()}
    for i,s in enumerate(samples):
        ni=len(s["actor_mask"])
        for key in ("history_features","target","future_valid","actor_mask"):
            out[key][i,:ni]=s[key]
        out["history_edges"][i,:ni,:ni]=s["history_edges"]
        for key in ("p0_shared","v0_local","R_local_to_shared"):
            out[key][i,:ni]=s["history_context"][key]
    out={k:torch.from_numpy(v) for k,v in out.items()}
    out["history_context"]={k:out.pop(k) for k in ("p0_shared","v0_local","R_local_to_shared")}
    out["forecast_times"]=torch.from_numpy(samples[0]["forecast_times"].copy())
    out["metadata"]=[s["metadata"] for s in samples]
    return out
