## using configs

Experiments using `prohmr_depth` or `prohmr_surfnormals` models requires `prohmr.yaml`.\
Experiments using `prohmr_fusion`, `prohmr_fusion_attention`, or `prohmr_fusion_flow` require `prohmr_fusion.yaml`. In this case, 2 key parameters exist, `CONFIG.MODEL.FUSION` and `CONFIG.MODEL.FLOW.MODE`. The former sets whether the fusion is done before the normalizing flow or after. The choices are `flow` for post-flow or anything else for before. The latter works only if the former is not set to `flow` and has choices `concat`, `mlp`, or `attention`. This is case sensitive and no other choices will work.

We apologize for these complications and look to make the scripts more robust in the future.