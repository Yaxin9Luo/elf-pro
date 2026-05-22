"""HOPE rendezvous helper: turn AFO_ENV_CLUSTER_SPEC into torch.distributed.run flags.

This is the project-agnostic helper used across HOPE submissions; copied verbatim
from
``/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/yaxinluo/projects/laionface-tokenzier/mm-pretrain-hfds/pretrain/hope_run_torch_distribute.py``
so the elf-pro submission is self-contained.

Usage (inside a HOPE worker.script):

    eval "HOPE_TRACKING_RANK=0 python3 -m torch.distributed.run \
        $(python3 hope_run_torch_distribute.py) src/train.py --config <yml>"

The script prints the rendezvous flags (``--nnodes ... --master_port ...``) on
stdout so they can be substituted into the torchrun invocation. Cluster topology
is read from the ``AFO_ENV_CLUSTER_SPEC`` env var that HOPE injects per-worker.
"""
import os
import json
import sys

cluster_spec = json.loads(os.environ["AFO_ENV_CLUSTER_SPEC"])
role = cluster_spec["role"]
assert role == "worker", "{} vs worker".format(role)
node_rank = cluster_spec["index"]
nnodes = len(cluster_spec[role])
nproc_per_node = os.popen("nvidia-smi --list-gpus | wc -l").read().strip()
master = cluster_spec[role][0]
print(cluster_spec, file=sys.stderr)
print(role, file=sys.stderr)
print(master, file=sys.stderr)
master_addr, master_ports = master.split(":")
master_ports = master_ports.split(",")
print(
   "--nnodes={} "
   "--nproc-per-node={} "
   "--node_rank={} "
   "--master_addr={} "
   "--master_port={}".format(
        nnodes, int(nproc_per_node), node_rank, master_addr, master_ports[0]
   )
)
