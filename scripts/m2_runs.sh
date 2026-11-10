#!/bin/bash
# Every Milestone 2 run, in order. Each step skips finished work, so after a disconnect just run it again.
# Results land on Drive; U3's crops live on the runtime's disk and are rebuilt in a new session.
set -e
python -m ror.data                     # GenImage images on this runtime (skips files already here)

# B+U3 first (about 40 min): noise branch on frozen CLIP, both fusions, 3 seeds
python -m ror.patches
for fusion in concat gate; do for seed in 0 1 2; do python -m ror.u3 train --fusion "$fusion" --seed "$seed"; done; done
python -m ror.u3 report

# B+U1+U2 (about 2 h): rank 8 (from the sweep), 3 seeds, clean and damaged tests
for seed in 0 1 2; do python -m ror.u1 train --rank 8 --seed "$seed" --u2 --damaged; done

# Full model: U3 on top of each seed's trained B+U1+U2
python -m ror.patches full
for fusion in concat gate; do for seed in 0 1 2; do python -m ror.u3 train --fusion "$fusion" --seed "$seed" --full; done; done

python -m ror.u3 report
