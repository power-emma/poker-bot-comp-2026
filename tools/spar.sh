#!/bin/bash
# Fast testing: run a sparring/ script (run.py or tune.py) inside ONE sandbox
# container with every core and no CPU limit. Matches run in-process, so
# there's no per-match container start-up. We're far under the 2s decision
# limit, so full-speed cores don't change results; only the real tournament
# needs the 0.5-core check (play.py --docker, or the validator).
#
#   tools/spar.sh run.py my_bots/tag --seeds 20                 # all sparring bots
#   tools/spar.sh run.py my_bots/tag --opp nit,shover --seeds 50
#   tools/spar.sh tune.py my_bots/tag --variant base= --variant "open60:BUTTON_OPEN_TOP=0.6"
#
# --workers defaults to the number of cores. Output files (--json) land in the repo.
cd "$(dirname "$0")/.." || exit 1
script=$1; shift
workers=()
[[ " $* " == *" --workers "* ]] || workers=(--workers "$(nproc)")
exec docker run --rm --network=none -u "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD":/kit -w /kit --entrypoint python3 fullhouse-sandbox:latest \
  "sparring/$script" "$@" "${workers[@]}"
