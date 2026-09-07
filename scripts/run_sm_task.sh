#!/bin/bash
cd /root/autodl-tmp/exp/scripts
./sm_profile > /root/autodl-tmp/exp/results/sm_profile_raw.txt 2>/dev/null
/root/miniconda3/bin/python3 /root/autodl-tmp/exp/scripts/analyze_sm_profile.py /root/autodl-tmp/exp/results/sm_profile_raw.txt
