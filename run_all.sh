#!/bin/bash
set -e
python validation.py
python run_demo.py
python experiments.py
python experiments2.py
python calibration.py
python calibration_checks.py
python lp_placement.py
echo "All results rebuilt in results/"