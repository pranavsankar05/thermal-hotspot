#!/bin/bash
set -e
python validation.py
python run_demo.py
python experiments.py
python experiments2.py
python calibration.py
python calibration2.py
echo "All results rebuilt in results/"