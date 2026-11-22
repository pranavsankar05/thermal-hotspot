# thermal-hotspot

# Thermal Hotspot Prediction in a Server Room (2D Heat Equation)

MFC end-semester project, Theme B (PDEs).

## Setup
python -m venv .venv
(activate it)
pip install -r requirements.txt

## Run
python run_demo.py        # steady-state heatmap -> results/demo_heatmap.png
python validation.py      # validation checks V1, V4, V6
python experiments.py            # all experiments -> results/*.csv, results/*.png
python experiments.py exp3a      # a single experiment (exp1, exp1b, exp2, exp3a, exp3b, exp4)

## Status
- [x] Phase 1: solver core + validation V1, V4, V6
- [x] Phase 2: sparse assembly, own CG, V3, V5
- [x] Phase 3: experiments 1-5, BE/CN
- [ ] Phase 4: Streamlit app