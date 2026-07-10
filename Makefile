PY=python3

all: tune e1 e4 e5 e6 e7-mock phase2 phase3 test

tune:
	$(PY) experiments/tune.py

e1:
	$(PY) experiments/e1_scaling.py

e4:
	$(PY) experiments/e4_memory.py

e5:
	$(PY) experiments/e5_delay.py

e6:
	for c in het-M2 hom-M2 het-M1; do for s in 0 1 2 3 4; do $(PY) experiments/e6_emergence.py --cond $$c --seed $$s; done; done
	$(PY) experiments/e6_emergence.py --cond assemble

e7-mock:
	$(PY) experiments/e7_llm.py --backend mock

e7-real:
	$(PY) experiments/e7_llm.py --backend anthropic --episodes 8

phase2:
	for d in 0 1 3 5 7; do $(PY) experiments/phase2.py map --d $$d; done
	$(PY) experiments/phase2.py equal-budget
	$(PY) experiments/phase2.py map-adaptive
	$(PY) experiments/phase2.py robustness
	$(PY) experiments/phase2.py nonstationary
	$(PY) experiments/phase2.py delay
	$(PY) experiments/phase2.py nonlinear
	$(PY) experiments/phase2.py figures

phase3:
	$(PY) experiments/phase3.py all

# Fast smoke: rebuild figures from SAVED data + validate headline claims.
# No heavy simulation; suitable for CI on every push.
smoke:
	$(PY) experiments/phase3.py concept
	$(PY) experiments/phase2.py figures
	$(PY) experiments/phase3.py zones
	$(PY) experiments/check_results.py

check:
	$(PY) experiments/check_results.py

test:
	$(PY) -m pytest tests/ -q

.PHONY: all tune e1 e4 e5 e6 e7-mock e7-real phase2 phase3 smoke check test
