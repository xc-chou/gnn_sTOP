# GNN-assisted solver for Stochastic TOP

Reproduction code and data for *A graph neural network-assisted solver for the two-stage stochastic team orienteering problem* by Xiaochen Chou, Enza Messina, and David Pisinger.

- `instances/` contains the benchmark CSVs supplied by David Pisinger. Filenames give the family, distribution, and scenario count, for example `STD_uniform_16scen.csv`. [Benchmark paper](https://doi.org/10.1016/j.cor.2025.107365)
- `code/` contains code developed. Choose the task with `--experiment`. `experiments.py` runs the tests, `solver.py` contains the GNN and routing algorithms, and `paper_results.py` builds the tables and figure.
- `results/` contains one archived CSV per test. `oos_results.csv` contains every STD, VAR, and LRG target in one file. Filter by `family` and `distribution`. `benchmark_scenarios` is the size of the source benchmark and `scenarios` is the number evaluated per target.

## Generate the paper tables

From this folder, run with Python 3.12.9. This step uses the archived CSVs and requires only the standard library. Tables 1–6 appear under `results/generated/paper_tables/`.

```sh
python code/reproduce.py --experiment report
```

## Run the experiments

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy==2.1.3 torch==2.5.1
```

Quick check: run one target and one evaluation scenario:

```sh
python code/reproduce.py --experiment oos --family STD --distribution uniform --instance 1 --scenario-count 1 --output-dir results/test
```

For the complete experiments, run these commands in order:

```sh
python code/reproduce.py --experiment characteristics
python code/reproduce.py --experiment screen
python code/reproduce.py --experiment validation
python code/reproduce.py --experiment oos --workers 6
python code/reproduce.py --experiment recourse-budget --workers 6
python code/reproduce.py --experiment report --input-dir results/generated
```

The OOS command runs all three families, or use `--family STD`, `VAR`, or `LRG` to run them separately into the same CSV. The budget experiment uses the fixed customer sets from the STD and VAR OOS runs. New results go to `results/generated/`. Full routing for all instances can take several hours. Use a different `--output-dir` to repeat an experiment. For an interrupted LRG run, use `--experiment oos --family LRG --resume` with the same target and output options.

The paper uses scenarios 1–16 for construction and 129–512 for STD/VAR evaluation. LRG uses P128 training labels and scenarios 17–128 for evaluation. The timing table combines results from different hardware, so fresh running times may vary.

To draw the recourse figure, install `matplotlib==3.10.0` and run `python code/reproduce.py --experiment plot`. Add `--input-dir results/generated` to use fresh results.
