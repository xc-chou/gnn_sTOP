#!/usr/bin/env python3
"""Run the journal experiments or rebuild the paper tables."""

import argparse
from pathlib import Path

from experiments import run_experiment
from paper_results import run_author_times, run_characteristics, run_plot, run_report
from solver import DISTRIBUTIONS, FAMILIES, TARGETS

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORT_TASKS = {
    "characteristics": run_characteristics,
    "author-times": run_author_times,
    "report": run_report,
    "plot": run_plot,
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="report",
                        choices=("screen", "validation", "oos", "recourse-budget", *REPORT_TASKS))
    parser.add_argument("--family", type=str.upper, choices=FAMILIES)
    parser.add_argument("--distribution", choices=DISTRIBUTIONS)
    parser.add_argument("--instance", type=int, choices=TARGETS)
    parser.add_argument("--scenario-count", type=int, help="Limit evaluation scenarios for a quick check.")
    parser.add_argument("--seeds", type=int, choices=(1, 2, 3), default=3,
                        help="Number of LRG training seeds per target.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel routing processes.")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted LRG run.")
    parser.add_argument("--instances-dir", type=Path, default=ROOT / "instances")
    parser.add_argument("--input-dir", type=Path, help="Read results from this folder.")
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "generated")
    parser.add_argument("--log-dir", type=Path, help="Author logs for the author-times experiment.")
    parser.add_argument("--author-times", type=Path, help="Use this timing CSV when reporting.")
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1.")
    if args.resume and (args.experiment != "oos" or args.family != "LRG"):
        parser.error("--resume is used with --experiment oos --family LRG.")
    if args.experiment == "recourse-budget" and args.family == "LRG":
        parser.error("The recourse-budget experiment covers STD and VAR.")
    if args.experiment == "author-times" and args.log_dir is None:
        parser.error("--experiment author-times requires --log-dir.")
    if args.scenario_count is not None:
        if args.experiment not in ("oos", "recourse-budget"):
            parser.error("--scenario-count applies to OOS or recourse-budget.")
        limit = 112 if args.experiment == "oos" and args.family in (None, "LRG") else 384
        if not 1 <= args.scenario_count <= limit:
            parser.error(f"--scenario-count must be between 1 and {limit}.")
    if args.output_dir.resolve() == RESULTS:
        parser.error("Choose an output subfolder to keep the archived results intact.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        REPORT_TASKS.get(args.experiment, run_experiment)(args)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")

if __name__ == "__main__":
    main()
