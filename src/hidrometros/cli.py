"""Command line entry point: `hidrometros <command>`."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="hidrometros")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("integrity", help="verify the Kaggle download and splits.csv")
    sub.add_parser("eda", help="print exploratory statistics")
    sub.add_parser("prep", help="rectify displays, fix orientation, write digits/ and index.csv")
    sub.add_parser("prep-extra", help="extra challenge: locate digit boundaries, write digits_extra/")
    sub.add_parser("smoke", help="2-epoch run per model on 256 train / 128 val crops (real code path)")
    p_train = sub.add_parser("train", help="train one model, or all four sequentially")
    p_train.add_argument("model", choices=["smallcnn", "alexnet", "vgg16_bn", "resnet18", "all"])
    p_train.add_argument("--variant", default="base", choices=["base", "extra", "diag"])
    p_train.add_argument("--no-early-stop", action="store_true", help="diagnostic only (variant diag)")
    p_train.add_argument("--resume", action="store_true")
    p_eval = sub.add_parser("evaluate", help="test-split metrics -> results csv, reports")
    p_eval.add_argument("--variant", default="base", choices=["base", "extra"])
    p_eval.add_argument("--models", nargs="+", choices=["smallcnn", "alexnet", "vgg16_bn", "resnet18"], help="re-evaluate only these; other rows are kept")
    sub.add_parser("figures", help="draw the 4 required figures from results and histories")
    sub.add_parser("readme", help="regenerate the generated parts of README.md from results.csv")
    args = parser.parse_args(argv)

    if args.cmd == "integrity":
        from hidrometros import data

        integrity = data.check_integrity()
        print(integrity)
        raise SystemExit(0 if integrity.ok else 1)
    if args.cmd == "eda":
        from hidrometros import eda

        eda.main()
    elif args.cmd == "prep":
        from hidrometros import build

        print(json.dumps(build.run_prep(), indent=2, default=str))
    elif args.cmd == "prep-extra":
        from hidrometros import build

        print(json.dumps(build.run_prep_extra(), indent=2))
    elif args.cmd == "smoke":
        from hidrometros import smoke

        sys.exit(smoke.main())
    elif args.cmd == "train":
        from hidrometros.models import EARLY_STOP_PATIENCE, MODEL_NAMES
        from hidrometros.train import VARIANTS, run_training

        # protocol order from the spec: cheapest first, vgg16_bn last
        order = ("smallcnn", "resnet18", "alexnet", "vgg16_bn") if args.model == "all" else (args.model,)
        assert set(order) <= set(MODEL_NAMES)
        for name in order:
            if args.no_early_stop and args.variant != "diag":
                parser.error("--no-early-stop is only allowed with --variant diag")
            run_training(
                name,
                variant=VARIANTS[args.variant],
                resume=args.resume,
                patience=10**6 if args.no_early_stop else EARLY_STOP_PATIENCE,
            )
    elif args.cmd == "evaluate":
        from hidrometros.evaluate import write_results
        from hidrometros.train import VARIANTS

        from hidrometros.models import MODEL_NAMES

        names = tuple(args.models) if args.models else MODEL_NAMES
        print(write_results(VARIANTS[args.variant], names).to_string(index=False))
    elif args.cmd == "figures":
        from hidrometros import figures

        figures.main()
    elif args.cmd == "readme":
        from hidrometros import report

        report.main()


if __name__ == "__main__":
    main()
