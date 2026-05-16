#!/usr/bin/env python3
"""
Standalone evaluation script that can help calculate various metrics based on the results (.csv file) from standalone_predictor.py.

This algorithm behind this script is provided by the `pm_rank` module.
It reads event data from CSV (from standalone_predictor.py) and runs evaluation.

Usage:
    # Install pm_rank (e.g. via pip), newest version by Sept 6, 2025
    pip install pm-rank>=0.2.25 

    # Get Brier score from an input csv results file
    python standalone_evaluator.py
        --input_csv subset_data_100.csv
        --output_csv predictions.csv
        --metric brier

    # Get average return from an input csv results file, with verbose output and save results
    python standalone_evaluator.py
        --input_csv subset_data_100.csv
        --output_csv predictions.csv
        --metric average_return
        --verbose
        --log_csv log.csv
"""

from pm_rank.model.average_return import AverageReturn, AverageReturnConfig
from pm_rank.model.scoring_rule import BrierScoringRule
from pm_rank.data.loaders import ProphetArenaChallengeLoader, ChallengeLoader
from ast import literal_eval

import argparse
import pandas as pd
import logging

def parse_output_csv_to_compatible_format(metadata_df: str, output_df: str) -> ChallengeLoader:
    results = []
    # iterate over each row of the output_df
    for i, row in output_df.iterrows():
        # get the index for the metadata_df
        event_ticker = row['event_ticker']
        metadata_row = metadata_df[metadata_df['event_ticker'] == event_ticker].iloc[0]

        prediction = literal_eval(row['prediction'])
        # skip if `prediction` is empty
        if not prediction:
            continue
        
        results.append({
            'prediction_id': i,
            'submission_id': i,  # for this script, we just assume that `submission_id = prediction_id`
            'prediction': prediction,
            'predictor_name': row['model'],
            'event_ticker': row['event_ticker'],
            'event_title': row['title'],
            'markets': metadata_row['markets'],
            'market_info': metadata_row['market_info'],
            'market_outcome': metadata_row['market_outcome'],
            'category': metadata_row['category'],
            'close_time': metadata_row['close_time']
        })

    results = pd.DataFrame(results)
    return ProphetArenaChallengeLoader(results, use_bid_for_odds=False).load_challenge()

def main():
    parser = argparse.ArgumentParser(
        description='Standalone evaluator')
    parser.add_argument('--input_csv', type=str, required=True,
        help='Input CSV file')
    parser.add_argument('--output_csv', type=str, required=True,
        help='Prediction CSV file')
    # has to be one of the following: brier, average_return
    parser.add_argument('--metric', type=str, required=True,
        help='Metric to use (brier or average_return)')
    parser.add_argument('--log_csv', type=str, default=None,
        help='The CSV file to log specific per-event info to')
    parser.add_argument('--verbose', action='store_true',
        help='Verbose output')
    args = parser.parse_args()

    if not args.verbose:
        logging.getLogger("pm_rank.data.loaders.ProphetArenaChallengeLoader").disabled = True

    if args.metric not in ["brier", "average_return"]:
        raise ValueError("Metric must be one of: brier, average_return")

    include_per_problem_info = args.log_csv is not None

    metadata_df = pd.read_csv(args.input_csv)
    output_df = pd.read_csv(args.output_csv)

    prophetarena_challenge = parse_output_csv_to_compatible_format(metadata_df, output_df)
    prophet_problems = prophetarena_challenge.get_problems()
    print(f"Loaded {len(prophet_problems)} problems")

    if args.metric == "brier":
        brier_scoring_rule = BrierScoringRule()
        result_tuple = brier_scoring_rule.fit(
            prophet_problems, include_scores=True, include_per_problem_info=include_per_problem_info)
    elif args.metric == "average_return":
        # Recommend to keep these as fixed. 
        # If you want to change them, check out the documentation of the `AverageReturn` class.
        # https://ai-prophet.github.io/pm_ranking/autoapi/src/pm_rank/model/index.html#src.pm_rank.model.AverageReturnConfig
        average_return_config = AverageReturnConfig(
            num_money_per_round=1, 
            use_approximate=True, 
            risk_aversion=0.0, 
            use_binary_reduction=True)
        result_tuple = AverageReturn(config=average_return_config).fit(
            prophet_problems, include_scores=True, include_per_problem_info=include_per_problem_info)

    result_score = {k: float(v) for k, v in result_tuple[0].items()}
    print(f"Resulting score for metric {args.metric}: {result_score}")

    if include_per_problem_info:
        result_per_problem_info = result_tuple[2]
        print(f"Logging per-event scoring info for metric {args.metric} to {args.log_csv}")

        result_df = pd.DataFrame(result_per_problem_info)
        result_df.to_csv(args.log_csv, index=False)


if __name__ == "__main__":
    main()