import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset


def ndcg_at_k(predicted_track_ids, ground_truth_track_id, k):
    for rank, track_id in enumerate(predicted_track_ids[:k], start=1):
        if track_id == ground_truth_track_id:
            return 1.0 / math.log2(rank + 1)
    return 0.0


def distinct_2(responses):
    total = 0
    unique = set()
    for response in responses:
        tokens = (response or "").lower().split()
        for idx in range(len(tokens) - 1):
            unique.add((tokens[idx], tokens[idx + 1]))
            total += 1
    return len(unique) / total if total else 0.0


def load_predictions(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    keyed = {}
    for row in rows:
        key = (row["session_id"], int(row["turn_number"]))
        if key in keyed:
            raise ValueError(f"Duplicate prediction for {key}")
        predicted = row.get("predicted_track_ids", [])
        if len(predicted) != len(set(predicted)):
            raise ValueError(f"Duplicate track IDs in prediction for {key}")
        keyed[key] = row
    return keyed


def ground_truth_track_id(conversations, turn_number):
    current_turn = [row for row in conversations if row["turn_number"] == turn_number]
    music_rows = [row for row in current_turn if row["role"] == "music"]
    if not music_rows:
        raise ValueError(f"No music row for turn {turn_number}")
    return music_rows[0]["content"]


def evaluate(tid, prediction_path, output_path, max_sessions=None):
    predictions = load_predictions(prediction_path)
    devset = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="test")
    if max_sessions is not None:
        devset = devset.select(range(min(max_sessions, len(devset))))
    catalog = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")

    per_turn = defaultdict(lambda: defaultdict(list))
    recommended_track_ids = []
    responses = []

    for item in devset:
        session_id = item["session_id"]
        for turn_number in range(1, 9):
            key = (session_id, turn_number)
            if key not in predictions:
                raise ValueError(f"Missing prediction for {key}")
            row = predictions[key]
            predicted = row["predicted_track_ids"][:20]
            ground_truth = ground_truth_track_id(item["conversations"], turn_number)
            recommended_track_ids.extend(predicted)
            responses.append(row.get("predicted_response", ""))
            for k in (1, 10, 20):
                per_turn[turn_number][f"ndcg@{k}"].append(ndcg_at_k(predicted, ground_truth, k))

    turn_metrics = {}
    for turn_number in sorted(per_turn):
        turn_metrics[str(turn_number)] = {
            metric: sum(values) / len(values)
            for metric, values in sorted(per_turn[turn_number].items())
        }

    macro = {}
    for metric in ("ndcg@1", "ndcg@10", "ndcg@20"):
        macro[metric] = sum(turn_metrics[str(turn)][metric] for turn in range(1, 9)) / 8
    macro["catalog_diversity"] = len(set(recommended_track_ids)) / len(catalog)
    macro["lexical_diversity"] = distinct_2(responses)
    macro["total_catalog_size"] = len(catalog)
    macro["num_predictions"] = len(predictions)

    result = {
        "tid": tid,
        "macro": macro,
        "turn_metrics": turn_metrics,
    }
    os.makedirs(Path(output_path).parent, exist_ok=True)
    Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tid", required=True)
    parser.add_argument("--prediction_path")
    parser.add_argument("--output_path")
    parser.add_argument("--max_sessions", type=int, default=None,
                        help="Truncate devset to first N sessions (for smoke evals; "
                             "must match the inference run's --max_sessions).")
    args = parser.parse_args()

    prediction_path = args.prediction_path or f"exp/inference/devset/{args.tid}.json"
    output_path = args.output_path or f"exp/scores/devset/{args.tid}.json"
    result = evaluate(args.tid, prediction_path, output_path, args.max_sessions)
    print(json.dumps(result["macro"], indent=2))


if __name__ == "__main__":
    main()
