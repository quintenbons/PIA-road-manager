#!/usr/bin/python3
import torch
import argparse
import os

from torch.utils.data import DataLoader
from ai.dataset import BenchNodeDataset
from ai.model import CrossRoadModel
import torch.nn.functional as F

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="model.pt")
    parser.add_argument("--dataset-path", type=str, default="dataset.pt")
    args = parser.parse_args()

    if not os.path.exists(args.model_path) or not os.path.exists(args.dataset_path):
        print(f"Dataset or model missing {args.model_path} {args.dataset_path}")
        exit(1)

    model = CrossRoadModel.load(args.model_path)
    dataset = BenchNodeDataset.load(args.dataset_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    hit = 0
    miss = 0
    pos1 = 0
    pos2 = 0
    total_score_difference = 0
    total_relative_error = 0
    total_worst_relative_error = 0

    with torch.no_grad():
        model = model.eval()
        for batch_idx, (inputs, expected, scores) in enumerate(dataloader):
            results = F.softmax(model(inputs), dim=1)
            for output, target, score in zip(results, expected, scores):
                hitormiss = "\033[92mHIT \033[0m" if output.argmax().item() == target.argmax().item() else "\033[91mMISS\033[0m"
                # print(f"{hitormiss} Got: {output.argmax().item():2d} Expected: {target.argmax().item():2d}")
                if output.argmax().item() == target.argmax().item():
                    hit += 1
                else:
                    miss += 1

                # Sort output indexes
                sorted_output = sorted(enumerate(output), key=lambda x: x[1], reverse=True)
                sorted_output = [x[0] for x in sorted_output]

                sorted_target = sorted(enumerate(target), key=lambda x: x[1], reverse=True)
                sorted_target = [x[0] for x in sorted_target]

                best_index = sorted_output.index(sorted_target[0])
                pos1 += best_index

                output_index = sorted_target.index(sorted_output[0])
                pos2 += output_index

                # Score check
                top_prediction_index = sorted_output[0]
                top_prediction_score = score[top_prediction_index]

                worst_score = score.max()
                optimal_score = score.min()
                score_difference = top_prediction_score - optimal_score
                relative_error = score_difference / optimal_score if optimal_score != 0 else score_difference

                worst_score_difference = worst_score - optimal_score
                worst_relative_error = worst_score_difference / optimal_score if optimal_score != 0 else worst_score_difference

                total_score_difference += score_difference
                total_relative_error += relative_error
                total_worst_relative_error += worst_relative_error

                # Output a bench_data.csv
                # print("strategy_id,strategy_name,mutator_id,loss_score,target_score,output_score")
                # strategy_manager = StrategyManager()
                # for idx, (typ, mutation) in enumerate(strategy_manager.enumerate_strategy_schemes(4)):
                #     tscore = score[idx].item()
                #     ttarget = target[idx].item()
                #     toutput = output[idx].item()
                #     print(f"{typ},{STRAT_NAMES[typ]},{mutation},{tscore},{ttarget},{toutput}")
                # return

        avg_score_difference = total_score_difference / (hit + miss)
        avg_relative_error = total_relative_error / (hit + miss)
        avg_worst_relative_error = total_worst_relative_error / (hit + miss)
        print()
        print(f"HIT rate: {hit / (hit + miss) * 100:.1f}% ({hit}/{hit+miss})")
        print(f"Average position of best case of expected in prediction: {pos1 / (hit + miss):.1f}")
        print(f"Average position of best case of prediction in expected: {pos2 / (hit + miss):.1f} (this one if more important)")
        print(f"Average score difference: {avg_score_difference}")
        print(f"Average relative error: {avg_relative_error}")
        print(f"Average worst relative error: {avg_worst_relative_error}")


if __name__ == "__main__":
    main()
