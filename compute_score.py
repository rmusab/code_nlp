# Implementation of the Singular Value Decomposition Word Rotator's Distance
# (SVDWRD) metric for multi-purpose evaluation of natural language generation models
# © Mussabayev Ravil, 2022

from process_tokens import *
from my_wrd import *
import argparse


def get_svdwrd_prec_and_recall(wrd_model, ref, pred):
    wrd_return_tuple = wrd_model.wrdistance(ref, pred)
    if wrd_return_tuple == float('inf'):
        return False, None, None
    _, T, dct, d_mtx, s_mtx = wrd_return_tuple
    n = len(dct)
    token2id = dct.token2id
    ref_total_sim = 0.
    for token in token2id:
        i = token2id[token]
        if token in ref:
            token_sim = 0.
            token_total_flow = 0.
            for j in range(n):
                if T[i][j] > 0.:
                    token_sim += T[i][j] * s_mtx[i, j]
                    token_total_flow += T[i][j]
            if token_total_flow == 0: continue
            token_sim /= token_total_flow
            ref_total_sim += token_sim
    ref_len = len(set(ref))
    recall = ref_total_sim / ref_len
    pred_total_sim = 0.
    for token in token2id:
        j = token2id[token]
        if token in pred:
            token_sim = 0.
            token_total_flow = 0.
            for i in range(n):
                if T[i][j] > 0.:
                    token_sim += T[i][j] * s_mtx[i, j]
                    token_total_flow += T[i][j]
            if token_total_flow == 0: continue
            token_sim /= token_total_flow
            pred_total_sim += token_sim
    pred_len = len(set(pred))
    precision = pred_total_sim / pred_len
    return True, precision, recall


def get_simple_prec_and_recall(ref, pred):
    ref = set(ref)
    pred = set(pred)
    TP = len(ref.intersection(pred))
    FP = len(pred.difference(ref))
    FN = len(ref.difference(pred))
    if TP == 0: return 0., 0., 0.
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0.
    return precision, recall, f1


def main(args):
    # Load true labels
    references = []
    with open(args.refs_path):
        for line in f:
            line_tokens = []
            for token in line.split():
                subtokens = wordninja.split(token)
                line_tokens.extend(subtokens)
            references.append(line_tokens)
    # Load model predictions
    predictions = []
    with open(args.preds_path) as f:
        for line in f:
            line_tokens = []
            for token in line.split():
                subtokens = wordninja.split(token)
                line_tokens.extend(subtokens)
            predictions.append(line_tokens)
    n = len(references)
    # assert len(references) == len(predictions), "Length of the prediction vector doesn't equal the length of the reference vector!"
    # Load our metric
    voc = read_tokens()  # load the vocabulary
    w = load_embedding_mtx()  # load the embedding matrix
    word_embeds = {voc[i]: w[i] for i in range(len(voc))}
    wrd_model = WRD(model=word_embeds)
    # Calculate total precision, recall, and F1 scores
    total_precision = 0.
    total_recall = 0.
    total = 0
    if args.rouge:
        total_old_precision = 0.
        total_old_recall = 0.
        total_old = 0
    output_file = open('model_evaluation.txt', 'w')
    for i in tqdm(range(n), total=n):
        if args.rouge:
            best_new_score = (0., 0., 0.)
        else:
            best_old_score = (0., 0., 0.)
        at_least_one_include = False
        best_new_j = -1
        best_old_j = -1
        for j in range(args.beam_width):
            include, precision, recall = get_svdwrd_prec_and_recall(wrd_model, references[i], predictions[args.beam_width * i + j])
            if args.rouge:
                old_p, old_r, old_f1 = get_simple_prec_and_recall(references[i], predictions[i + j])
            if include:
                at_least_one_include = True
                f1 = 2 * precision * recall / (precision + recall)
                if f1 > best_new_score[2]:
                    best_new_score = (precision, recall, f1)
                    best_new_j = j
                if args.rouge and old_f1 > best_old_score[2]:
                    best_old_score = (old_p, old_r, old_f1)
                    best_old_j = j
        if at_least_one_include:
            if args.rouge:
                print(f"{' '.join(references[i])} <-> {' '.join(predictions[args.beam_width * i + best_new_j])}: p={best_new_score[0]:0.4f}, r={best_new_score[1]:0.4f}, f1={best_new_score[2]:0.4f}; <-> {' '.join(predictions[args.beam_width * i + best_old_j])}: old_p={best_old_score[0]:0.4f}, old_r={best_old_score[1]:0.4f}, old_f1={best_old_score[2]:0.4f}", file=output_file)
            else:
                print(f"{' '.join(references[i])} <-> {' '.join(predictions[args.beam_width * i + best_new_j])}: p={best_new_score[0]:0.4f}, r={best_new_score[1]:0.4f}, f1={best_new_score[2]:0.4f}", file=output_file)
            total += 1
            total_precision += best_new_score[0]
            total_recall += best_new_score[1]
            if args.rouge:
                total_old += 1
                total_old_precision += best_old_score[0]
                total_old_recall += best_old_score[1]
    output_file.close()
    total_precision /= total
    total_recall /= total
    f1 = 2 * total_precision * total_recall / (total_precision + total_recall)
    if args.rouge:
        total_old_precision /= total_old
        total_old_recall /= total_old
        f1_old = 2 * total_old_precision * total_old_recall / (total_old_precision + total_old_recall)
        print(f"Total ROUGE-1 precision: {total_old_precision}")
        print(f"Total ROUGE-1 recall: {total_old_recall}")
        print(f"Total ROUGE-1 f1: {f1_old}")
    print(f"Total SVDWRD precision: {total_precision}")
    print(f"Total SVDWRD recall: {total_recall}")
    print(f"Total SVDWRD f1: {f1}")


parser = argparse.ArgumentParser(description="In the default regime, compute the Singular Value Decomposition Word Rotator's Distance (SVDWRD) score. \
                                Provide the -r and -p parameters for that case. For both cases specification of the beam width -b (default = 1) is needed. \
                                Additionally, compute the regular (ROUGE-1) precision, recall, and F1 scores for the given set of \
                                references and predictions (ROUGE is True).")
parser.add_argument('-r', '--rouge', action='store_true', default=False,
                    help='compute the ROUGE-1 scores (default: compute the SVDWRD scores')
parser.add_argument('-r', '--refs', dest='refs_path',
                    help='path to the database of references line by line')
parser.add_argument('-p', '--preds', dest='preds_path',
                    help='path to the database of model predictions line by line')
parser.add_argument('-b', '--beam', dest='beam_width', type=int, default=1,
                    help='beam width (default = 1)')
args = parser.parse_args()

if __name__ == '__main__':
    main(args)