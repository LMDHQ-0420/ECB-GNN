import os
import torch
import numpy as np
import csv
import pickle
from utilities import GraphDataset, pad_tensor, load_flat_samples, compute_extended_variable_features, preprocess_variable_features
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

def _build_cand_features_from_batch(batch, feat_specs, normalization):

    constraint_features = batch.constraint_features.cpu().numpy()
    edge_index = batch.edge_index.cpu().numpy()
    edge_attr = batch.edge_attr.cpu().numpy()
    if edge_attr.ndim == 1:
        edge_attr = edge_attr.reshape(-1, 1)
    variable_features = batch.variable_features.cpu().numpy()
    candidates = batch.candidates.cpu().numpy()

    state = (
        {'values': constraint_features},
        {'indices': edge_index, 'values': edge_attr},
        {'values': variable_features},
    )

    feat_type = feat_specs.get('type', 'khalil')
    feat_augment = feat_specs.get('augment', False)
    feat_qbnorm = feat_specs.get('qbnorm', False)

    cand_states = []
    if feat_type in ('all', 'gcnn_agg'):
        cand_states.append(compute_extended_variable_features(state, candidates))
    if feat_type in ('all', 'khalil'):
        cand_states.append(compute_extended_variable_features(state, candidates))
    if not cand_states:
        return None, None
    cand_states = np.concatenate(cand_states, axis=1)

    cand_states = preprocess_variable_features(
        cand_states,
        interaction_augmentation=feat_augment,
        normalization=feat_qbnorm,
    )

    if normalization is not None:
        x_shift, x_scale = normalization
        x_scale = np.where(x_scale == 0, 1.0, x_scale)
        cand_states = (cand_states - x_shift) / x_scale

    return cand_states, candidates.astype(np.int64)


def evaluate_gnn_on_test_set(model, device, model_name):
    test_dir = "data/samples/test"
    output_dir = "results/samples_evaluation"
    os.makedirs(output_dir, exist_ok=True)

    top_k = [1, 3, 5, 10]
    fieldnames = ['type', 'instance', 'n_samples', 'expert_valid_frac', 'replication_accuracy'] + [f'acc@{k}' for k in top_k] + ['truebest_accuracy']

    subdirs = ['Core', 'StructTransfer', 'ParamRobustness']
    difficulties = ['simple', 'middle', 'hard']

    output_csv = os.path.join(output_dir, f"eval_{model_name}.csv")
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for sub in subdirs:
            for diff in difficulties:
                category_dir = os.path.join(test_dir, sub, diff)
                if not os.path.exists(category_dir):
                    print(f"[skip] No instances found for {sub}/{diff}")
                    continue

                print(f"\n-- Evaluating Category: {sub}/{diff} --")
                instances = [os.path.join(category_dir, f) for f in os.listdir(category_dir) if f.endswith('.pkl')]
                if not instances:
                    print(f"[skip] No .pkl files in {category_dir}")
                    continue

                dataset = GraphDataset(instances)
                dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

                n_samples = 0
                total_expert_valid = 0
                n_tb_samples = 0
                k_matches = np.zeros(len(top_k), dtype=np.int64)

                all_true_labels = []
                all_pred_labels = []
                all_tb_true = []
                all_tb_pred = []

                for batch in tqdm(dataloader, desc=f"Processing {sub}/{diff}"):
                    batch = batch.to(device)
                    logits = model(batch.constraint_features, batch.edge_index, batch.edge_attr, batch.variable_features)

                    cand_logits = logits[batch.candidates]
                    logits_padded = pad_tensor(cand_logits, batch.nb_candidates)

                    true_scores = pad_tensor(batch.candidate_scores, batch.nb_candidates)

                    true_idx = int(batch.candidate_choices.cpu().item())

                    choice_score = true_scores[0, true_idx]
                    expert_valid = (choice_score > -1e7) and (not torch.isnan(choice_score))
                    if expert_valid:
                        total_expert_valid += 1

                    pred_top1 = int(logits_padded.argmax(dim=-1)[0].cpu().item())
                    all_true_labels.append(true_idx)
                    all_pred_labels.append(pred_top1)

                    valid_mask = (true_scores > -1e7) & (~torch.isnan(true_scores))
                    if bool(valid_mask.any().cpu().item()):
                        masked_true = true_scores.clone()
                        masked_true[~valid_mask] = -1e30
                        tb_idx = int(masked_true[0].argmax().cpu().item())
                        all_tb_true.append(tb_idx)
                        all_tb_pred.append(pred_top1)
                        n_tb_samples += 1

                        for i, k in enumerate(top_k):
                            C = logits_padded.size(1)
                            if C < k:
                                k_matches[i] += 1
                            else:
                                topk_idx = logits_padded[0].topk(k).indices.cpu().numpy()
                                if tb_idx in topk_idx:
                                    k_matches[i] += 1

                    n_samples += 1

                replication_acc = accuracy_score(all_true_labels, all_pred_labels) if len(all_true_labels) > 0 else 0.0
                precision = precision_score(all_true_labels, all_pred_labels, average='weighted', zero_division=0) if len(all_true_labels) > 0 else 0.0
                recall = recall_score(all_true_labels, all_pred_labels, average='weighted', zero_division=0) if len(all_true_labels) > 0 else 0.0
                f1 = f1_score(all_true_labels, all_pred_labels, average='weighted', zero_division=0) if len(all_true_labels) > 0 else 0.0

                if n_tb_samples > 0:
                    accs = (k_matches / n_tb_samples).tolist()
                    truebest_acc = accuracy_score(all_tb_true, all_tb_pred) if len(all_tb_true) > 0 else 0.0
                else:
                    accs = [0.0] * len(top_k)
                    truebest_acc = 0.0

                expert_valid_frac = (total_expert_valid / n_samples) if n_samples > 0 else 0.0

                row = {
                    'type': f"{sub}/{diff}",
                    'instance': category_dir,
                    'n_samples': n_samples,
                    'expert_valid_frac': round(float(expert_valid_frac), 6),
                    'replication_accuracy': replication_acc,
                    'truebest_accuracy': truebest_acc,
                }
                for k, acc in zip(top_k, accs):
                    row[f'acc@{k}'] = acc

                writer.writerow(row)
                csvfile.flush()


def evaluate_ml_on_test_set(ml_model, feat_specs, normalization, model_name):
    test_dir = "data/samples/test"
    output_dir = "results/samples_evaluation"
    os.makedirs(output_dir, exist_ok=True)

    top_k = [1, 3, 5, 10]
    fieldnames = ['type', 'instance', 'n_samples', 'expert_valid_frac', 'replication_accuracy'] + [f'acc@{k}' for k in top_k] + ['truebest_accuracy']

    subdirs = ['Core', 'StructTransfer', 'ParamRobustness']
    difficulties = ['simple', 'middle', 'hard']

    output_csv = os.path.join(output_dir, f"eval_{model_name}.csv")
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for sub in subdirs:
            for diff in difficulties:
                category_dir = os.path.join(test_dir, sub, diff)
                if not os.path.exists(category_dir):
                    print(f"[skip] No instances found for {sub}/{diff}")
                    continue

                print(f"\n-- Evaluating Category: {sub}/{diff} --")
                instances = [os.path.join(category_dir, f) for f in os.listdir(category_dir) if f.endswith('.pkl')]
                if not instances:
                    print(f"[skip] No .pkl files in {category_dir}")
                    continue

                dataset = GraphDataset(instances)
                dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

                n_samples = 0
                total_expert_valid = 0
                n_tb_samples = 0
                k_matches = np.zeros(len(top_k), dtype=np.int64)

                all_true_labels = []
                all_pred_labels = []
                all_tb_true = []
                all_tb_pred = []

                for batch in tqdm(dataloader, desc=f"Processing {sub}/{diff}"):
                    cand_feats, candidates = _build_cand_features_from_batch(batch, feat_specs, normalization)
                    if cand_feats is None or cand_feats.shape[0] == 0:
                        continue

                    if hasattr(ml_model, 'predict'):
                        pred = ml_model.predict(cand_feats)
                    else:
                        try:
                            import dlib
                            pred = np.array([ml_model(dlib.vector(x.tolist())) for x in cand_feats], dtype=float)
                        except Exception:
                            pred = np.zeros(cand_feats.shape[0], dtype=float)

                    true_scores = pad_tensor(batch.candidate_scores, batch.nb_candidates)
                    true_idx = int(batch.candidate_choices.cpu().item())

                    choice_score = true_scores[0, true_idx]
                    expert_valid = (choice_score > -1e7) and (not torch.isnan(choice_score))
                    if expert_valid:
                        total_expert_valid += 1

                    pred_top1 = int(np.argmax(pred))
                    all_true_labels.append(true_idx)
                    all_pred_labels.append(pred_top1)

                    valid_mask = (true_scores > -1e7) & (~torch.isnan(true_scores))
                    if bool(valid_mask.any().cpu().item()):
                        masked_true = true_scores.clone()
                        masked_true[~valid_mask] = -1e30
                        tb_idx = int(masked_true[0].argmax().cpu().item())
                        all_tb_true.append(tb_idx)
                        all_tb_pred.append(pred_top1)
                        n_tb_samples += 1

                        C = len(pred)
                        for i, k in enumerate(top_k):
                            if C < k:
                                k_matches[i] += 1
                            else:
                                topk_idx = np.argpartition(-pred, k-1)[:k]
                                if tb_idx in topk_idx:
                                    k_matches[i] += 1

                    n_samples += 1

                replication_acc = accuracy_score(all_true_labels, all_pred_labels) if len(all_true_labels) > 0 else 0.0
                precision = precision_score(all_true_labels, all_pred_labels, average='weighted', zero_division=0) if len(all_true_labels) > 0 else 0.0
                recall = recall_score(all_true_labels, all_pred_labels, average='weighted', zero_division=0) if len(all_true_labels) > 0 else 0.0
                f1 = f1_score(all_true_labels, all_pred_labels, average='weighted', zero_division=0) if len(all_true_labels) > 0 else 0.0

                if n_tb_samples > 0:
                    accs = (k_matches / n_tb_samples).tolist()
                    truebest_acc = accuracy_score(all_tb_true, all_tb_pred) if len(all_tb_true) > 0 else 0.0
                else:
                    accs = [0.0] * len(top_k)
                    truebest_acc = 0.0

                expert_valid_frac = (total_expert_valid / n_samples) if n_samples > 0 else 0.0

                row = {
                    'type': f"{sub}/{diff}",
                    'instance': category_dir,
                    'n_samples': n_samples,
                    'expert_valid_frac': round(float(expert_valid_frac), 6),
                    'replication_accuracy': replication_acc,
                    'truebest_accuracy': truebest_acc,
                }
                for k, acc in zip(top_k, accs):
                    row[f'acc@{k}'] = acc

                writer.writerow(row)
                csvfile.flush()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate trained model on test set.")
    parser.add_argument('--model', required=True, choices=['baseline', 'ecbgnn', 'extratrees', 'lambdamart', 'svmrank'], help="Model type to evaluate")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model in ('baseline', 'ecbgnn'):
        if args.model == 'baseline':
            from model.baseline import GNNPolicy
        elif args.model == 'ecbgnn':
            from model.ecbgnn import GNNPolicy

        model = GNNPolicy().to(device)
        model_path = f"runs/{args.model}/params_{args.model}.pkl"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}")

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        evaluate_gnn_on_test_set(model, device, args.model)
    else:
        model_candidates = []
        if args.model == 'svmrank':
            model_candidates.extend(['model_dlib_svm.pkl', 'model_sklearn_svr.pkl'])
        if args.model == 'lambdamart':
            model_candidates.append('model_lambdamart_gbr.pkl')
        if args.model == 'extratrees':
            model_candidates.append('model.pkl')

        model_file = None
        for fname in model_candidates:
            p = os.path.join('runs', args.model, fname)
            if os.path.exists(p):
                model_file = p
                break
        if model_file is None:
            raise FileNotFoundError(f"No trained model file for {args.model} (searched {model_candidates})")

        norm_file = os.path.join('runs', args.model, 'normalization.pkl')
        feat_file = os.path.join('runs', args.model, 'feat_specs.pkl')
        if not os.path.exists(norm_file) or not os.path.exists(feat_file):
            raise FileNotFoundError(f"Missing normalization or feat specs for {args.model}: {norm_file}, {feat_file}")

        with open(model_file, 'rb') as f:
            ml_model = pickle.load(f)
        with open(norm_file, 'rb') as f:
            normalization = pickle.load(f)
        with open(feat_file, 'rb') as f:
            feat_specs = pickle.load(f)

        evaluate_ml_on_test_set(ml_model, feat_specs, normalization, args.model)