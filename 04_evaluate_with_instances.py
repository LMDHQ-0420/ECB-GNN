import os
import sys
import argparse
import csv
import numpy as np
import time
import glob

import ecole
import pyscipopt
import torch
import pickle
from utilities import compute_extended_variable_features, preprocess_variable_features


def collect_test_instances_for(subdir: str, difficulty: str):
    instances = []
    test_dir = os.path.join('data', 'instances', 'test', subdir, difficulty)
    for lp in glob.glob(os.path.join(test_dir, '**', '*.lp'), recursive=True):
        instances.append({'type': f"{subdir}/{difficulty}", 'path': lp})
    return instances


def select_device(gpu_index: int):
    if gpu_index == -1 or not torch.cuda.is_available():
        return 'cpu'
    total = torch.cuda.device_count()
    if gpu_index >= total:
        return 'cuda:0'
    return f'cuda:{gpu_index}'


def load_policy(model_name: str, device: str):
    if model_name == 'baseline':
        from model.baseline import GNNPolicy
    elif model_name == 'ecbgnn':
        from model.ecbgnn import GNNPolicy

    weights_path = os.path.join('runs', model_name, f'params_{model_name}.pkl')
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    policy = GNNPolicy().to(device)
    policy.load_state_dict(torch.load(weights_path, map_location=device))
    policy.eval()
    return policy


def run_internal_brancher(instance_path: str, brancher: str, seed: int, scip_parameters: dict):
    env = ecole.environment.Configuring(scip_params={**scip_parameters, f"branching/{brancher}/priority": 9999999})
    env.seed(seed)
    walltime = time.perf_counter()
    proctime = time.process_time()
    env.reset(instance_path)
    _ = env.step({})
    walltime = time.perf_counter() - walltime
    proctime = time.process_time() - proctime
    scip_model = env.model.as_pyscipopt()
    return scip_model, walltime, proctime


def run_gnn_policy(instance_path: str, model, seed: int, device: str, scip_parameters: dict):
    env = ecole.environment.Branching(observation_function=ecole.observation.NodeBipartite(), scip_params=scip_parameters)
    env.seed(seed)
    torch.manual_seed(seed)
    walltime = time.perf_counter()
    proctime = time.process_time()
    observation, action_set, _, done, _ = env.reset(instance_path)
    while not done:
        with torch.no_grad():
            row_feats = np.nan_to_num(observation.row_features.astype(np.float32), nan=-1e8)
            edge_idx = observation.edge_features.indices.astype(np.int64)
            edge_vals = np.nan_to_num(observation.edge_features.values.astype(np.float32), nan=-1e8)
            var_feats = np.nan_to_num(observation.variable_features.astype(np.float32), nan=-1e8)

            obs_t = (
                torch.from_numpy(row_feats).to(device),
                torch.from_numpy(edge_idx).to(device),
                torch.from_numpy(edge_vals).view(-1, 1).to(device),
                torch.from_numpy(var_feats).to(device),
            )

            logits = model(*obs_t)

            action_set_np = action_set.astype(np.int64)
            if action_set_np.size == 0:
                break
            action_idx = torch.from_numpy(action_set_np).long().to(device)
            cand_logits = logits[action_idx]
            local_choice = int(cand_logits.argmax().cpu().item())
            action = int(action_set_np[local_choice])
            observation, action_set, _, done, _ = env.step(action)
    walltime = time.perf_counter() - walltime
    proctime = time.process_time() - proctime
    scip_model = env.model.as_pyscipopt()
    return scip_model, walltime, proctime


def build_candidate_features(observation, action_set, feat_specs, normalization):
    row_feats = np.nan_to_num(observation.row_features.astype(np.float32), nan=-1e8)
    edge_idx = observation.edge_features.indices.astype(np.int64)
    edge_vals = np.nan_to_num(observation.edge_features.values.astype(np.float32), nan=-1e8)
    if edge_vals.ndim == 1:
        edge_vals = edge_vals.reshape(-1, 1)
    var_feats = np.nan_to_num(observation.variable_features.astype(np.float32), nan=-1e8)

    state = (
        {'values': row_feats},
        {'indices': edge_idx, 'values': edge_vals},
        {'values': var_feats},
    )

    cands = action_set.astype(np.int64)

    feat_type = feat_specs.get('type', 'khalil')
    feat_augment = feat_specs.get('augment', False)
    feat_qbnorm = feat_specs.get('qbnorm', False)

    cand_states = []
    if feat_type in ('all', 'gcnn_agg'):
        cand_states.append(compute_extended_variable_features(state, cands))
    if feat_type in ('all', 'khalil'):
        cand_states.append(compute_extended_variable_features(state, cands))
    if len(cand_states) == 0:
        return None
    cand_states = np.concatenate(cand_states, axis=1)
    cand_states = preprocess_variable_features(
        cand_states,
        interaction_augmentation=feat_augment,
        normalization=feat_qbnorm,
    )

    if normalization is not None:
        x_shift, x_scale = normalization
        cand_states = (cand_states - x_shift) / x_scale

    return cand_states, cands


def run_ml_policy(instance_path: str, model, normalization, feat_specs, seed: int, scip_parameters: dict):
    env = ecole.environment.Branching(observation_function=ecole.observation.NodeBipartite(), scip_params=scip_parameters)
    env.seed(seed)
    torch.manual_seed(seed)
    walltime = time.perf_counter()
    proctime = time.process_time()
    observation, action_set, _, done, _ = env.reset(instance_path)
    while not done:
        action_set_np = action_set.astype(np.int64)
        if action_set_np.size == 0:
            break

        cand_feats, cands = build_candidate_features(observation, action_set_np, feat_specs, normalization)
        if cand_feats is None or cand_feats.shape[0] == 0:
            break

        # predict scores
        if hasattr(model, 'predict'):
            pred = model.predict(cand_feats)
        else:
            try:
                import dlib
                pred = np.array([model(dlib.vector(x.tolist())) for x in cand_feats], dtype=float)
            except Exception:
                pred = np.zeros(len(cands), dtype=float)

        action_local = int(np.argmax(pred))
        action = int(cands[action_local])
        observation, action_set, _, done, _ = env.step(action)

    walltime = time.perf_counter() - walltime
    proctime = time.process_time() - proctime
    scip_model = env.model.as_pyscipopt()
    return scip_model, walltime, proctime


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained policies (GNN or ML) under gap limit')
    parser.add_argument(
        '--model',
        help='Which GNN model to evaluate',
        choices=['baseline', 'ecbgnn', 'scip', 'extratrees', 'lambdamart', 'svmrank'],
        required=True,
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--brancher',
        help='SCIP internal brancher to evaluate when running scip',
        type=str,
        default='relpscost',
    )
    parser.add_argument(
        '--gap_limit',
        help='Gap limit for SCIP solving in seconds.',
        type=float,
        default=2,
    )

    args = parser.parse_args()

    run_gnn = args.model in ('baseline', 'attention')
    run_scip = args.model == 'scip'
    run_ml = args.model in ('extratrees', 'lambdamart', 'svmrank')

    timestamp = time.strftime('%Y%m%d-%H%M%S')
    SEED = 42
    internal_brancher = args.brancher
    gap_limit = args.gap_limit

    # 3x3 test categories
    subdirs = ['Core', 'StructTransfer', 'ParamRobustness']
    difficulties = ['simple', 'middle', 'hard']

    print(f"model: {args.model}")
    print(f"gpu: {args.gpu}")
    if run_scip:
        print(f"brancher: {internal_brancher}")
    print(f"gap limit: {gap_limit} %")

    device = select_device(args.gpu)
    gnn_policy = None
    ml_model = None
    normalization = None
    feat_specs = None
    if run_gnn:
        gnn_policy = load_policy(args.model, device)
    if run_ml:
        model_candidates = []
        if args.model == 'svmrank':
            model_candidates.extend(['model_dlib_svm.pkl', 'model_sklearn_svr.pkl'])
        if args.model == 'lambdamart':
            model_candidates.append('model_lambdamart_gbr.pkl')
        if args.model == 'extratrees':
            model_candidates.append('model.pkl')

        model_file = None
        for fname in model_candidates:
            path = os.path.join('runs', args.model, fname)
            if os.path.exists(path):
                model_file = path
                break
        if model_file is None:
            raise FileNotFoundError(f"No trained model found for {args.model} (looked for {model_candidates})")

        norm_file = os.path.join('runs', args.model, 'normalization.pkl')
        feat_file = os.path.join('runs', args.model, 'feat_specs.pkl')
        if not os.path.exists(norm_file) or not os.path.exists(feat_file):
            raise FileNotFoundError(f"Normalization or feature spec missing for {args.model}: {norm_file}, {feat_file}")

        with open(model_file, 'rb') as f:
            ml_model = pickle.load(f)
        with open(norm_file, 'rb') as f:
            normalization = pickle.load(f)
        with open(feat_file, 'rb') as f:
            feat_specs = pickle.load(f)

    fieldnames = [
        'policy', 'seed', 'type', 'instance', 'nnodes', 'nlps', 'stime', 'gap', 'status', 'walltime', 'proctime',
    ]
    # Prepare base results dir
    os.makedirs(f'results/gap_limit_{gap_limit}', exist_ok=True)
    scip_parameters = {
        'separating/maxrounds': 0,
        'presolving/maxrestarts': 0,
        'limits/gap': gap_limit,
        'timing/clocktype': 1,
        'branching/vanillafullstrong/idempotent': True,
        'display/verblevel': 4,
        'display/freq': 1,
        'display/width': 120,
    }

    for sub in subdirs:
        for diff in difficulties:
            category_instances = collect_test_instances_for(sub, diff)
            if not category_instances:
                print(f"[skip] No instances found for {sub}/{diff}")
                continue

            print(f"\n-- Category: {sub}/{diff} --")

            # Ensure per-category directory exists
            out_dir = os.path.join(f'results/gap_limit_{gap_limit}', sub, diff)
            os.makedirs(out_dir, exist_ok=True)

            # GNN evaluation per category (CSV in same category dir)
            if run_gnn:
                out_path_gnn = os.path.join(out_dir, f"eval_{args.model}_{timestamp}.csv")
                print(f"==> Writing GNN results to {out_path_gnn}")
                with open(out_path_gnn, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for instance in category_instances:
                        print(f"{instance['type']}: {instance['path']}...")
                        seed = SEED
                        scip_model, walltime, proctime = run_gnn_policy(instance['path'], gnn_policy, seed, device, scip_parameters)

                        stime = scip_model.getSolvingTime()
                        nnodes = scip_model.getNNodes()
                        nlps = scip_model.getNLPs()
                        gap = scip_model.getGap()
                        status = scip_model.getStatus()

                        writer.writerow({
                            'policy': f"gnn:{args.model}",
                            'seed': seed,
                            'type': instance['type'],
                            'instance': instance['path'],
                            'nnodes': nnodes,
                            'nlps': nlps,
                            'stime': stime,
                            'gap': gap,
                            'status': status,
                            'walltime': walltime,
                            'proctime': proctime,
                        })
                        csvfile.flush()

                        print(f"  gnn:{args.model} {seed} - {nnodes} nodes {nlps} lps {stime:.2f} ({walltime:.2f} wall {proctime:.2f} proc) s. {status}")

            if run_ml:
                out_path_ml = os.path.join(out_dir, f"eval_{args.model}_{timestamp}.csv")
                print(f"==> Writing ML results to {out_path_ml}")
                with open(out_path_ml, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for instance in category_instances:
                        print(f"{instance['type']}: {instance['path']}...")
                        seed = SEED
                        scip_model, walltime, proctime = run_ml_policy(instance['path'], ml_model, normalization, feat_specs, seed, scip_parameters)

                        stime = scip_model.getSolvingTime()
                        nnodes = scip_model.getNNodes()
                        nlps = scip_model.getNLPs()
                        gap = scip_model.getGap()
                        status = scip_model.getStatus()

                        writer.writerow({
                            'policy': f"ml:{args.model}",
                            'seed': seed,
                            'type': instance['type'],
                            'instance': instance['path'],
                            'nnodes': nnodes,
                            'nlps': nlps,
                            'stime': stime,
                            'gap': gap,
                            'status': status,
                            'walltime': walltime,
                            'proctime': proctime,
                        })
                        csvfile.flush()

                        print(f"  ml:{args.model} {seed} - {nnodes} nodes {nlps} lps {stime:.2f} ({walltime:.2f} wall {proctime:.2f} proc) s. {status}")

            if run_scip:
                out_path_scip = os.path.join(out_dir, f"eval_scip_{timestamp}.csv")
                print(f"==> Writing SCIP results to {out_path_scip}")
                with open(out_path_scip, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for instance in category_instances:
                        print(f"{instance['type']}: {instance['path']}...")
                        seed = SEED
                        scip_model, walltime, proctime = run_internal_brancher(instance['path'], internal_brancher, seed, scip_parameters)

                        stime = scip_model.getSolvingTime()
                        nnodes = scip_model.getNNodes()
                        nlps = scip_model.getNLPs()
                        gap = scip_model.getGap()
                        status = scip_model.getStatus()

                        writer.writerow({
                            'policy': f"internal:{internal_brancher}",
                            'seed': seed,
                            'type': instance['type'],
                            'instance': instance['path'],
                            'nnodes': nnodes,
                            'nlps': nlps,
                            'stime': stime,
                            'gap': gap,
                            'status': status,
                            'walltime': walltime,
                            'proctime': proctime,
                        })
                        csvfile.flush()

                        print(f"  internal:{internal_brancher} {seed} - {nnodes} nodes {nlps} lps {stime:.2f} ({walltime:.2f} wall {proctime:.2f} proc) s. {status}")
