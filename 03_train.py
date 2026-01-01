import os
import sys
import argparse
import pathlib
import numpy as np
import torch
import pickle
import utilities
from utilities import log, load_flat_samples


def pretrain(policy, pretrain_loader):
    policy.pre_train_init()
    i = 0
    while True:
        for batch in pretrain_loader:
            batch.to(device)
            if not policy.pre_train(batch.constraint_features, batch.edge_index, batch.edge_attr, batch.variable_features):
                break

        if policy.pre_train_next() is None:
            break
        i += 1
    return i


def process(policy, data_loader, top_k=[1, 3, 5, 10], optimizer=None):
    mean_loss = 0
    mean_kacc = np.zeros(len(top_k))
    mean_entropy = 0

    mean_rep_acc = 0
    total_expert_valid = 0

    n_samples_processed = 0
    n_tb_samples_processed = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in data_loader:
            batch = batch.to(device)
            logits = policy(batch.constraint_features, batch.edge_index, batch.edge_attr, batch.variable_features)
            logits = pad_tensor(logits[batch.candidates], batch.nb_candidates)

            cross_entropy_loss = F.cross_entropy(logits, batch.candidate_choices, reduction='mean')
            entropy = (-F.softmax(logits, dim=-1)*F.log_softmax(logits, dim=-1)).sum(-1).mean()
            loss = cross_entropy_loss - entropy_bonus*entropy

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            true_scores = pad_tensor(batch.candidate_scores, batch.nb_candidates)

            choice_idx = batch.candidate_choices.unsqueeze(-1)
            choice_scores = true_scores.gather(1, choice_idx).squeeze(1)
            expert_valid_mask = (choice_scores > -1e7) & (~torch.isnan(choice_scores))
            # count expert-valid samples in this batch
            batch_expert_valid_count = int(expert_valid_mask.sum().item())
            total_expert_valid += batch_expert_valid_count

            pred_top1 = logits.argmax(dim=-1)
            rep_acc_batch = (pred_top1 == batch.candidate_choices).float().mean().item()
            mean_rep_acc += rep_acc_batch * batch.num_graphs

            valid_mask = (true_scores > -1e7) & (~torch.isnan(true_scores))
            masked_true_scores = true_scores.clone()
            masked_true_scores[~valid_mask] = -1e30
            tb_idx = masked_true_scores.argmax(dim=-1)

            valid_tb_mask = valid_mask.any(dim=-1)
            valid_tb_count = int(valid_tb_mask.sum().item())

            kacc = []
            for k in top_k:
                if logits.size()[-1] < k:
                    kacc.append(1.0)
                    continue
                pred_top_k = logits.topk(k).indices
                matches = (pred_top_k == tb_idx.unsqueeze(-1))
                matches_any = matches.any(dim=-1)
                if valid_tb_count > 0:
                    accuracy = matches_any[valid_tb_mask].float().mean().item()
                else:
                    accuracy = 0.0
                kacc.append(accuracy)
            kacc = np.asarray(kacc)
            mean_loss += cross_entropy_loss.item() * batch.num_graphs
            mean_kacc += np.asarray(kacc) * (valid_tb_count if valid_tb_count > 0 else 0)
            n_tb_samples_processed += valid_tb_count
            mean_entropy += entropy.item() * batch.num_graphs
            mean_rep_acc += 0
            n_samples_processed += batch.num_graphs

    if n_samples_processed == 0:
        return 0.0, np.zeros(len(top_k)), 0.0, 0.0, 0.0

    mean_loss /= n_samples_processed
    mean_entropy /= n_samples_processed
    mean_rep_acc /= n_samples_processed
    expert_valid_frac = total_expert_valid / n_samples_processed

    if n_tb_samples_processed > 0:
        mean_kacc /= n_tb_samples_processed
    else:
        mean_kacc = np.zeros(len(top_k))

    return mean_loss, mean_kacc, mean_entropy, mean_rep_acc, expert_valid_frac


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        help='Which model to train.',
        choices=['baseline', 'ecbgnn', 'extratrees', 'lambdamart', 'svmrank'],
        required=True,
    )
    args = parser.parse_args()

    max_epochs = 10000
    batch_size = 16
    pretrain_batch_size = 128
    valid_batch_size = 128
    lr = 1e-3
    entropy_bonus = 0.0
    top_k = [1, 3, 5, 10]

    SEED = 42
    rng = np.random.RandomState(SEED)
    running_dir = f"runs/{args.model}"
    os.makedirs(running_dir, exist_ok=True)

    if args.model in ('extratrees', 'lambdamart', 'svmrank'):
        if args.model == 'extratrees':
            feat_type = 'gcnn_agg'
            feat_qbnorm = False
            feat_augment = False
            label_type = 'scores'

        elif args.model == 'lambdamart':
            feat_type = 'khalil'
            feat_qbnorm = True
            feat_augment = False
            label_type = 'bipartite_ranks'

        elif args.model == 'svmrank':
            feat_type = 'khalil'
            feat_qbnorm = True
            feat_augment = True
            label_type = 'bipartite_ranks'

        def load_samples(
            filenames,
            feat_type,
            label_type,
            augment,
            qbnorm,
            max_size=None,
            logfile=None
        ):
            x_list, y_list, ncands = [], [], []
            total_ncands = 0

            for i, fname in enumerate(filenames):
                cand_x, cand_y_raw, best = load_flat_samples(
                    str(fname), feat_type, label_type, augment, qbnorm
                )

                n = cand_x.shape[0]
                if n == 0:
                    continue

                if label_type == 'scores':
                    if cand_y_raw.shape[0] != n:
                        cand_y = np.zeros(n, dtype=np.float32)
                        if best is not None and 0 <= best < n:
                            cand_y[best] = 1.0
                    else:
                        cand_y = cand_y_raw.astype(np.float32)

                elif label_type == 'bipartite_ranks':
                    cand_y = np.zeros(n, dtype=np.int32)
                    if best is not None and 0 <= best < n:
                        cand_y[best] = 1
                else:
                    raise ValueError(f"Unknown label_type: {label_type}")

                x_list.append(cand_x)
                y_list.append(cand_y)
                ncands.append(n)
                total_ncands += n

                if (i + 1) % 100 == 0:
                    log(f"  {i+1} files processed ({total_ncands} candidate variables)", logfile)

                if max_size is not None and total_ncands >= max_size:
                    log(f"  reached max_size={max_size}; stopping early", logfile)
                    break

            if len(x_list) == 0:
                return np.empty((0,)), np.empty((0,)), np.empty((0,))

            x = np.concatenate(x_list, axis=0)
            y = np.concatenate(y_list, axis=0)
            ncands = np.asarray(ncands, dtype=np.int32)

            if max_size is not None and total_ncands > max_size:
                overflow = total_ncands - max_size
                x = x[:-overflow]
                y = y[:-overflow]
                ncands[-1] -= overflow

            return x, y, ncands


        train_files = sorted(pathlib.Path('data/samples/train').rglob('*.pkl'))
        valid_files = sorted(pathlib.Path('data/samples/valid').rglob('*.pkl'))

        if args.model == 'extratrees':
            feat_type = 'gcnn_agg'
            feat_qbnorm = False
            feat_augment = False
            label_type = 'scores'
            train_max_size = 100_000_000_000_000
            valid_max_size = 100_000_000_000_000
        elif args.model == 'lambdamart':
            feat_type = 'khalil'
            feat_qbnorm = True
            feat_augment = False
            label_type = 'bipartite_ranks'
            train_max_size = 250_000_000
            valid_max_size = 100_000_000
        elif args.model == 'svmrank':
            feat_type = 'khalil'
            feat_qbnorm = True
            feat_augment = True
            label_type = 'bipartite_ranks'
            train_max_size = 100_000_000_000_000
            valid_max_size = 100_000_000_000_000

        log(
            f"Using {len(train_files)} train files and {len(valid_files)} valid files "
            f"(subsampled for ML baseline)",
            os.path.join(running_dir, 'log.txt')
        )


        train_x, train_y, train_ncands = load_samples(train_files, feat_type, label_type, feat_augment, feat_qbnorm, max_size=train_max_size, logfile=os.path.join(running_dir, 'log.txt'))
        valid_x, valid_y, valid_ncands = load_samples(valid_files, feat_type, label_type, feat_augment, feat_qbnorm, max_size=valid_max_size, logfile=os.path.join(running_dir, 'log.txt'))

        if train_x.size == 0:
            raise RuntimeError('No training samples loaded for ML model')
        x_shift = train_x.mean(axis=0)
        x_scale = train_x.std(axis=0)
        x_scale[x_scale == 0] = 1
        train_x = (train_x - x_shift) / x_scale
        valid_x = (valid_x - x_shift) / x_scale if valid_x.size > 0 else valid_x

        with open(os.path.join(running_dir, 'normalization.pkl'), 'wb') as f:
            pickle.dump((x_shift, x_scale), f)
        with open(os.path.join(running_dir, 'feat_specs.pkl'), 'wb') as f:
            pickle.dump({'type': feat_type, 'augment': feat_augment, 'qbnorm': feat_qbnorm}, f)

        logfile = os.path.join(running_dir, 'log.txt')

        if args.model == 'extratrees':
            from sklearn.ensemble import ExtraTreesRegressor
            rng = np.random.RandomState(SEED)
            model = ExtraTreesRegressor(n_estimators=100, random_state=rng)
            model.fit(train_x, train_y)
            with open(os.path.join(running_dir, 'model.pkl'), 'wb') as f:
                pickle.dump(model, f)
            if valid_x.size > 0:
                loss = np.mean((model.predict(valid_x) - valid_y) ** 2)
                log(f"Validation RMSE: {np.sqrt(loss):.2f}", logfile)
            sys.exit(0)

        elif args.model == 'lambdamart':
            try:
                import pyltr
            except Exception as e:
                log('pyltr is required for lambdamart training but is not available', logfile)
                raise
            train_qids = np.repeat(np.arange(len(train_ncands)), train_ncands)
            valid_qids = np.repeat(np.arange(len(valid_ncands)), valid_ncands) if valid_ncands.size > 0 else np.array([])
            rng = np.random.RandomState(SEED)
            model = pyltr.models.LambdaMART(verbose=1, random_state=rng, n_estimators=500)
            model.fit(train_x, train_y, train_qids, monitor=pyltr.models.monitors.ValidationMonitor(valid_x, valid_y, valid_qids, metric=model.metric))
            with open(os.path.join(running_dir, 'model_lambdamart_gbr.pkl'), 'wb') as f:
                pickle.dump(model, f)
            if valid_x.size > 0:
                loss = model.metric.calc_mean(valid_qids, valid_y, model.predict(valid_x))
                log(f"Validation log-NDCG: {np.log(loss)}", logfile)
            sys.exit(0)

        elif args.model == 'svmrank':
            train_qids = np.repeat(np.arange(len(train_ncands)), train_ncands)
            valid_qids = np.repeat(np.arange(len(valid_ncands)), valid_ncands) if valid_ncands.size > 0 else np.array([])
            
            from sklearn.svm import SVR
            best_loss = np.inf
            best_c = None
            for c in (1e-3, 1e-2, 1e-1, 1e0):
                log(f"SVR C: {c}", logfile)
                model = SVR(C=c)
                model.fit(train_x, train_y)
                pred_val = model.predict(valid_x) if valid_x.size > 0 else None
                loss_val = np.mean((pred_val - valid_y) ** 2) if valid_x.size > 0 else np.inf
                log(f"  validation loss: {loss_val}", logfile)
                if loss_val < best_loss:
                    best_loss = loss_val
                    best_model = model
                    best_c = c
                    # save model
                    with open(os.path.join(running_dir, 'model_sklearn_svr.pkl'), 'wb') as f:
                        pickle.dump(model, f)
            log(f"Best SVR model with C={best_c}, validation loss: {best_loss}", logfile)
        sys.exit(0)

    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(torch.cuda.device_count())))
        device = "cuda"
    else:
        device = "cpu"
    import torch.nn.functional as F
    import torch_geometric
    from utilities import log, pad_tensor, GraphDataset, Scheduler

    rng = np.random.RandomState(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    if args.model == 'baseline':
        from model.baseline import GNNPolicy
    elif args.model == 'ecbgnn':
        from model.ecbgnn import GNNPolicy

    logfile = os.path.join(running_dir, f'log_{args.model}.txt')
    if os.path.exists(logfile):
        os.remove(logfile)

    log(f"max_epochs: {max_epochs}", logfile)
    log(f"batch_size: {batch_size}", logfile)
    log(f"pretrain_batch_size: {pretrain_batch_size}", logfile)
    log(f"valid_batch_size : {valid_batch_size }", logfile)
    log(f"lr: {lr}", logfile)
    log(f"entropy bonus: {entropy_bonus}", logfile)
    log(f"top_k: {top_k}", logfile)
    log(f"model: {args.model}", logfile)
    log(f"seed {SEED}", logfile)

    log(f"CUDA Available: {torch.cuda.is_available()}", logfile)
    log(f"Number of GPUs: {torch.cuda.device_count()}", logfile)

    policy = GNNPolicy().to(device)
    if torch.cuda.device_count() > 1:
        policy = torch.nn.DataParallel(policy)

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    scheduler = Scheduler(optimizer, mode='min', patience=10, factor=0.2)

    all_files = sorted(str(p) for p in pathlib.Path('data/samples/train/').rglob('*.pkl'))
    if len(all_files) == 0:
        raise RuntimeError('No training files found under data/samples/train/')
    split_idx = int(len(all_files) * 0.9)
    train_files = all_files[:split_idx]
    valid_files = all_files[split_idx:]
    pretrain_files = [f for i, f in enumerate(train_files) if i % 10 == 0]

    pretrain_data = GraphDataset(pretrain_files)
    pretrain_loader = torch_geometric.loader.DataLoader(pretrain_data, pretrain_batch_size, shuffle=False)
    valid_data = GraphDataset(valid_files)
    valid_loader = torch_geometric.loader.DataLoader(valid_data, valid_batch_size, shuffle=False)

    for epoch in range(max_epochs + 1):
        log(f"EPOCH {epoch}...", logfile)
        if epoch == 0:
            n = pretrain(policy, pretrain_loader)
            log(f"PRETRAINED {n} LAYERS", logfile)
        else:
            epoch_train_files = rng.choice(train_files, int(np.floor(min(10000, len(train_files))/batch_size))*batch_size, replace=True)
            train_data = GraphDataset(epoch_train_files)
            train_loader = torch_geometric.loader.DataLoader(train_data, batch_size, shuffle=True)
            train_loss, train_kacc, entropy, train_rep_acc, train_expert_valid_frac = process(policy, train_loader, top_k, optimizer)
            log(f"TRAIN LOSS: {train_loss:0.3f} rep_acc: {train_rep_acc:0.3f} expert_valid_frac: {train_expert_valid_frac:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, train_kacc)]), logfile)

        valid_loss, valid_kacc, entropy, valid_rep_acc, valid_expert_valid_frac = process(policy, valid_loader, top_k, None)
        log(f"VALID LOSS: {valid_loss:0.3f} rep_acc: {valid_rep_acc:0.3f} expert_valid_frac: {valid_expert_valid_frac:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, valid_kacc)]), logfile)

        scheduler.step(valid_loss)
        if scheduler.num_bad_epochs == 0:
            torch.save(policy.state_dict(), pathlib.Path(running_dir)/f'params_{args.model}.pkl')
            log(f"  best model so far", logfile)
        elif scheduler.num_bad_epochs == 10:
            log(f"  10 epochs without improvement, decreasing learning rate", logfile)
        elif scheduler.num_bad_epochs == 20:
            log(f"  20 epochs without improvement, early stopping", logfile)
            break

    weights_path = pathlib.Path(running_dir)/f'params_{args.model}.pkl'
    try:
        state = torch.load(weights_path, map_location=device)
        if isinstance(policy, torch.nn.DataParallel):
            policy.module.load_state_dict(state)
        else:
            policy.load_state_dict(state)
    except TypeError:
        state = torch.load(weights_path, map_location=device)
        if isinstance(policy, torch.nn.DataParallel):
            policy.module.load_state_dict(state)
        else:
            policy.load_state_dict(state)
    valid_loss, valid_kacc, entropy, valid_rep_acc, valid_expert_valid_frac = process(policy, valid_loader, top_k, None)
    log(f"BEST VALID LOSS: {valid_loss:0.3f} rep_acc: {valid_rep_acc:0.3f} expert_valid_frac: {valid_expert_valid_frac:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, valid_kacc)]), logfile)
