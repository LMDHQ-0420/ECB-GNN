import os
import glob
import gzip
import argparse
import pickle
import queue
import shutil
import threading
import numpy as np
import ecole
from collections import namedtuple

class ExploreThenStrongBranch:
    def __init__(self, expert_probability):
        self.expert_probability = expert_probability
        self.pseudocosts_function = ecole.observation.Pseudocosts()
        self.strong_branching_function = ecole.observation.StrongBranchingScores()

    def before_reset(self, model):
        self.pseudocosts_function.before_reset(model)
        self.strong_branching_function.before_reset(model)

    def extract(self, model, done):
        probabilities = [1-self.expert_probability, self.expert_probability]
        expert_chosen = bool(np.random.choice(np.arange(2), p=probabilities))
        if expert_chosen:
            return (self.strong_branching_function.extract(model,done), True)
        else:
            return (self.pseudocosts_function.extract(model,done), False)
            

def send_orders(orders_queue, instances, seed, query_expert_prob, time_limit, out_dir, stop_flag):

    rng = np.random.RandomState(seed)

    episode = 0
    while not stop_flag.is_set():
        instance = rng.choice(instances)
        seed = rng.randint(2**32)
        orders_queue.put([episode, instance, seed, query_expert_prob, time_limit, out_dir])
        episode += 1
        

def make_samples(in_queue, out_queue, stop_flag):

    sample_counter = 0
    while not stop_flag.is_set():
        episode, instance, seed, query_expert_prob, time_limit, out_dir = in_queue.get()

        scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0,
                           'limits/time': time_limit, 'timing/clocktype': 2}
        observation_function = { "scores": ExploreThenStrongBranch(expert_probability=query_expert_prob),
                                 "node_observation": ecole.observation.NodeBipartite() }
        env = ecole.environment.Branching(observation_function=observation_function,
                                          scip_params=scip_parameters, pseudo_candidates=True)

        print(f"[w {threading.current_thread().name}] episode {episode}, seed {seed}, "
              f"processing instance '{instance}'...\n", end='')
        out_queue.put({
            'type': 'start',
            'episode': episode,
            'instance': instance,
            'seed': seed,
        })

        env.seed(seed)
        observation, action_set, _, done, _ = env.reset(instance)
        while not done:
            scores, scores_are_expert = observation["scores"]
            # sanitize scores: replace NaN with sentinel used in dataset preprocessing
            scores = np.nan_to_num(scores, nan=-1e8)

            node_observation = observation["node_observation"]
            # sanitize numeric observation arrays (row features, edge values, variable features)
            row_feats = np.nan_to_num(node_observation.row_features.astype(np.float32), nan=-1e8)
            edge_idx = node_observation.edge_features.indices.astype(np.int64)
            edge_vals = np.nan_to_num(node_observation.edge_features.values.astype(np.float32), nan=-1e8)
            var_feats = np.nan_to_num(node_observation.variable_features.astype(np.float32), nan=-1e8)
            node_observation = (row_feats, (edge_idx, edge_vals), var_feats)

            # select action using sanitized scores
            action = action_set[scores[action_set].argmax()]

            if scores_are_expert and not stop_flag.is_set():
                data = [node_observation, action, action_set, scores]
                filename = f'{out_dir}/sample_{episode}_{sample_counter}.pkl'

                with gzip.open(filename, 'wb') as f:
                    pickle.dump({
                        'episode': episode,
                        'instance': instance,
                        'seed': seed,
                        'data': data,
                    }, f)
                out_queue.put({
                    'type': 'sample',
                    'episode': episode,
                    'instance': instance,
                    'seed': seed,
                    'filename': filename,
                })
                sample_counter += 1

            try:
                observation, action_set, _, done, _ = env.step(action)
            except Exception as e:
                done = True
                with open("error_log.txt","a") as f:
                    f.write(f"Error occurred solving {instance} with seed {seed}\n")
                    f.write(f"{e}\n")

        print(f"[w {threading.current_thread().name}] episode {episode} done, {sample_counter} samples\n", end='')
        out_queue.put({
            'type': 'done',
            'episode': episode,
            'instance': instance,
            'seed': seed,
        })


def collect_samples(instances, out_dir, rng, n_samples, n_jobs,
                    query_expert_prob, time_limit):

    os.makedirs(out_dir, exist_ok=True)

    orders_queue = queue.Queue(maxsize=2*n_jobs)
    answers_queue = queue.SimpleQueue()

    tmp_samples_dir = f'{out_dir}/tmp'
    os.makedirs(tmp_samples_dir, exist_ok=True)

    dispatcher_stop_flag = threading.Event()
    dispatcher = threading.Thread(
            target=send_orders,
            args=(orders_queue, instances, rng.randint(2**32), query_expert_prob,
                  time_limit, tmp_samples_dir, dispatcher_stop_flag),
            daemon=True)
    dispatcher.start()

    workers = []
    workers_stop_flag = threading.Event()
    for i in range(n_jobs):
        p = threading.Thread(
                target=make_samples,
                args=(orders_queue, answers_queue, workers_stop_flag),
                daemon=True)
        workers.append(p)
        p.start()

    buffer = {}
    current_episode = 0
    i = 0
    in_buffer = 0
    while i < n_samples:
        sample = answers_queue.get()

        # add received sample to buffer
        if sample['type'] == 'start':
            buffer[sample['episode']] = []
        else:
            buffer[sample['episode']].append(sample)
            if sample['type'] == 'sample':
                in_buffer += 1

        while current_episode in buffer and buffer[current_episode]:
            samples_to_write = buffer[current_episode]
            buffer[current_episode] = []

            for sample in samples_to_write:

                if sample['type'] == 'done':
                    del buffer[current_episode]
                    current_episode += 1

                else:
                    os.rename(sample['filename'], f'{out_dir}/sample_{i+1}.pkl')
                    in_buffer -= 1
                    i += 1
                    print(f"[m {threading.current_thread().name}] {i} / {n_samples} samples written, "
                          f"ep {sample['episode']} ({in_buffer} in buffer).\n", end='')

                    if in_buffer + i >= n_samples and dispatcher.is_alive():
                        dispatcher_stop_flag.set()
                        print(f"[m {threading.current_thread().name}] dispatcher stopped...\n", end='')

                    if i == n_samples:
                        buffer = {}
                        break

    workers_stop_flag.set()
    for p in workers:
        p.join()

    print(f"Done collecting samples for {out_dir}")
    shutil.rmtree(tmp_samples_dir, ignore_errors=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a sampling data set.")
    parser.add_argument("--mode", choices=["train", "valid", "test"], required=True, help="Please select to generate the training set, test set or validation set.")
    parser.add_argument("--njobs", type=int, default=12, help="Number of parallel threads.")
    args = parser.parse_args()

    SEED = 42
    train_size = 50000
    valid_size = 10000
    test_size = 5000
    node_record_prob = 0.05
    time_limit = 3600

    train_dir = "data/instances/train"
    valid_dir = "data/instances/valid"
    test_dir = "data/instances/test"
    out_dir = "data/samples"

    os.makedirs(out_dir, exist_ok=True)

    if args.mode == "train":
        instances = glob.glob(f"{train_dir}/*.lp")
        rng = np.random.RandomState(SEED)
        collect_samples(instances, out_dir + '/train', rng, train_size,
                        args.njobs, query_expert_prob=node_record_prob,
                        time_limit=time_limit)
    elif args.mode == "valid":
        instances = glob.glob(f"{valid_dir}/*.lp")
        rng = np.random.RandomState(SEED + 1)
        collect_samples(instances, out_dir + '/valid', rng, valid_size,
                        args.njobs, query_expert_prob=node_record_prob,
                        time_limit=time_limit)
    elif args.mode == "test":
        test_difficulties = ["hard", "middle", "simple"]
        test_subdirs = ["Core", "StructTransfer", "ParamRobustness"]
        for diff in test_difficulties:
            for idx, sub in enumerate(test_subdirs):
                test_subdir = os.path.join(test_dir, sub, diff)
                out_test_subdir = os.path.join(out_dir, 'test', sub, diff)
                instances_test_sub = glob.glob(f"{test_subdir}/**/*.lp", recursive=True)
                rng = np.random.RandomState(SEED + 100 * idx + 10 * test_difficulties.index(diff))
                os.makedirs(out_test_subdir, exist_ok=True)
                collect_samples(instances_test_sub, out_test_subdir, rng, test_size,
                                args.njobs, query_expert_prob=node_record_prob,
                                time_limit=time_limit)