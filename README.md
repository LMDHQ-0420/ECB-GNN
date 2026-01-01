# ECB-GNN: Edge conditioned bipartite graph neural networks for accelerating mixed integer programming in isolated microgrid scheduling

## Install

```bash
conda create -n ecb-gnn python=3.10.19 -y
conda activate ecb-gnn

pip install -r requirements.txt
```

## Workflow

1) **Generate LP instances**
```bash
python 01_generate_instances.py --mode train
python 01_generate_instances.py --mode valid
python 01_generate_instances.py --mode test
```

2) **Collect samples**
```bash
python 02_generate_dataset.py --mode train  --njobs 12
python 02_generate_dataset.py --mode valid  --njobs 12
python 02_generate_dataset.py --mode test   --njobs 12
```

3) **Train**
- GNN: `baseline` or `ecbgnn`
- ML baselines: `extratrees`, `lambdamart`, `svmrank`
```bash
python 03_train.py --model baseline
python 03_train.py --model ecbgnn
python 03_train.py --model extratrees
python 03_train.py --model lambdamart
python 03_train.py --model svmrank
```
Outputs go to `runs/<model>/` (weights, normalization, feature specs, logs).

4) **Evaluate on instances**
```bash
python 04_evaluate_with_instances.py --model baseline --gpu 0 --gap_limit 2
python 04_evaluate_with_instances.py --model ecbgnn --gpu 0 --gap_limit 2
python 04_evaluate_with_instances.py --model extratrees --gap_limit 2
python 04_evaluate_with_instances.py --model lambdamart --gap_limit 2
python 04_evaluate_with_instances.py --model svmrank --gap_limit 2
```
CSV results are saved under `results/gap_limit_<X>/<category>/<difficulty>/`.

5) **Evaluate on sampled batches**
```bash
python 05_evaluate_with_samples.py --model baseline
python 05_evaluate_with_samples.py --model ecbgnn
python 05_evaluate_with_samples.py --model extratrees
python 05_evaluate_with_samples.py --model lambdamart
python 05_evaluate_with_samples.py --model svmrank
```
Results go to `results/samples_evaluation/`.

