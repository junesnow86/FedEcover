# FedEcover

[PyTorch](https://pytorch.org/) experimental implementation for paper **FedEcover: Fast and Stable Converging Model-Heterogeneous Federated Learning with Efficient-Coverage Submodel Extraction** (Accepted by ICDE 2025).

## Download Datasets

### CIFAR-10 and CIFAR-100

Download the datasets by specifying `download=True` when instantiating the `datasets.CIFAR-10` and `datasets.CIFAR-100` objects from torchvision.

### Tiny ImageNet

Enter the [data](data) directory of this repository, and execute the following commands to download the Tiny ImageNet dataset.

```bash
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
```

### FEMNIST

Generate data with [the official code of LEAF](https://github.com/TalwalkarLab/leaf/tree/master/data/femnist) and utilize `scripts/data_processing/extract_femnist_data.py` to extract generated data.

## Reproduce Experimental Results

After preparing required packages specified in [requirments.txt](requirements.txt), you can run an experiment with the following command:

```bash
python main.py --method ${1} --model ${2} --dataset ${3} --distribution ${4} --num-clients ${5} --client-select-mode ${6} --client-select-ratio ${7} --rounds ${8} --epochs ${9} --client-capacity-distribution ${10} --global-lr-decay ${11} --gamma ${12} --data-augmentation ${13} >> logs/{method}-{model}-{dataset}-{distribution}-capacity{capacity}-{num_clients}clients.log
```

This command runs an FL process for one method with specified settings and it produces a log file containing accuracy information of each round. Save the log file in the [logs](logs) directory for future use like extracting results and plotting figures. And we provide log files that we use in our paper experiments in the [logs](logs) directory. For more details about optional configuration arguments, refer to [modules/args_parser.py](modules/args_parser.py).

For example, to run an experiment of our method **FedEcover** on Tiny ImageNet with $\alpha=0.5$ under 100-clients-sampling-20% regime, run:

```bash
python main.py --method fedecover --model resnet --dataset tiny-imagenet --distribution alpha0.5 --num-clients 100 --client-selection-mode ratio --client-select-ratio 0.2 --rounds 300 --epochs 10 --client-capacity-distribution 0 --global-lr-decay True --gamma 0.9 --data-augmentation True >> logs/fedecover-cnn-cifar100-alpha0.5-capacity0-100clients.log
```

For example, to plot the accuracy comparison figure showed in Fig. 7 (c), you need to run the above command five times by specifying different methods (`fedavg`, `heterofl`, `fedrolex`, `fd`, `fedecover`) each time. Then manually run the extracting script and the plotting script in the repository.

We suggest the same naming way as above for log files because the naming among training, extracting results, and plotting figures codes are coupled in this implementation. We provide an example of extracting results with [extract_accuracy_from_log.py](scripts/visualization/extract_accuracy_from_log.py) and an example of plotting figures with [plot_accuracy_comparison.py](scripts/visualization/plot_accuracy_comparison.py). 

We also provide:

- [plot_different_alpha.py](scripts/visualization/plot_different_alpha.py) for reproducing Fig. 9. Note that it needs user to manually collect accuracy results from different methods with different alpha values first. We provide the results used for our paper in [results/different-alpha](results/different-alpha/) directory as examples.
- [plot_param_sensitivity_gamma.py](scripts/visualization/plot_param_sensitivity_gamma.py) for reproducing the results in hyperparameter analysis using the results in [results/param-sensitivity](results/param-sensitivity).
