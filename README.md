# FedEcover

Experimental implementation for paper **FedEcover: Fast and Stable Converging Model-Heterogeneous Federated Learning with Efficient-Coverage Submodel Extraction** using [Pytorch](https://pytorch.org/).

## Download Datasets

### CIFAR-10 and CIFAR-100

Download the datasets by specifying `download=True` when instantiating the `datasets.CIFAR-10` and `datasets.CIFAR-100` objects from torchvision.

### Tiny ImageNet

Enter the [data](data) directory of this repository, and execute the following commands to download the Tiny ImageNet dataset.

```bash
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
```

## Reproduce the Experimental Results

After preparing required packages specified in [requirments.txt](requirements.txt), run an experiment with the following command:

```bash
python partial_training.py --method ${1} --model ${2} --dataset ${3} --distribution ${4} --num-clients ${5} --client-select-ratio ${6} --rounds ${7} --epochs ${8} --client-capacity-distribution ${9} --global-lr-decay ${10} --gamma ${11} --data-augmentation ${12} >> logs/{method}-{model}-{dataset}-{distribution}-capacity{capacity}-{num_clients}clients.log
```

This command runs a FL training process for one single method with specified settings and it produces a log file containing accuracy information of each round. Save the log file in the [logs](logs) directory for future use like extracting results and plotting figures. For more details about optional configuration arguments, refer to [modules/args_parser.py](modules/args_parser.py).

For example, to run an experiment of our method **FedEcover** on CIFAR-100 with $\alpha=0.5$ under 100-clients-sampling-20% regime, run:

```bash
python partial_training.py --method fedecover --model cnn --dataset cifar100 --distribution alpha0.5 --num-clients 100 --client-select-ratio 0.2 --rounds 300 --epochs 10 --client-capacity-distribution 0 --global-lr-decay True --gamma 0.9 --data-augmentation True >> logs/fedecover-cnn-cifar100-alpha0.5-capacity0-100clients.log
```

To run different methods, e.g., FedAvg, FedRolex, etc, you need to specify the `method` argument with valid choices, see in [modules/args_parser.py](modules/args_parser.py).

We suggest the same naming way as above for log files because the naming among training, extracting results, and plotting figures codes are coupled in this implementation. We provide an example of extracting results in [extract_accuracy_from_log.py](extract_accuracy_from_log.py) and an example of plotting figures in [plot_accuracy_comparison.py](plot_accuracy_comparison.py).
