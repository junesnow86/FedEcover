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

After preparing required packages specified in [requirments.txt](requirements.txt), you can run an experiment with the following command:

```bash
python partial_training.py --method ${1} --model ${2} --dataset ${3} --distribution ${4} --num-clients ${5} --client-select-ratio ${6} --rounds ${7} --epochs ${8} --client-capacity-distribution ${9} --global-lr-decay ${10} --gamma ${11} --data-augmentation ${12} >> logs/{method}-{model}-{dataset}-{distribution}-capacity{capacity}-{num_clients}clients.log
```

This command runs an FL process for one method with specified settings and it produces a log file containing accuracy information of each round. Save the log file in the [logs](logs) directory for future use like extracting results and plotting figures. And we provide log files that we use in our paper experiments in the [logs](logs) directory. For more details about optional configuration arguments, refer to [modules/args_parser.py](modules/args_parser.py).

For example, to run an experiment of our method **FedEcover** on Tiny ImageNet with $\alpha=0.5$ under 100-clients-sampling-20% regime, run:

```bash
python partial_training.py --method fedecover --model resnet --dataset tiny-imagenet --distribution alpha0.5 --num-clients 100 --client-select-ratio 0.2 --rounds 300 --epochs 10 --client-capacity-distribution 0 --global-lr-decay True --gamma 0.9 --data-augmentation True >> logs/fedecover-cnn-cifar100-alpha0.5-capacity0-100clients.log
```

For example, to plot the accuracy comparison figure showed in Fig. 7 (c), you need to run the above command five times by specifying different methods (`fedavg`, `heterofl`, `fedrolex`, `fd`, `fedecover`) each time. Then manually run the extracting script and the plotting script in the repository.

We suggest the same naming way as above for log files because the naming among training, extracting results, and plotting figures codes are coupled in this implementation. We provide an example of extracting results in [extract_accuracy_from_log.py](extract_accuracy_from_log.py) and an example of plotting figures in [plot_accuracy_comparison.py](plot_accuracy_comparison.py). Running [plot_accuracy_comparison.py](plot_accuracy_comparison.py) will also print the mean accuracy and std statistics.

We also provide a plotting script for reproducing Fig. 10. Note that it needs user to manually collect accuracy results from different methods with different alpha values first. We provide the results used for our paper in [results](results) directory as examples.
