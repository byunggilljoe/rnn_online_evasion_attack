# Codes for "Online Evasion Attacks on Recurrent Models: The Power of Hallucinating the Future"
This is a repository of Codes for IJCAI 2022 paper, "Online Evasion Attacks on Recurrent Models: The Power of Hallucinating the Future"

# Base Codes
This repository includes the following repositories.
- MIMIC-III: https://github.com/YerevaNN/mimic3-benchmarks
- Udacity Driving Predictive Model: https://github.com/gnosisyuw/CrevNet-Traffic4cast.git
- Udacity Driving Victim Model: https://github.com/ITSEG-MQ/Adv-attack-and-defense-on-driving-model.git
# Experiments
- You can find scripts to execute experiments in "experiments/"
- We demonstrate a sequence of script execution for an MNIST experiment. 
- You can conduct experiments for ohter datasets similarly.
- You can skip Step 0 and Step 1, if you just want to conduct our attack. (MNIST, FashionMNIST, energy and user datasets)
# Step 0. Training a Victim RNN $f_\theta, g_\theta$
```bash
bash 0_MNIST-train-victim.sh
```
# Step 1. Training a Predictor $Q_\phi$
```bash
bash 1_MNIST-train-predictor.sh
```
# Step 2. Attacking the Victim RNN
```bash
bash 2_MNIST-attack.sh <clairvoyant/greedy/predictive/IID_predictive> <lookahead> <GPU_NUM>
```
- The first argument specifies attack approach. It should be one of clairvoyant, greedy, predicitve and IID_predictive.
- The second argument specifies lookahead.
    - MNIST/FashionMNIST:1-28
    - Mortality: 1-48
    - User: 1-50
    - Udacity: 1-20
    - Energy: 1-50
    

- The third agument specifies the GPU (If you have one GPU, it is 0).
# Dependency
Please refer to DEPENDENCY.txt. It is a result of `conda list`.

# Data Aquisition
We do not include MIMIC-III and Udacity Driving dataset.

- MIMIC-III dataset can be aquired after submitting a certificate.
Please refer [__Requirement__](https://github.com/YerevaNN/mimic3-benchmarks#Requirements) section and following data processing instructions in the original repository.
    - The data should be located in "mimic3_mnist_sentiment/data/in-hospital-mortality".
    - The directory contains
        - test
        - test_listfile.csv
        - train
        - train_listfile.csv
        - val_listfile.csv
- Udacity data should be donwloaded from https://github.com/udacity/self-driving-car.
    - The data should be located in "udacity-data".



# Contacts
If you have problem, please contact byunggill.joe@gmail.com or create an issue.
