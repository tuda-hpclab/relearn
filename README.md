# RELeARN
The connectivity of the brain is constantly changing. Even in the mature brain, new connections between neurons are formed, and existing ones are deleted, a phenomenon called structural plasticity.
Understanding the dynamics of these neuronal networks is crucial to understanding learning, memory, and diseases such as Alzheimer’s.
The Model of Structural Plasticity enables simulation with structural plasticity as described by  Markus Butz-Ostendorf and Arjen van Ooyen in [*A Simple Rule for Dendritic Spine and Axonal Bouton Formation Can Account for Cortical Reorganization after Focal Retinal Lesions*](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003259).
However, with a naive approach to modeling structural plasticity, we need to calculate the probability of forming a synapse between each neuron pair in each plasticity step.
This is unfeasible for a larger number of neurons.
The RELeARN (REwiring of LARge-scale Neural networks) code addresses this challenge via an approximation, reducing the required computations from $\mathcal{O}(n^2)$ to $\mathcal{O}(n \log n)$. Using it, we could conduct simulations with up to one billion neurons. RELeARN is implemented in C++ and parallelized with MPI and OpenMP, making it suitable for almost every HPC system.

## Prerequisites
Using the project requires CMake (>= 4.0 ), a C++ compiler capable of C++20, and MPI.
If the simulation shall run on GPUS, CUDA is also required.

## Installation
Configure and build the project
```shell
git clone https://github.com/tuda-hpclab/relearn.git
cd relearn/relearn
mkdir build && cd build
cmake -DENABLE_CUDA=<CUDA> -DBUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..
make -j
# Run the tests
./bin/relearn_tests
```
Where `<CUDA>` is `0` for a cpu and `1` for a GPU build.

## Run a simulation
You can find all of the CLI arguments in [relearn/README.md](relearn/README.md). 
For a quickstart, you can try one of the commands below:
```shell
# Izhikevich model - CPU only
mpirun -n 2 ./bin/relearn --steps 500000 --algorithm barnes-hut-location-aware --neuron-model izhikevich --plasticity-update-step 100 --num-neurons-per-rank 100 --synapse-conductance 3 --activity-input "combined(synaptic_equally_weighted,fastnormal(5,2,100))" --growth-rate-axon 0.0001 --growth-rate-dendrite-exc 0.0001 --growth-rate-dendrite-inh 0.0001 --fired-status-communicator-type map --no-print-calcium --no-print-fire-steps --no-print-fire-rate --no-print-sums --no-print-overview
# Izhikevich model - GPU only
mpirun -n 2 ./bin/relearn --steps 500000 --algorithm barnes-hut-cuda --neuron-model izhikevich --plasticity-update-step 100 --num-neurons-per-rank 100 --synapse-conductance 3 --activity-input "combined(synaptic_equally_weighted,normal(5,2))" --growth-rate-axon 0.0001 --growth-rate-dendrite-exc 0.0001 --growth-rate-dendrite-inh 0.0001 --fired-status-communicator-type map --no-print-calcium --no-print-fire-steps --no-print-fire-rate --no-print-sums --no-print-overview
```

## Citation
Please cite RELeARN in your publications if it helps your research:
```
@article{rinke2018,
author = {Rinke, Sebastian and Butz-Ostendorf, Markus and Hermanns, Marc-Andr{\'{e}} and Naveau, Mika{\"{e}}l and Wolf, Felix},
title = {A Scalable Algorithm for Simulating the Structural Plasticity of the Brain},
journal = {Journal of Parallel and Distributed Computing},
volume = {120},
year = {2018},
pages = {251--266},
doi = {10.1016/j.jpdc.2017.11.019}
}
```

## Media coverage
The RELeARN simulator was covered in the German TV show [SAT.1 Regionalmagazin für Rheinland-Pfalz und Hessen (July 18, 2019)](https://www.1730live.de/wissenschaftler-wollen-gehirn-nachbauen/).

## Publications
1) Rinke, S., Butz-Ostendorf, M., Hermanns, M.A., Naveau, M., & Wolf, F. (2018). _A Scalable Algorithm for Simulating the Structural Plasticity of the Brain_. Journal of Parallel and Distributed Computing, 120, 251–266. [PDF](https://apps.fz-juelich.de/jsc-pubsystem/aigaion/attachments/rinke_ea.pdf-5c72f91c90128cfe0433a70f61fa4693.pdf)
2) Czappa, F., Geiß, A., & Wolf, F. (2023). _Simulating Structural Plasticity of the Brain more Scalable than Expected_. Journal of Parallel and Distributed Computing, 171, 24–27. [PDF](https://arxiv.org/pdf/2210.05267.pdf)

