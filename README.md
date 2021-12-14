# Repository with an example case of Genetic Algorithm and Particle Swarm Optimization for deep-model optimization

### If you use this code, please cite us: _Multiple Time Series Fusion Based on LSTM: An Application to CAP A Phase Classification Using EEG_ (submitted) ### 
**Authors:** Fábio Mendonça, Sheikh Shanawaz Mostafa, Diogo Freitas, Fernando Morgado-Dias, and Antonio G. Ravelo-García

## Abstract

Biomedical decision making involves multiple signal processing, either from different sensors or from different channels. In both cases, information fusion plays a significant role. A deep learning based electroencephalogram channels’ feature level  fusion  is carried out in this work for the electroencephalogram cyclic alternating pattern A phase classification. Channel selection, fusion, and classification procedures were optimized by two optimization algorithms, namely, Genetic Algorithm and Particle Swarm Optimization. The developed methodologies were evaluated by fusing the information from multiple electroencephalogram channels for patients with nocturnal frontal lobe epilepsy and patients without any neurological disorder, which was significantly more challenging when compared to other state of the art works. Results showed that both optimization algorithms selected a comparable structure with similar feature level fusion, consisting of three electroencephalogram channels, which is in line with the CAP protocol to ensure multiple channels’ arousals for CAP detection. Moreover, the two optimized models reached an area under the receiver operating characteristic curve of 0.82, with average accuracy ranging from 77% to 79%, a result which is in the upper range of the specialist agreement. The proposed approach is still in the upper range of the best state of the art works despite a difficult dataset, and  has the advantage of providing  a fully automatic analysis without requiring any manual procedure. Ultimately, the models revealed to be noise resistant and resilient to multiple channel loss.

## About the code

In this repository, you will find several Python scripts. Among them, 
- `ga.py` script that runs the Genetic Algorithm (GA) according to the chromosome codification defined by the user.
- `pso.py` script that runs the Particle Swarm Optimization (PSO) according to the particle codification defined by the user.

## Notes:

1. To run the algorithm, load the matrices in the root. The CAP sleep databases can be downloaded [here](https://physionet.org/content/capslpdb/1.0.0/).
2. The file `read_pickle.py` is just an additional script for reading pickle files.
3. The file `LOOTest.py` is used to test the model with leave-one-out.
