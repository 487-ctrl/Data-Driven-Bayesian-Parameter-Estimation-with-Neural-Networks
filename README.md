# Data-Driven Bayesian Parameter Estimation with Neural Networks

This Repository consists of the main document of my Bachelor's Thesis called "Data-driven Bayesian Parameter Estimation with Neural Networks", along with the Code needed to reproduce my findings.

## Acknowledgements

- [BWUniCluster 2.0](https://wiki.bwhpc.de/e/BwUniCluster2.0)

## experiment plans

### Synthetic-Data

- increasing number of simulations used
  - might correlate with the use of better prior choices
  - if the prior is closer to reality, it should not change much with higher simulation numbers
  - if the prior doesnt make many assumptions on the parameter distribution, more simulated data might be needed
  - this obviously is reasoned by the decision process, on when we accept a simulation or not
    - the more assumptive the prior is the more simulations will be accepted and we need less simulations to obtain the same results as:
    - the more simulations we use the less assumptive the prior has to be, but we need more simulations and more computation time
    - this rises the question, how can i improve the decision process on which simulations are accepted, since this is essential to balance computation time and model accuracy
    - if so, then finding prior assumptions is the key to more efficient training and parameter estimation
    - (general context of simualtion based inference)
- increasing the timespan of simulations
  - how can i decide which simulation to accept at great length of simulations
  - (no further ideas by now)
- influence of different priors
  - as said earlier, this might correlate with the amount of simulations used to fit the model
- methods used to train the density estimator
  - this basically is what i am asking myself rightnow
- influence of noise or bias in the prior
  - by removing bad simulations the influence of errors in the prior wont falsify the posterior

### Empirical-Data

- this should be more topic specific and results in real data experiments

### Questions asked

- what exactly are steps that should be taken when the trained posterior doesn't differ from the prior
- what are the possible reasons why the previous phenomena happend (not enough training? found the true parameters? -> this possible assumption doesn't always hold)
- 