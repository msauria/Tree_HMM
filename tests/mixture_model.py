#!/usr/bin/env python

import sys
import multiprocessing

import numpy
import Tree_HMM

def main():
    RNG = numpy.random.default_rng(seed=20106)
    # define 2 distributions
    fair1 = Tree_HMM.EmissionGaussianDistribution(mu=-2.0, sigma=0.5, label="F1")
    fair2 = Tree_HMM.EmissionGaussianDistribution(mu=2.0, sigma=0.5, label="F2")
    unfair1 = Tree_HMM.EmissionGaussianDistribution(mu=5.0, sigma=1.25, label="UF1")
    unfair2 = Tree_HMM.EmissionGaussianDistribution(mu=12.0, sigma=2., label="UF2")
    fair = Tree_HMM.EmissionContinuousMixtureDistribution(
        distributions=[fair1, fair2], proportions=[0.33, 0.67], label="F")
    unfair = Tree_HMM.EmissionContinuousMixtureDistribution(
        distributions=[unfair1, unfair2], proportions=[0.75, 0.25], label="UF")

    tree = {
        "A": None,
        "AA": "A",
        "AB": "A",
        "AAA": "AA",
        "AAB": "AA",
        "ABA": "AB",
    }

    # define transitions
    TM = numpy.array([[0.9, 0.1], [0.05, 0.95]], numpy.float64)
    TMb = numpy.array([[0.8, 0.2], [0.1, 0.9]], numpy.float64)
    # define initial probabilities
    IP = numpy.array([0.4, 0.6], numpy.float64)
    IPb = numpy.array([0.5, 0.5], numpy.float64)
    # define model
    with Tree_HMM.Tree(tree, emissions=[[fair], [unfair]], state_names=["Fair", "Unfair"],
                        transition_matrix=TM, initial_probabilities=IP,
                        RNG=RNG, nthreads=8) as model:
        model.save('temp_model.npz')

    with Tree_HMM.Tree(fname='temp_model.npz') as model:
        print(model)
        obs, states, node_names = model.generate_sequences(num_seqs=10000)

    fair1 = Tree_HMM.EmissionGaussianDistribution(mu=-1.0, sigma=2.0, label="F1")
    fair2 = Tree_HMM.EmissionGaussianDistribution(mu=1.0, sigma=2.0, label="F2")
    unfair1 = Tree_HMM.EmissionGaussianDistribution(mu=7.0, sigma=1.25, label="UF1")
    unfair2 = Tree_HMM.EmissionGaussianDistribution(mu=9.0, sigma=1.25, label="UF2")
    fairb = Tree_HMM.EmissionContinuousMixtureDistribution(
        distributions=[fair1, fair2], proportions=[0.5, 0.5], label="F")
    unfairb = Tree_HMM.EmissionContinuousMixtureDistribution(
        distributions=[unfair1, unfair2], proportions=[0.25, 0.75], label="UF")

    with Tree_HMM.Tree(tree, emissions=[[fairb], [unfairb]], state_names=["Fair", "Unfair"],
                        transition_matrix=TMb, initial_probabilities=IPb,
                        RNG=RNG, nthreads=8) as model:
        model.train_model(obs, node_names, iterations=100)
        print(model)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
