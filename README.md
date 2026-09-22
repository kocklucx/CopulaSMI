# Bayesian Modular Inference for Copula Models with Potentially Misspecified Marginals
Copula models of multivariate data are popular because they allow separate specification
of marginal distributions and the copula function. These components can be treated as
inter-related modules in a modified Bayesian inference approach called “cutting feedback”
that is robust to their misspecification. Recent work uses a two module approach, where all
d marginals form a single module, to robustify inference for the marginals against copula
function misspecification, or vice versa. However, marginals can exhibit differing levels of
misspecification, and it is attractive to assign each its own module with an individual influence parameter controlling its contribution 
to a joint semi-modular inference (SMI) posterior. This generalizes existing two module SMI methods, which interpolate between cut
and conventional posteriors using a single influence parameter. We develop a novel copula
SMI method and select the influence parameters using Bayesian optimization. It provides
an efficient continuous relaxation of the discrete optimization problem over 2d cut/uncut
configurations. We establish theoretical properties of the resulting semi-modular posterior
and demonstrate the approach on simulated and real data. The real data application uses
a skew-normal copula model of asymmetric dependence between equity volatility and bond
yields, where robustifying copula estimation against marginal misspecification is strongly
motivated.

## Reproduction of results from the paper

The code has been tested with Python 3.12.2.

a) copula_smi.py: Contains code to train a Gaussian variational approximation to the SMI objective defined in the paper. The function takes an influence parameter gamma, the observed data, and functions to evaluate the statistical base model (copula and marginals) as input. Further documentation can be found in the file.

b) minimal_working_example.py: Contains a minimal working example on a single simulated dataset. It is recommended that you familiarize yourself with this example first.

c) Simulation study: simulation_study.py contains the code necessary to reproduce the simulation study and generate the figures presented in the manuscript.

d) highdim_simulations.py: contains the code to reproduce the additional high-dimensional simulation study presented in the Supporting Information. The simulated data set is provided as data_highdimensional.p.

e) Financial application: financial_application.py contains code to train the fully cut, conventional posterior, and optimal SMI models on the financial data. Plots are generated with financial_application_plots.py, which can be run after financial_application.py has run successfully. The preprocessed data are provided in yields.csv.

## Citation

If you use this work, please cite

```bibtex
@article{KocFraSmiNot2026,
  title={Bayesian Modular Inference for Copula Models with Potentially Misspecified Marginals},
  author={Kock, Lucas and Frazier, David T and Smith, Michael Stanley and Nott, David J},
  journal={arXiv preprint arXiv:2603.11457},
  year={2026}
}
