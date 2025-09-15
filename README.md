# probabilistic-emissions

This repository tracks code and scripts used to develop CanESM simulations using emission projections sourced from CEPM.


The objective of this project is to demonstrate how CanESM can be used to create a large ensemble (LE) to study the effect of GHG emissions, and demonstrate the scientific benefits of this type of experiment. 

LE are typically used to assess the effect of internal variability. In those single model initial-conditions large ensembles (SMILEs) experiments, the same model is run multiple times with slightly different initial conditions, generating equally likely realizations of the future climate. These experiments are useful to quantify the response to an external forcing and separate it from natural variability.  

Here, we create a new type of LE from a wide array of GHG emission trajectories and a few different initial conditions. That is, instead of driving CanESM with 4 Shared Socioeconomic Pathways (SSPs), we drive it with around 17 different GHGs, and for each one the model is initialized with three different initial conditions. There are two preliminary objectives for this experiment. One is to  identify with greater accuracy emission thresholds leading to specific climate impacts, or tipping points. The other is to demonstrate how this type of experiment could enable probabilistic climate hazards assessments. 

Indeed, many decisions regarding climate change adaptation would benefit from probabilistic climate risks assessments. However, there are gaps at the moment in the state of the science which prevent us from delivering this type of information. One of those gaps is the fact that GHG emission scenarios have no probabilities attached, but the other is that even if they had, they span very crudely the "emission space". By creating a LE that samples much more finely this emission space, we're hoping to enable new approaches for climate risk evaluations. 



References
----------
Variability in historical emissions trends suggests a need for a wide range of global scenarios and regional analyses https://www.nature.com/articles/s43247-020-00045-y
High radiative forcing climate scenario relevance analyzed with a ten-million-member ensemble https://www.nature.com/articles/s41467-024-52437-9
