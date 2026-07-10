# LIP_Internships26_SNDATLAS
## Project description
 - Identify proton-proton collisions in the ATLAS detector where a neutrino is produced in the direction of SND@LHC.
 - Tasks:
   - Train neural network to discriminate between signal and background. A simple architecture is currently implemented. Possible project direction: implement more sophisticated architectures, such as the [Particle Flow Network](https://energyflow.network/docs/archs/#pfn).
   - Compute new variables to use in the neural network input
     - Event-level:
       - Sphericity
       - Thrust
     - Particle-level:
       - Impact parameter

## File description
 - `to_df.ipynb`: converts input files into pandas dataframe. New variables can be computed here.
 - `Template_ML.ipynb`: loads pandas data and trains neural network.
 - `event_shapes.py`: example implementation of event-level variables.