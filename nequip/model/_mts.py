import torch

from nequip.nn import GraphModuleMixin, MTSEnergySum
from nequip.model import model_from_config
from nequip.utils import Config


def AllegroMTS(
    config: Config, initialize: bool, deploy: bool, dataset=None
) -> GraphModuleMixin:
    configs = {k: Config(config) for k in ("inner", "outer")}
    for prefix, c in configs.items():
        c.update(c.pop(f"{prefix}_model"))
    models = {
        prefix: model_from_config(
            config=c, initialize=initialize, dataset=dataset, deploy=deploy
        ).model
        for prefix, c in configs.items()
    }
    # Update the original config (that gets saved) with the actual configs operated
    # on by the model builders of the submodels (important, e.g., for avg_num_neighbors)
    for prefix, c in configs.items():
        config[f"{prefix}_model"] = Config.as_dict(c)

    if initialize:
        with torch.no_grad():
            # we don't want the outer model to contribute at init
            # this is specific to Allegro
            models["outer"].edge_eng._module._forward._weight_0.fill_(0.0)

    return MTSEnergySum(inner_model=models["inner"], outer_model=models["outer"])


# Usage example
r"""
# !! PLEASE NOTE: `minimal.yaml` is meant as a _minimal_ example of a tiny, fast
#                 training that can be used to verify your nequip+allegro install,
#                 the syntax of your configuration edits, etc.
#                 These are NOT recommended hyperparameters for real applications!
#                 Please see `example.yaml` for a reasonable starting point.

# general
root: results/aspirin
run_name: minimal-mts
seed: 123456
dataset_seed: 123456

# -- network --
model_builders:
 - nequip.model.AllegroMTS
 - StressForceOutput
 - RescaleEnergyEtc

# cutoffs
r_max: 6.0
# network symmetry
l_max: 1
parity: o3_full

# allegro layers:
# can specify common options for the inner and outer models here:
two_body_latent_mlp_latent_dimensions: [32, 64]
two_body_latent_mlp_nonlinearity: silu
num_bessels_per_basis: 8

latent_mlp_latent_dimensions: [64]
latent_mlp_nonlinearity: silu

env_embed_mlp_latent_dimensions: []
env_embed_mlp_nonlinearity: null

edge_eng_mlp_latent_dimensions: [32]
edge_eng_mlp_nonlinearity: null

# can then specify particular keys here, which should override common keys:
inner_model:
  model_builders:
    - allegro.model.Allegro
    - PerSpeciesRescale
  r_max: 4.0
  num_tensor_features: 8
  num_layers: 1

# same for the outer model:
outer_model:
  model_builders:
    - allegro.model.Allegro
    - PerSpeciesRescale
  per_species_rescale_shifts: null  # only want one model applying shifts
  r_max: 6.0
  num_tensor_features: 16
  num_layers: 2


# -- data --
dataset: npz                                                                       # type of data set, can be npz or ase
dataset_url: http://quantum-machine.org/gdml/data/npz/aspirin_ccsd.zip             # url to download the npz. optional
dataset_file_name: ./benchmark_data/aspirin_ccsd-train.npz                         # path to data set file
key_mapping:
  z: atomic_numbers                                                                # atomic species, integers
  E: total_energy                                                                  # total potential eneriges to train to
  F: forces                                                                        # atomic forces to train to
  R: pos                                                                           # raw atomic positions
npz_fixed_field_keys:                                                              # fields that are repeated across different examples
  - atomic_numbers

# A mapping of chemical species to type indexes is necessary if the dataset is provided with atomic numbers instead of type indexes.
chemical_symbol_to_type:
  H: 0
  C: 1
  O: 2

# logging
wandb: false
verbose: debug

# training
n_train: 25
n_val: 5
batch_size: 1
max_epochs: 10
learning_rate: 0.002

# loss function
loss_coeffs: forces

# optimizer
optimizer_name: Adam

"""

# Deploy example
"""
# After training a joint model with minimal-mts.yaml, use this file to
# deploy the inner/outer model from that for separate use in LAMMPS
# (or elsewhere)
# For example (with the file as is)
#   $ nequip-deploy build --model configs/minimal_mts_deploy.yaml inner.pth
# Or, for the outer model edit at the end and
#   $ nequip-deploy build --model configs/minimal_mts_deploy.yaml outer.pth

# This special key loads ALL KEYS from the specified YAML file
# We use it here to load all the hyperparameters, etc. of the model we're trying to deploy from its training dir
# This guerantees they can't get out of sync/disagree
include_file_as_baseline_config: results/aspirin/minimal-mts/config.yaml

# repeat the model builders but add a load_model_state to load the trained weights
model_builders:
 # from minimal-mts.yaml
 - nequip.model.AllegroMTS
 - StressForceOutput
 - RescaleEnergyEtc
 # !! NEW !!
 - load_model_state

load_model_state: results/aspirin/minimal-mts/best_model.pth

# but only use the inner or outer model for deploy
use_inner: true
use_outer: false
"""
