import numpy as np

def get_encoding_config(encoding_type, resolution, bound, n_bands=4, n_levels=16, base_resolution=16):
    '''## Given encoding_type, format and return corresponding config dict for tcnn back-end'''
    config = {
        "otype": encoding_type
    }

    if encoding_type == "HashGrid":
        per_level_scale = np.exp2(
                np.log2(resolution * bound / base_resolution) / (n_levels - 1)
                )
        config.update({"n_levels": n_levels,
                       "n_features_per_level": 2,
                       "log2_hashmap_size": 19,
                       "base_resolution": base_resolution,
                       "per_level_scale": per_level_scale,
                       })
    elif encoding_type == "Frequency":
        config.update({
            "degree": n_bands
        })
    elif encoding_type == "SphericalHarmonics":
        config.update({
            "degree": n_bands
        })
    else:
        raise RuntimeError(f"ERROR: {encoding_type} is not a valid choice for\
                           TCNN encoding type. Valid options are: [HashGrid, \
                           Frequency, SphericalHarmonics]")

    return config

def get_mlp_config(hidden_dim, n_hidden_layers):
    '''
    ## Given hidden layer size and the # of hidden layers, format and\
    return corresponding Fully-Fused MLP config dict for tcnn back-end
    '''
    config = {"otype": "FullyFusedMLP",
              "activation": "ReLU",
              "output_activation": "None",
              "n_neurons": hidden_dim,
              "n_hidden_layers": n_hidden_layers - 1
              }
    return config