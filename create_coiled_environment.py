#!/usr/bin/env python3

############################################################
#        CREATE A COILED ENVIRONMENT FOR THE PROJECT       #
############################################################

import coiled

coiled.create_software_environment(
    name="lidar-classification",
    conda='environment.yml'
)

