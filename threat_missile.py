import pandas as pd
import numpy as np

class ThreatMissile():
    def __init__(self, location, target_location, target_name, config):
        # Save location attributes
        self.location = np.array(location).astype(np.float64)
        self.target_location = np.array(target_location).astype(np.float64)
        self.target_name = target_name

        # Save config
        self.config = config

        # Set speed
        self.location_step = (self.target_location - self.location) / float(self.config['steps_to_impact'])

        # Initialize states
        self.aquired = False

    def step(self):
        self.location += self.location_step
        return self.location

    def aquire(self):
        self.aquired = True