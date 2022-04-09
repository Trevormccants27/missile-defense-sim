import pandas as pd
import numpy as np
import geopy.distance as dist

from threat_missile import ThreatMissile

class ThreatPlatform():
    def __init__(self, location, config, seed=0):
        # Save attributes
        self.location = location

        # Initialize random seed
        #np.random.seed(seed)

        # Save config
        self.config = config
        self.n_missiles = np.random.randint(self.config['n_missiles'][0], self.config['n_missiles'][1])

        # Initialize time to next attack
        self.time_to_next_attack = None
        self.set_time_to_next_attack()
        
    def step(self, defended_assets):
        # Return None if no missiles were launched
        result = None

        # Launch a new threat missile if it's time to attack
        if self.time_to_next_attack <= 0 and self.n_missiles > 0:
            target_idx = np.random.choice(defended_assets.index, p=defended_assets['population']/np.sum(defended_assets['population']))
            target_loc = (defended_assets.loc[target_idx]['lat'], defended_assets.loc[target_idx]['lng'])
            target_name = defended_assets.loc[target_idx]['city']
            result = ThreatMissile(self.location, target_loc, target_name, self.config['threat_missile'])

            # Decrement number of missilles
            self.n_missiles -= 1

        if result is not None:
            # Reset time to next attack
            self.set_time_to_next_attack()
        else:
            # Decrement time to attack
            self.time_to_next_attack -= 1

        return result

    def set_time_to_next_attack(self, time=None):
        self.time_to_next_attack = np.random.normal(self.config['time_between_attacks']['mean'], self.config['time_between_attacks']['std'])