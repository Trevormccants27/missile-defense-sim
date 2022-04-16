import numpy as np
from scipy.stats import qmc

class actor():
    def __init__(self, action_space, threat_platforms, defended_assets, placement_lat_band, placement_lng_band):
        self.action_space_size = action_space.shape[0]

        # Save asset locations
        self.threat_platforms_loc = [x.location for x in threat_platforms]
        self.defended_assets_loc = [(x['lat'], x['lng']) for _, x in defended_assets.iterrows()]

        # Save scaling parameters
        self.placement_lat_band = placement_lat_band
        self.placement_lng_band = placement_lng_band

    def get_action(self):
        return np.random.random(self.action_space_size)

    def scale_action(self, loc):
        action = [0, 0]
        action[0] = (loc[0] - self.placement_lat_band[0])/(self.placement_lat_band[1] - self.placement_lat_band[0]) 
        action[1] = (loc[1] - self.placement_lng_band[0])/(self.placement_lng_band[1] - self.placement_lng_band[0])

        return action

class actor_in_between(actor):
    def __init__(self, action_space, threat_platforms, defended_assets, placement_lat_band, placement_lng_band):
        super().__init__(action_space, threat_platforms, defended_assets, placement_lat_band, placement_lng_band)

    def get_action(self):
        # Select random asset
        threat_platform_loc = self.threat_platforms_loc[np.random.choice(len(self.threat_platforms_loc))]
        defended_asset_loc = self.defended_assets_loc[np.random.choice(len(self.defended_assets_loc))]

        # Determine location
        loc = np.array(threat_platform_loc) + (np.random.random()/2+0.25)*(np.array(defended_asset_loc) - np.array(threat_platform_loc))
        return self.scale_action(loc)


class actor_near_defense(actor):
    def __init__(self, action_space, threat_platforms, defended_assets, placement_lat_band, placement_lng_band):
        super().__init__(action_space, threat_platforms, defended_assets, placement_lat_band, placement_lng_band)

    def get_action(self):
        # Select random asset
        defended_asset_loc = self.defended_assets_loc[np.random.choice(len(self.defended_assets_loc))]

        # Determine location
        loc = defended_asset_loc
        return self.scale_action(loc)

class actor_spaced_out(actor):
    def __init__(self, action_space, threat_platforms, defended_assets, placement_lat_band, placement_lng_band):
        super().__init__(action_space, threat_platforms, defended_assets, placement_lat_band, placement_lng_band)

        # Initialize sobol sequence sampler
        self.sampler = qmc.Sobol(d=self.action_space_size, scramble=True)
        self.actions = list(self.sampler.random(1000))

    def get_action(self):
        return self.actions.pop(0)

