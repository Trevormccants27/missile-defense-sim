import gym
from gym import Env, spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from circles import circle
from shapely.geometry import Polygon
from descartes import PolygonPatch

import geopy.distance as dist

from threat_platform import ThreatPlatform
from defense_platform import DefensePlatform

class MDSEnv(Env):
    def __init__(self, config):
        super(MDSEnv, self).__init__()
        
        # Save config
        self.config = config

        # Set observation/action space for RL
        self.observation_space = spaces.Box(0, 255, [self.config['obs_height'], self.config['obs_width'], 3])
        self.action_space = spaces.Box(np.array([0,1]),np.array([0,1]),dtype=np.float32)

        # Create wgs grid
        self.grid_lat = np.linspace(self.config['obs_lat_band'][0], self.config['obs_lat_band'][1], self.observation_space.shape[0]+1)
        self.grid_lng = np.linspace(self.config['obs_lng_band'][0], self.config['obs_lng_band'][1], self.observation_space.shape[1]+1)
        #self.grid_lat, self.grid_lng = np.meshgrid(lat, lng)

        # Read in city location for defended assets and threat launchers
        self.japanese_cities = pd.read_csv('jp.csv')
        self.chinese_cities = pd.read_csv('cn.csv')

        # Create basemap object for checking if a coordinate is land or sea
        self.bm = Basemap()

        self.fig = None
                        
    def reset(self, seed=42):
        # Initialize random seed
        np.random.seed(seed)

        # Initialize budget
        self.budget = {}
        self.budget['US'] = self.config['US']['budget']
        self.budget['Japan'] = self.config['Japan']['budget']

        # Initialize Defended Assets
        self.defended_assets = self.japanese_cities.head(self.config['n_defended_assets'])
        self.initial_pop = np.sum(self.defended_assets['population'])

        # Initialize Threats
        self.threat_missiles = []
        self.threat_platforms = []
        for i in range(self.config['Enemy']['n_threat_platforms']):
            self.threat_platforms.append(ThreatPlatform((self.chinese_cities.iloc[i]['lat'], self.chinese_cities.iloc[i]['lng']), self.config['threat_platform'], seed=np.random.randint(1,1E6)))
        
        # Initialize defense platforms
        self.defense_platforms = []

        # Initialize global variables
        self.time_to_war = self.config['time_to_war']
        self.deploy_time = 0

        return self.get_observation()

    def get_observation(self):
        '''
        Observation Space:
        3 channel image each corresponding to:
            [0] - n_missiles in threat platform at each pixel (0 for no threat platform)
            [1] - P_a of detectors within range at that pixel
            [2] - P_k of interceptors in range at that pixel
        '''
        # Initialize observation
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        # for lat_idx in range(len(self.grid_lat)-1):
        #     for lng_idx in range(len(self.grid_lng)-1):
        #         lat_lower = self.grid_lat[lat_idx]
        #         lng_lower = self.grid_lng[lng_idx]
        #         lat_upper = self.grid_lat[lat_idx+1]
        #         lng_upper = self.grid_lng[lng_idx+1]
                
        #         # Populate threats
        #         for threat_platform in self.threat_platforms:
        #             if lat_lower <= threat_platform.location[0] < lat_upper and lng_lower <= threat_platform.location[1] < lng_upper:
        #                 obs[lat_idx,lng_idx,0] += threat_platform.n_missiles
        #         obs[lat_idx,lng_idx,0] /= self.config['threat_platform']['n_missiles'][1]

        #         for defense_platform in self.defense_platforms:
        #             # Populate Detectors
        #             if defense_platform.check_detection_range(((lat_lower+lat_upper)/2.0, (lng_lower+lng_upper)/2.0)):
        #                 obs[lat_idx,lng_idx,1] = 1 - ((1-obs[lat_idx,lng_idx,1]) * (1-defense_platform.config['p_a']))

        #             # Populate Interceptors
        #             if defense_platform.check_intercept_range(((lat_lower+lat_upper)/2.0, (lng_lower+lng_upper)/2.0)):
        #                 obs[lat_idx,lng_idx,2] = 1 - ((1-obs[lat_idx,lng_idx,2]) * (1-defense_platform.config['p_k']))

        return (obs*255).astype('uint8')

    def render(self, mode = "human"):
        assert mode in ["human", "observation"], "Invalid mode, must be \"human\" or \"observation\""

        # Create figure
        if self.fig is None:
            self.fig = plt.figure(figsize=(8, 8))
        else:
            plt.clf()

        if mode == "human":
            # Create basemap (background map)
            m = Basemap(projection='lcc', resolution='c',
                        width=4E6, height=4E6, 
                        lat_0=35, lon_0=125)

            # Make background map semi-transparent
            m.etopo(alpha=0.5)

            # Plot defended assets
            for r in range(len(self.defended_assets)):
                city = self.defended_assets.iloc[r]
                m.scatter(city['lng'], city['lat'], latlon=True, s=city['population']/1E5, c='k', alpha=0.5)

            # Plot threat platforms
            for threat_platform in self.threat_platforms:
                m.scatter(threat_platform.location[1], threat_platform.location[0], latlon=True, s=threat_platform.n_missiles*10, c='r', alpha=0.5)

            # Plot defense platforms
            for defense_platform in self.defense_platforms:
                if defense_platform.config['detection_range'] > 0:
                    self.plot_circle(m, defense_platform.location, defense_platform.config['detection_range'], fc='y', alpha=0.3)
                if defense_platform.config['n_interceptors'] > 0:
                    self.plot_circle(m, defense_platform.location, defense_platform.config['intercept_range'], fc='m', alpha=0.7*float(defense_platform.n_interceptors)/defense_platform.config['n_interceptors'])

            # Plot threat missiles
            for missile in self.threat_missiles:
                if missile is not None:
                    if missile.aquired:
                        missile_color = 'r'
                    else:
                        missile_color = 'k'
                    m.scatter(missile.location[1], missile.location[0], latlon=True, c=missile_color, s=50)
                    m.drawgreatcircle(missile.location[1], missile.location[0], missile.target_location[1], missile.target_location[0], markersize=5, c='r', linestyle='--')

        if mode == "observation":
            obs = self.get_observation()
            obs[:,:,0] = ((obs[:,:,0] - obs[:,:,0].min()) * (1/(obs[:,:,0].max() - obs[:,:,0].min()) * 255)).astype('uint8')
            obs[:,:,1] = ((obs[:,:,1] - obs[:,:,1].min()) * (1/(obs[:,:,1].max() - obs[:,:,1].min()) * 255)).astype('uint8')
            obs[:,:,2] = ((obs[:,:,2] - obs[:,:,2].min()) * (1/(obs[:,:,2].max() - obs[:,:,2].min()) * 255)).astype('uint8')
            plt.imshow(obs)

        # Render plot
        plt.draw()
        plt.pause(self.config['plot_delay'])

    def plot_circle(self, m, location, radius, **kwargs):
        '''
        Plots a circle with specified radius in km on map
        '''
        pol = Polygon(circle(m, location[1], location[0], radius))
        patch = PolygonPatch(pol, **kwargs)
        plt.gca().add_patch(patch)
        
    def close(self):
        plt.close()

    def step(self, action, nation):
        '''
        Action space:
        [lat, lng] of where to place a defense platform
        If budget constraints are violated, nothing will happen.
        Time will move forward by the deploy time of the asset placed
        '''
        done = False

        # Place defense platforms
        lat = self.config['placement_lat_band'][0] + action[0]*(self.config['placement_lat_band'][1] - self.config['placement_lat_band'][0])
        lng = self.config['placement_lng_band'][0] + action[1]*(self.config['placement_lng_band'][1] - self.config['placement_lng_band'][0])
        if self.deploy_time <= 0:
            self.deploy_time = self.place_defense_platforms((lat, lng), nation)
        
        # Update deploy time
        self.deploy_time -= 1

        # Update threat missiles
        if self.time_to_war <= self.get_political_tension() or self.budget['US'] < 2000:
            self.step_threat_platforms()
            self.step_threat_missiles()
        
        # Stop when all defended assets are destroyed
        if len(self.defended_assets) < 1:
            done = True

        # Stop when the threat launchers have no more missiles
        if np.sum([x.n_missiles for x in self.threat_platforms]) < 1:
            done = True

        # Update time to war
        self.time_to_war -= 1

        # Update reward
        reward = 0
        if done:
            for r in range(len(self.defended_assets)):
                reward += self.defended_assets.iloc[r]['population']

        return self.get_observation(), reward, done, {'US_budget': self.budget['US'], 'reward_perc': reward/self.initial_pop}

    def place_defense_platforms(self, location, nation):
        deploy_time = 0
        if self.bm.is_land(location[1], location[0]):
            # Place land defense platform
            cost = self.config['ground_interceptor']['cost']
            if self.budget[nation] > cost:
                self.defense_platforms.append(DefensePlatform(location, self.config['ground_interceptor']))
                self.budget[nation] -= cost
                deploy_time += self.config['ground_interceptor']['deploy_time']

            cost = self.config['ground_detector']['cost']
            if self.budget[nation] > cost:
                self.defense_platforms.append(DefensePlatform(location, self.config['ground_detector']))
                self.budget[nation] -= cost
                deploy_time += self.config['ground_detector']['deploy_time']
        else:
            # Place sea defense platform
            cost = self.config['ship']['cost']
            if self.budget[nation] > cost:
                self.defense_platforms.append(DefensePlatform(location, self.config['ship']))
                self.budget[nation] -= cost
                deploy_time += self.config['ship']['deploy_time']

        return deploy_time

    def step_threat_platforms(self):
        for threat_platform in self.threat_platforms:
            result = threat_platform.step(self.defended_assets)
            if result is not None:
                self.threat_missiles.append(result)

    def step_threat_missiles(self):
        '''
        Updates threat missiles by 1 time step
        '''
        for missile_idx, missile in enumerate(self.threat_missiles):
            if missile is not None:
                # Update missile location
                threat_loc = missile.step()

                # Check if defense platforms can act
                for defense_platform in self.defense_platforms:
                    # Attempt to detect missile
                    if not missile.aquired:
                        if defense_platform.check_detection_range(threat_loc):
                            if defense_platform.check_detection(missile_idx):
                                missile.aquire()
                    
                    # Attempt to intercept missile
                    if missile.aquired:
                        if defense_platform.check_intercept_range(threat_loc):
                            if defense_platform.check_intercept(missile_idx):
                                self.threat_missiles[missile_idx] = None

                # If missile arrives at it's target, destroy it
                if np.linalg.norm(missile.location - missile.target_location) < 1:
                    self.defended_assets.drop(self.defended_assets.index[self.defended_assets['city'] == missile.target_name], inplace=True)
                    self.threat_missiles[missile_idx] = None

    def get_political_tension(self):
        political_tension = 0

        for defense_platform in self.defense_platforms:
            distances = []
            for threat_platform in self.threat_platforms:
                distances.append(dist.geodesic(defense_platform.location, threat_platform.location).km)

            political_tension += np.min(distances)/self.config['political_tensions_factor']

        return political_tension