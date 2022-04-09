import pandas as pd
import numpy as np
import geopy.distance as dist

class DefensePlatform():
    def __init__(self, location, config, seed=0):
        # Save location attributes
        self.location = location

        # Save config
        self.config = config
        self.n_interceptors = self.config['n_interceptors']
        
        # Initialize random seed
        #np.random.seed(seed)

        # Keep track of what threats have been processed
        self.threat_id_detection_list = []
        self.threat_id_intercept_list = []

    def check_intercept(self, threat_id):
        if threat_id not in self.threat_id_intercept_list:
            self.threat_id_intercept_list.append(threat_id)
            if self.n_interceptors > 0:
                self.n_interceptors -= 1
                return np.random.random() < self.config['p_k']
            else:
                return False
        else:
            return False

    def check_intercept_range(self, threat_location):
        return dist.geodesic(self.location, threat_location).km < self.config['intercept_range']

    def check_detection(self, threat_id):
        if threat_id not in self.threat_id_detection_list:
            self.threat_id_detection_list.append(threat_id)
            return np.random.random() < self.config['p_a']
        else:
            return False

    def check_detection_range(self, threat_location):
        return dist.geodesic(self.location, threat_location).km < self.config['detection_range']