'''
Created on Apr 5, 2012

@author: lanquarden
'''

import msgParser

class CarState:
    '''
    Class that holds all the car state variables
    '''

    def __init__(self):
        '''Constructor'''
        self.parser = msgParser.MsgParser()
        self.sensors = None
        self.angle = None
        self.cur_lap_time = None
        self.damage = None
        self.dist_from_start = None
        self.dist_raced = None
        self.focus = None
        self.fuel = None
        self.gear = None
        self.last_lap_time = None
        self.opponents = None
        self.race_pos = None
        self.rpm = None
        self.speed_x = None
        self.speed_y = None
        self.speed_z = None
        self.track = None
        self.track_pos = None
        self.wheel_spin_vel = None
        self.z = None
    
    def set_from_msg(self, str_sensors):
        self.sensors = self.parser.parse(str_sensors)
        
        self.angle = self.get_float_d('angle')
        self.cur_lap_time = self.get_float_d('curLapTime')
        self.damage = self.get_float_d('damage')
        self.dist_from_start = self.get_float_d('distFromStart')
        self.dist_raced = self.get_float_d('distRaced')
        self.focus = self.get_float_list_d('focus')
        self.fuel = self.get_float_d('fuel')
        self.gear = self.get_int_d('gear')
        self.last_lap_time = self.get_float_d('lastLapTime')
        self.opponents = self.get_float_list_d('opponents')
        self.race_pos = self.get_int_d('racePos')
        self.rpm = self.get_float_d('rpm')
        self.speed_x = self.get_float_d('speedX')
        self.speed_y = self.get_float_d('speedY')
        self.speed_z = self.get_float_d('speedZ')
        self.track = self.get_float_list_d('track')
        self.track_pos = self.get_float_d('trackPos')
        self.wheel_spin_vel = self.get_float_list_d('wheelSpinVel')
        self.z = self.get_float_d('z')
    
    def to_msg(self):
        self.sensors = {
            'angle': [self.angle],
            'curLapTime': [self.cur_lap_time],
            'damage': [self.damage],
            'distFromStart': [self.dist_from_start],
            'distRaced': [self.dist_raced],
            'focus': self.focus,
            'fuel': [self.fuel],
            'gear': [self.gear],
            'lastLapTime': [self.last_lap_time],
            'opponents': self.opponents,
            'racePos': [self.race_pos],
            'rpm': [self.rpm],
            'speedX': [self.speed_x],
            'speedY': [self.speed_y],
            'speedZ': [self.speed_z],
            'track': self.track,
            'trackPos': [self.track_pos],
            'wheelSpinVel': self.wheel_spin_vel,
            'z': [self.z]
        }
        return self.parser.stringify(self.sensors)
    
    def get_float_d(self, name):
        try:
            return float(self.sensors[name][0])
        except (KeyError, TypeError, ValueError):
            return None
    
    def get_float_list_d(self, name):
        try:
            return [float(v) for v in self.sensors[name]]
        except (KeyError, TypeError, ValueError):
            return None
    
    def get_int_d(self, name):
        try:
            return int(self.sensors[name][0])
        except (KeyError, TypeError, ValueError):
            return None
    
    def set_angle(self, angle):
        self.angle = angle

    def get_angle(self):
        return self.angle

    def set_cur_lap_time(self, cur_lap_time):
        self.cur_lap_time = cur_lap_time

    def get_cur_lap_time(self):
        return self.cur_lap_time

    def set_damage(self, damage):
        self.damage = damage

    def get_damage(self):
        return self.damage

    def set_dist_from_start(self, dist_from_start):
        self.dist_from_start = dist_from_start

    def get_dist_from_start(self):
        return self.dist_from_start

    def set_dist_raced(self, dist_raced):
        self.dist_raced = dist_raced

    def get_dist_raced(self):
        return self.dist_raced

    def set_focus(self, focus):
        self.focus = focus

    def get_focus(self):
        return self.focus

    def set_fuel(self, fuel):
        self.fuel = fuel

    def get_fuel(self):
        return self.fuel

    def set_gear(self, gear):
        self.gear = gear

    def get_gear(self):
        return self.gear

    def set_last_lap_time(self, last_lap_time):
        self.last_lap_time = last_lap_time

    def get_last_lap_time(self):
        return self.last_lap_time

    def set_opponents(self, opponents):
        self.opponents = opponents

    def get_opponents(self):
        return self.opponents

    def set_race_pos(self, race_pos):
        self.race_pos = race_pos

    def get_race_pos(self):
        return self.race_pos

    def set_rpm(self, rpm):
        self.rpm = rpm

    def get_rpm(self):
        return self.rpm

    def set_speed_x(self, speed_x):
        self.speed_x = speed_x

    def get_speed_x(self):
        return self.speed_x

    def set_speed_y(self, speed_y):
        self.speed_y = speed_y

    def get_speed_y(self):
        return self.speed_y

    def set_speed_z(self, speed_z):
        self.speed_z = speed_z

    def get_speed_z(self):
        return self.speed_z

    def set_track(self, track):
        self.track = track

    def get_track(self):
        return self.track

    def set_track_pos(self, track_pos):
        self.track_pos = track_pos

    def get_track_pos(self):
        return self.track_pos

    def set_wheel_spin_vel(self, wheel_spin_vel):
        self.wheel_spin_vel = wheel_spin_vel

    def get_wheel_spin_vel(self):
        return self.wheel_spin_vel

    def set_z(self, z):
        self.z = z

    def get_z(self):
        return self.z
