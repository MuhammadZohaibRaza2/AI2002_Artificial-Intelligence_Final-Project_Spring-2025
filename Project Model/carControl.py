'''
Created on Apr 5, 2012

@author: lanquarden
'''

import msgParser

class CarControl:
    '''
    An object holding all the control parameters of the car
    '''
    # TODO range check on set parameters

    def __init__(self, accel=0.0, brake=0.0, gear=1, steer=0.0, clutch=0.0, focus=0, meta=0):
        '''Constructor'''
        self.parser = msgParser.MsgParser()
        self.actions = None

        self.accel = accel
        self.brake = brake
        self.gear = gear
        self.steer = steer
        self.clutch = clutch
        self.focus = focus
        self.meta = meta

    def to_msg(self):
        self.actions = {
            'accel': [self.accel],
            'brake': [self.brake],
            'gear': [self.gear],
            'steer': [self.steer],
            'clutch': [self.clutch],
            'focus': [self.focus],
            'meta': [self.meta]
        }
        return self.parser.stringify(self.actions)

    def set_accel(self, accel):
        self.accel = accel

    def get_accel(self):
        return self.accel

    def set_brake(self, brake):
        self.brake = brake

    def get_brake(self):
        return self.brake

    def set_gear(self, gear):
        self.gear = gear

    def get_gear(self):
        return self.gear

    def set_steer(self, steer):
        self.steer = steer

    def get_steer(self):
        return self.steer

    def set_clutch(self, clutch):
        self.clutch = clutch

    def get_clutch(self):
        return self.clutch

    def set_meta(self, meta):
        self.meta = meta

    def get_meta(self):
        return self.meta
