'''
Created on Apr 5, 2012

@author: lanquarden
'''

class MsgParser:
    '''
    A parser for received UDP messages and building UDP messages
    '''
    
    def __init__(self):
        '''Constructor'''
        pass  # Empty constructor, explicit pass added

    def parse(self, str_sensors):
        '''Return a dictionary with tags and values from the UDP message'''
        sensors = {}
        
        b_open = str_sensors.find('(')
        
        while b_open >= 0:
            b_close = str_sensors.find(')', b_open)
            if b_close >= 0:
                substr = str_sensors[b_open + 1: b_close]
                items = substr.split()
                
                if len(items) < 2:
                    print(f"Problem parsing substring: {substr}")
                else:
                    sensors[items[0]] = items[1:]  # Extract values directly
                
                b_open = str_sensors.find('(', b_close)
            else:
                print(f"Problem parsing sensor string: {str_sensors}")
                return None
        
        return sensors

    def stringify(self, dictionary):
        '''Build a UDP message from a dictionary'''
        msg = ''

        for key, value in dictionary.items():
            if value and value[0] is not None:
                msg += f"({key} {' '.join(map(str, value))})"
        
        return msg
