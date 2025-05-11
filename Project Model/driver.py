import msgParser
import carState
import carControl
import csv
import os
import time
import torch
import torch.nn as nn
import joblib
import numpy as np
from datetime import datetime  # Added import to fix 'datetime' not defined error
try:
    import keyboard
except ImportError:
    print("Keyboard module not available - running in autonomous mode")
    keyboard = None

class RaceController(nn.Module):
    def __init__(self, input_dim, num_gear_classes=8):
        super(RaceController, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2)
        )
        self.regression = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Tanh()
        )
        self.classification = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_gear_classes)
        )
    
    def forward(self, x):
        shared = self.shared(x)
        reg_out = self.regression(shared)
        cls_out = self.classification(shared)
        return reg_out, cls_out

class Driver:
    def __init__(self, stage, client_id="CAR1", model_path='race_controller.pth', scaler_path='scaler.pkl'):
        self.WARM_UP = 0
        self.QUALIFYING = 1
        self.RACE = 2
        self.UNKNOWN = 3
        self.stage = stage
        self.client_id = client_id

        self.log_fields = [
            'timestamp', 'client_id', 'Angle', 'CurrentLapTime', 'Damage', 'DistanceFromStart',
            'DistanceCovered', 'FuelLevel', 'Gear', 'LastLapTime', 'RacePosition', 'RPM',
            'SpeedX', 'SpeedY', 'SpeedZ', 'TrackPosition', 'WheelSpinVelocity_1',
            'WheelSpinVelocity_2', 'WheelSpinVelocity_3', 'WheelSpinVelocity_4', 'Z',
            'Acceleration', 'Braking', 'Clutch', 'Steering', 'Gear_cmd'
        ] + [f'Track_{i}' for i in range(1, 20)] + [f'Opponent_{i}' for i in range(1, 37)]

        self.log_file = self._init_logfile()
        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()

        self.steer_lock = 0.785398
        self.max_speed = 200
        self.prev_rpm = None
        
        self.angles = [0] * 19
        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15
        for i in range(5, 9):
            self.angles[i] = -20 + (i - 5) * 5
            self.angles[18 - i] = 20 - (i - 5) * 5

        self.last_log_time = time.time()
        self.log_interval = 0.02
        self.last_lap_time = 0.0
        self.use_ai = True

        self.model = None
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model(model_path, scaler_path)

    def _init_logfile(self):
        log_path = os.path.abspath('telemetry_log.csv')
        write_headers = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
        if write_headers:
            with open(log_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.log_fields)
                writer.writeheader()
        return log_path

    def load_model(self, model_path, scaler_path):
        try:
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            self.scaler = joblib.load(scaler_path)
            self.model = RaceController(input_dim=43)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print("Model and scaler loaded successfully")
        except Exception as e:
            print(f"Failed to load model or scaler: {e}")
            self.model = None
            self.scaler = None
            self.use_ai = False

    def init(self):
        return self.parser.stringify({'init': self.angles})

    def drive(self, msg):
        try:
            self.state.set_from_msg(msg)
            if keyboard and keyboard.is_pressed('i'):
                if self.model and self.scaler:
                    self.use_ai = not self.use_ai
                    print(f"AI control: {'ON' if self.use_ai else 'OFF'}")
                else:
                    print("AI model not available")
                time.sleep(0.3)
            
            if self.use_ai and self.model and self.scaler:
                self._autonomous_control()
            else:
                self._manual_control()
            
            current_time = time.time()
            if current_time - self.last_log_time >= self.log_interval:
                self._log_telemetry()
                self.last_log_time = current_time
        except Exception as e:
            print(f"Drive error: {str(e)}")
        return self.control.to_msg()

    def _autonomous_control(self):
        try:
            input_features = [
                self._safe_get(self.state.angle),
                self._safe_get(self.state.cur_lap_time),
                self._safe_get(self.state.damage),
                self._safe_get(self.state.dist_from_start),
                self._safe_get(self.state.dist_raced),
                self._safe_get(self.state.fuel),
                self._safe_get(self.state.gear),
                self._safe_get(self.state.last_lap_time),
                self._safe_get(self.state.race_pos),
                self._safe_get(self.state.rpm),
                self._safe_get(self.state.speed_x),
                self._safe_get(self.state.speed_y),
                self._safe_get(self.state.speed_z),
                self._safe_get(self.state.track_pos),
            ]
            wheel_spin = self.state.get_wheel_spin_vel() or [0.0] * 4
            input_features.extend([self._safe_get(w) for w in wheel_spin])
            input_features.append(self._safe_get(self.state.z))
            track = self.state.get_track() or [0.0] * 19
            input_features.extend([self._safe_get(t) for t in track])
            opponents = self.state.get_opponents() or [0.0] * 36
            input_features.extend([self._safe_get(opponents[i]) for i in [0, 8, 17, 26, 35]])  # Fixed indexing
            
            X = np.array([input_features])
            X_scaled = self.scaler.transform(X)
            
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            with torch.no_grad():
                reg_out, cls_out = self.model(X_tensor)
                reg_out = reg_out.cpu().numpy()[0]
                cls_out = torch.argmax(cls_out, dim=1).cpu().numpy()[0]
            
            accel = np.clip(reg_out[0], 0.0, 1.0)
            brake = np.clip(reg_out[1], 0.0, 1.0)
            steer = np.clip(reg_out[2], -1.0, 1.0)
            gear = int(cls_out - 1)
            
            self.control.set_accel(accel)
            self.control.set_brake(brake)
            self.control.set_steer(steer)
            self.control.set_gear(gear)
        except Exception as e:
            print(f"Neural network prediction error: {str(e)}")
            self._rule_based_control()

    def _manual_control(self):
        accel, brake = self._get_manual_inputs()
        self.control.set_accel(accel)
        self.control.set_brake(brake)
        self._steer()
        self._gear()

    def _get_manual_inputs(self):
        accel = 0.0
        brake = 0.0
        if keyboard and keyboard.is_pressed('w'):
            accel = 1.0
        if keyboard and keyboard.is_pressed('s'):
            brake = 1.0
        return accel, brake

    def _steer(self):
        angle = self.state.angle or 0
        dist = self.state.track_pos or 0
        speed = self.state.speed_x or 0
        dist_weight = 0.1 + (speed / self.max_speed) * 0.2
        steer = (angle - dist * dist_weight) / self.steer_lock
        max_steer = 0.2
        steer = max(min(steer, max_steer), -max_steer)
        current_steer = self.control.steer or 0
        smoothed_steer = 0.7 * current_steer + 0.3 * steer
        self.control.set_steer(smoothed_steer)

    def _gear(self):
        rpm = self.state.rpm or 0
        gear = self.state.gear or 1
        speed = self.state.speed_x or 0
        brake = self.control.brake or 0
        if brake > 0.1:
            gear = max(gear - 1, 1)
        else:
            if rpm > 6800 and speed > 50:
                gear = min(gear + 1, 6)
            elif rpm < 3800 and speed < 150:
                gear = max(gear - 1, 1)
        self.control.set_gear(gear)
        self.prev_rpm = rpm

    def _rule_based_control(self):
        self._steer()
        self._gear()
        speed = self.state.speed_x or 0
        if speed < self.max_speed:
            self.control.set_accel(min(1.0, self.control.accel + 0.1))
        else:
            self.control.set_accel(max(0.0, self.control.accel - 0.1))

    def _log_telemetry(self):
        try:
            entry = {
                'timestamp': datetime.now().isoformat(),
                'client_id': self.client_id,
                'Angle': self._safe_get(self.state.angle),
                'CurrentLapTime': self._safe_get(self.state.cur_lap_time),
                'Damage': self._safe_get(self.state.damage),
                'DistanceFromStart': self._safe_get(self.state.dist_from_start),
                'DistanceCovered': self._safe_get(self.state.dist_raced),
                'FuelLevel': self._safe_get(self.state.fuel),
                'Gear': self._safe_get(self.state.gear),
                'LastLapTime': self._safe_get(self.state.last_lap_time),
                'RacePosition': self._safe_get(self.state.race_pos),
                'RPM': self._safe_get(self.state.rpm),
                'SpeedX': self._safe_get(self.state.speed_x),
                'SpeedY': self._safe_get(self.state.speed_y),
                'SpeedZ': self._safe_get(self.state.speed_z),
                'TrackPosition': self._safe_get(self.state.track_pos),
                'Z': self._safe_get(self.state.z),
                'Acceleration': self._safe_get(self.control.accel),
                'Braking': self._safe_get(self.control.brake),
                'Clutch': self._safe_get(self.control.clutch),
                'Steering': self._safe_get(self.control.steer),
                'Gear_cmd': self._safe_get(self.control.gear),
            }
            if self.state.track:
                for i in range(19):
                    entry[f'Track_{i+1}'] = self._safe_get(self.state.track[i])
            if self.state.opponents:
                for i in range(min(36, len(self.state.opponents))):
                    entry[f'Opponent_{i+1}'] = self._safe_get(self.state.opponents[i])
            if self.state.wheel_spin_vel:
                for i in range(4):
                    entry[f'WheelSpinVelocity_{i+1}'] = self._safe_get(self.state.wheel_spin_vel[i])
            if self.state.last_lap_time and self.state.last_lap_time > self.last_lap_time:
                print(f"Lap completed in {self.state.last_lap_time:.2f} seconds")
                self.last_lap_time = self.state.last_lap_time
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.log_fields)
                writer.writerow(entry)
        except Exception as e:
            print(f"Logging failed: {str(e)}")

    def _safe_get(self, value, default=0.0):
        return value if value is not None else default

    def on_shutdown(self):
        print(f"Shutting down driver {self.client_id}")

    def on_restart(self):
        print(f"Restarting driver {self.client_id}")
        self.prev_rpm = None
        self.last_lap_time = 0.0