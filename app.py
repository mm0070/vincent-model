from flask import Flask
import json
app = Flask(__name__)

ZONE_BED_X = [570, 785]
ZONE_BED_Y = [500, 850]

ZONE_BOWLS_X = [1250, 1550]
ZONE_BOWLS_Y = [460, 680]

ZONE_PLAY_X = [820, 1200]
ZONE_PLAY_Y = [420, 900]

PLAYING_THRESHOLD = 2.5 

def whats_vincent_up_to():
    # read latest detect.json and load into vars
    with open('detect.json') as json_file:
        data = json.load(json_file)

    x = data["average_position_x"]
    y = data["average_position_y"]
    avg_delta = data["average_position_delta_x"] + data["average_position_delta_y"]

    # if position delta is more than this then he's playing regardless of his actual position
    if avg_delta >= PLAYING_THRESHOLD:
        return "playing"

    # establish where he is
    if (ZONE_BED_X[0] <= x <= ZONE_BED_X[1]) and (ZONE_BED_Y[0] <= y <= ZONE_BED_Y[1]):
        return "sleeping"

    if (ZONE_BOWLS_X[0] <= x <= ZONE_BOWLS_X[1]) and (ZONE_BOWLS_Y[0] <= y <= ZONE_BOWLS_Y[1]):
        return "eating"

    if (ZONE_PLAY_X[0] <= x <= ZONE_PLAY_X[1]) and (ZONE_PLAY_Y[0] <= y <= ZONE_PLAY_Y[1]):
        return "chilling"

    # not in any of the defined zones
    return "no idea"

@app.route("/")
def home():
    return "It looks like he's " + whats_vincent_up_to()