import numpy as np
import matplotlib.pyplot as plt

SIMULATION_PATH = "results"

# h_pond, w_pond = 550, 550
w_pond, h_pond = 20, 20
pond_color = [177, 220, 234]

fish_w, fish_h = 2, 5
fish_color = [255, 125, 0]
fish_head_color = [255, 200, 0]

initial_pond_frame = np.full((w_pond, h_pond, 3), pond_color, dtype=np.uint8)
# position_x, position_y = w_pond // 2, h_pond // 2
position_x, position_y = 12, 12

# draw fish in the pond
# initial_pond_frame[position_y - fish_h // 2:position_y + fish_h // 2, position_x - fish_w // 2:position_x + fish_w // 2] = fish_color
# initial_pond_frame[position_y - fish_h // 2:position_y - fish_h // 3, position_x - fish_w // 2:position_x + fish_w // 2] = fish_head_color

initial_pond_frame[position_y - fish_h // 2:position_y + fish_h // 2, position_x - fish_w // 2:position_x + fish_w // 2] = fish_color
initial_pond_frame[position_y - fish_h // 2:position_y - fish_h // 3, position_x - fish_w // 2:position_x + fish_w // 2] = fish_head_color


plt.imshow(initial_pond_frame)
plt.axis('off')
plt.show()