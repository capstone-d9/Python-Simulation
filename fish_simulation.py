from utils import FishWaterQualitySimulation, FishAgent

ecosystem = FishWaterQualitySimulation(width_pond=550, height_pond=550)
fish = FishAgent(xpos=25, ypos=25)
oxygen_sources = [(100, 100), (100, 50)]

total_frames = 1000
for _ in range(total_frames):
    fish.FollowMostHightOxygen(oxygentSource=oxygen_sources)
    
    tempframe = ecosystem.drawFish(
        fish_h=fish.fish_h,
        fish_w=fish.fish_w,
        x_pos=fish.xpos,
        y_pos=fish.ypos,
        to=fish.direction
    )
    
    ecosystem.__frameDatas__.append(tempframe)

ecosystem.AnimateFrame()