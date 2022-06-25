import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data

def visualize_solution(locations, best):
    plt.rcParams["figure.figsize"] = [10, 10]
    fig, ax = plt.subplots()
    ax.imshow(plt.imread("osna_map.png", format="png"), origin= 'lower', extent=[0, 1000, 1000, 0])
    
    house_img = OffsetImage(plt.imread("house.png", format="png"), zoom=0.015)
    hospital_img = OffsetImage(plt.imread("hospital.png", format="png"), zoom=0.013)

    for i, loc in enumerate(locations):
        if i in best:
            ab = AnnotationBbox(hospital_img, (loc[0], loc[1]), frameon=False)
        else:
            ab = AnnotationBbox(house_img, (loc[0], loc[1]), frameon=False)
        ax.add_artist(ab)

    # Fix the display limits to see everything
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)

    plt.show()


def get_locations():
    locations = np.asarray([[70,170], [80,220], [120,250], [125,200], [150,310], [135,350], [210,350], [260,380], [290,380], [315,380], [340,390],
                            [210,240], [260,235], [310,265], [335,290], [360,310], [380,330], [405,350], [430,370], [460,320], [435,310], [410,290],
                            [390,270], [365,250], [490,300], [465,280], [490,250], [510,275], [550,270], [589,265], [440,260], [620,280], [230,310],
                            [605,320], [650,325], [555,330], [590,355], [680,360], [700,400], [670,430], [575,380], [625,480], [460,460], [440,540],
                            [255,410], [285,415], [315,420], [340,425], [375,430], [420,430], [285,470], [315,475], [340,480], [375,490], [415,495],
                            [650,510], [700,540], [730,605], [675,580], [770,545], [670,290], [685,230], [770,235], [835,245], [735,265], [590,645],
                            [480,655], [420,630], [350,530], [280,550], [225,560], [180,470], [190,515], [350,600], [140,430], [365,655], [330,730],
                            [550,454], [600,685], [485,750], [500,800], [115,520], [150,600], [640,770], [605,535], [235,710], [790,280], [755,495],
                            [370,700], [700,460], [750,580], [690,200], [690,750], [220,435], [240,470], [600,570], [590,740], [500,700], [425,570], [190,400]
                            ])
    return locations