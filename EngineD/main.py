from movement import *


cam = Camera(Point(-2, 0, 0), Vector(1, 0, 0), 90, 20)

map1 = Map()
map1.append(Sphere(Point(1, 0.5, 0), Vector(0, 0, 1), 0.8))
map1.append(Sphere(Point(1, 0.5, -2), Vector(0, 10, 0), 0.8))
map1.append(Sphere(Point(1, 0.5, -4), Vector(0, 1, 0), 0.8))

cons = Console(map1, cam)

launch(cons, move_speed=0.25, sensitivity=2)
