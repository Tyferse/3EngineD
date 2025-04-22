from movement import *


cam = Camera(Point(0, 1, -4), Vector(0, 0, 1), 90, 20)

map1 = Map(Sphere(Point(2, 2, 0), Vector(0, 0, 1), 1))
map1.append(Sphere(Point(3, 2.75, 0.5), Vector(0, 0, 1), 0.75))
map1.append(BoundedPlane(Point(2, 2, 0), Vector(2, -1, 0), 1, 1.25))
map1.append(Plane(Vector.vs.init_pt, Vector(0, 1, 0)))
map1[0].pr.move(Point(0, 0.5, 0))

cons = Console(map1, cam)

launch(cons, move_speed=0.5, sensitivity=4)
