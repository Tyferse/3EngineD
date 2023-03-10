class PointAlphaTest:
    pi = 3.1415926
    brain = -1
    iq = 0.0
    
    def print(self):
        print("this thing does\'nt make sense")


class PointBetaTest:
    pi = 6.263
    iq = 90
    brain = False


class Point(PointBetaTest):
    x = 10
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def print(self):
        print(f'({self.x}, {self.y})')
        return 0

    
p = Point(4, 5)
p.print()
print(p.pi)
