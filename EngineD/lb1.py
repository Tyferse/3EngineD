import configparser
import math
from abc import ABCMeta, abstractmethod


class Point:
    def __init__(self, x, y, z):
        self.coords = [x, y, z]
    
    def __str__(self):
        return "Point({:.4f}, {:.4f}, {:.4f})".format(*self.coords)
    
    def __bool__(self):
        return True
    
    def __add__(self, other):
        return Point(*[self.coords[i] + other.coords[i]
                       for i in range(3)])
    
    def __mul__(self, other):
        assert isinstance(other, (int, float))
        
        return Point(*[self.coords[i] * other
                       for i in range(3)])
    
    def __sub__(self, other):
        return self.__add__(-1 * other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other: [int, float]):
        assert other != 0
        return self.__mul__(1 / other)
    
    def distance(self, pt):
        return math.sqrt(sum((self.coords[i] - pt.coords[i]) ** 2
                             for i in range(3)))


class Vector:
    # vs = VectorSpace()
    def __init__(self, pt1):
        self.point = pt1  # Point(x, y, z)
        # self.vs = vs
    
    def __str__(self):
        return "Vector({:.4f}, {:.4f}, {:.4f})".format(
            *self.point.coords)
    
    def len(self):
        return self.vs.init_pt.distance(self.point)
    
    def norm(self):
        return Vector(self.point / self.len())
    
    def __bool__(self):
        return True
    
    def __add__(self, other: "Vector"):
        """
        Сумма векторов
        
        :param other:
        :return:
        """
        return Vector(self.point + other.point)
    
    def __sub__(self, other):
        """
        Разность векторов
        
        :param other:
        :return:
        """
        return Vector(self.point - other.point)
    
    def __mul__(self, other):
        """
        Умножение на число, скалярное произведение векторов.
        
        :param other:
        :return:
        """
        if isinstance(other, Vector):
            return sum(self.point.coords[i] * other.point.coords[i]
                       for i in range(3))
        else:
            return Vector(self.point * other)
    
    def __rmul__(self, other):
        """
        Умножение числа на вектор слева

        :param other:
        :return:
        """
        return Vector(self.point * other)
    
    def __truediv__(self, other):
        """
        Деление на число.
        
        :param other:
        :return:
        """
        assert isinstance(other, int)
        
        return Vector(self.point / other)
    
    def __pow__(self, other):
        """
        Векторное произведение.
        
        :param other:
        :return:
        """
        x1 = self.point.coords[0]
        y1 = self.point.coords[1]
        z1 = self.point.coords[2]
        x2 = other.point.coords[0]
        y2 = other.point.coords[1]
        z2 = other.point.coords[2]
        
        x = self.vs.basis[0] * (y1 * z2 - y2 * z1)
        y = self.vs.basis[1] * -(x1 * z2 - x2 * z1)
        z = self.vs.basis[2] * (y2 * x1 - y1 * x2)
        
        return x + y + z


class VectorSpace:
    init_pt = Point(0, 0, 0)
    basis = [Vector(Point(1, 0, 0)),
             Vector(Point(0, 1, 0)),
             Vector(Point(0, 0, 1))]
    
    def __init__(self, init_pt: Point = init_pt, dir1: Vector = None,
                 dir2: Vector = None, dir3: Vector = None):
        self.init_pt = init_pt
        
        for i, d in enumerate((dir1, dir2, dir3)):
            if d is not None:
                VectorSpace.basis[i] = d.norm()


class Camera:
    def __init__(self, position, look_at, look_dir, fov, draw_dist):
        """
        vfov = fov * H / W - вертикальный угол наклона
        
        :param position: Расположение.
        :param look_at: Направление камеры (Point).
        :param look_dir: Углы наклона камеры (Vector).
        :param fov: Горизонтальный угол наклона.
        :param draw_dist: Дистанция прорисовки.
        """
        self.pos = position
        self.look_at = look_at
        self.look_dir = look_dir
        
        self.fov = fov
        # self.vfov = fov * h / w
        
        self.draw_dist = draw_dist
    
    def send_rays(self, count) -> list[Vector]:
        pass


class Object:
    def __init__(self, pos: Point, rotation: Vector):
        self.pos = pos
        self.rot = rotation
    
    def contains(self, pt: Point) -> bool:
        pass


if __name__ == "__main__":
    vs = VectorSpace()
    Vector.vs = vs  # Передача векторного пространства в класс Vector
    p1 = Point(1, 2, 3)
    p2 = Point(3, 2, 1)
    v1 = Vector(p1)
    v2 = Vector(p2)
    print(v1 ** v2)
    print(v1.len())
    print(p1 * 3, 3 * p1)
