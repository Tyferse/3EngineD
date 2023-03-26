import configparser
import itertools
import math
from abc import ABCMeta, abstractmethod


class Point:
    def __init__(self, x, y, z):
        self.coords = [x, y, z]
    
    def __str__(self):
        return "Point({:.4f}, {:.4f}, {:.4f})".format(*self.coords)
    
    def __bool__(self):
        return bool(sum(self.coords))
    
    def __eq__(self, other):
        return self.coords == other.coords
    
    def __ne__(self, other):
        return self.coords != other.coords
    
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
    def __init__(self, *args):
        if len(args) == 1:
            assert isinstance(args[0], Point)
            self.point = args[0]  # Point(x, y, z)
        elif len(args) == 3:
            assert all(map(isinstance, args, [(int, float)] * 3))
            self.point = Point(*args)
            
        # self.vs = vs
    
    def __str__(self):
        return "Vector({:.4f}, {:.4f}, {:.4f})".format(
            *self.point.coords)
    
    def len(self):
        return self.vs.init_pt.distance(self.point)
    
    def norm(self):
        return Vector(self.point / self.len())
    
    def __bool__(self):
        return bool(self.point)
    
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
        assert isinstance(other, (int, float))
        
        return Vector(self.point * other)
    
    def __truediv__(self, other):
        """
        Деление на число.
        
        :param other:
        :return:
        """
        assert isinstance(other, (int, float))
        
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
    basis = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
    
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

        config = configparser.ConfigParser()
        config.read("config.cfg")
        h = int(config['SCREEN']['width'])
        w = int(config['SCREEN']['hight'])
        self.fov = fov
        self.vfov = fov * h / w
        
        self.draw_dist = draw_dist
    
    def send_rays(self, count) -> list[Vector]:
        pass


class Object:
    def __init__(self, pos: Point, rotation: Vector, **params):
        self.pos = pos
        self.rot = rotation
        self.pr = params
    
    @abstractmethod
    def contains(self, pt: Point) -> bool:
        return False
    
    @abstractmethod
    def intersect(self, v: Vector, pt_begin: Point) -> Point:
        """
        Точка пересечения или выходящая за
        поле видимости (дальности прорисовки) (draw_distance)
        
        :param v: Вектор (радиус-вектор)
        :param pt_begin: Точка начала вектора
        :return:
        """
        return Vector.vs.init_pt
    
    @abstractmethod
    def nearest_point(self, *pts: list[Point]) -> Point:
        pass


class Plane(Object):
    """
    Плоскость
    
    Параметрическое уравнение плоскости
    {
     x = a * t + b * s + c
     y = f * t + g * s + h
     z = p * t + q * s + v
    }
    
    a, b, f, g, p, q - координаты двух направляющих векторов
    c, h, v - координаты точки, принадлежащей плоскости
    t, s - параметры
    """
    def contains(self, pt: Point) -> bool:
        # return sum(self.rot.point.coords[i] *
        #            (pt.coords[i] - self.pos.coords[i])
        #            for i in range(3)) == 0
        return self.rot * Vector(pt - self.pos) == 0
    
    def intersect(self, v: Vector, pt_begin: Point = Vector.vs.init_pt):
        """
        Скалярное произведение вектора нормали к плоскости и вектора v
        не равно нулю и вектор v не лежит на плоскости, то вектор v
        не пересекает плоскость.
        
        Если координаты вектора v соответствуют уравнению плоскости,
        то этот вектор лежит на плоскости.
        
        Иначе, если скалярное произведение вектора нормали к плоскости
        и вектора v равно нулю и вектор v не лежит на плоскости,
        то этот вектор параллелен ей и не пересекает её
        ни в какой точке.
        
        :param v: Радиус-вектор отрезка.
        :param pt_begin: Точка начала вектора.
        :return: Точка пересечения или точка,
                 ближайшая к центру плоскости.
        """
        if self.rot * v != 0 and not self.contains(pt_begin):
            """
            Параметрическое уравнение прямой, на которой лежит вектор v:
            x = v.point.coords[0] * t0 + pt_begin.coords[0]
            y = v.point.coords[1] * t0 + pt_begin.coords[1]
            z = v.point.coords[2] * t0 - pt_begin.coords[2]
            
            Подставляем параметрические значения в уравнение плоскости:
            self.rot.point.coords[0] * (v.point.coords[0] * t0
                + pt_begin.coords[0] - self.pos.coords[0]) +
            self.rot.point.coords[1] * (v.point.coords[1] * t0
                + pt_begin.coords[1] - self.pos.coords[1]) +
            self.rot.point.coords[2] * (v.point.coords[2] * t0
                + pt_begin.coords[2] - self.pos.coords[2]) = 0
            
            Ищем параметр t0:
            xi, yi, zi - координаты начальной точки
            xv, yv, zv - координаты вектора v
            xс, yс, zс - координаты точки центра плоскости
            A, B, C - координаты вектора нормали к плоскости
            
            A * xv * t0 - A * xс + A * xi +
            B * yv * t0 - B * yс + B * yi +
            C * zv * t0 - C * zс + C * zi = 0
            
            t0 * (A * xv + B * yv + C * zv) = A * xс + B * yс + C * zс
                                            - A * xi - B * yi - C * zi
            
            t0 = (A * xс + B * yс * C * zс - A * xi - B * yi - C * zi) /
                 (A * xv + B * yv + C * zv)
            
            t0 = (self.rot * self.pos - self.rot * Vector(pt_begin)) /
                 (self.rot * v)
            
            Подставляем значение параметра в параметрическое
            уравнение прямой и находим координаты точки пересечения:
            Q = self.rot.point * t0 - vs.init_pt
            Q(self.rot.point.coords[0] * t0 - 0,
              self.rot.point.coords[1] * t0 - 0,
              self.rot.point.coords[2] * t0 - 0) - точка пересечения
            """
            t0 = (self.rot * Vector(self.pos) -
                  self.rot * Vector(pt_begin)) / (self.rot * v)
            if 0 <= t0 <= 1:
                return v * t0
            
        elif self.contains(pt_begin):
            """
            Возвращаем ближайшую к центру плоскости точку.
            
            (x - xi) / xv = (y - yi) / yv = (z - zi)  zv
            
            projection = v * Vector(self.pos - pt_begin) / v.len() -
            """
            # Расстояние от точки до прямой
            r = self.rot * Vector(pt_begin) / self.rot.len()
            if r == 0 and 0 <= (self.rot.point.coords[0]
                                - pt_begin) / v.point.coords[0] <= 1:
                return self.pos
            
            # Вектор, соединяющий точку центра и точку начала вектора v
            tempv = Vector(self.pos - pt_begin)
            # Проекция вектора tempv на вектор V
            projection = v * tempv / tempv.len()
            # Возвращаем координаты точки, ближайшей к вектору
            if 0 <= projection <= 1:
                return projection * v.point + pt_begin
            elif projection > 1:
                return v.point
            else:
                return pt_begin
        
        return Vector.vs.init_pt
    
    def nearest_point(self, *pts: Point) -> Point:
        r_min = 10 ** 9
        min_pt = Vector.vs.init_pt
        for pt in pts:
            r = abs(self.rot * Vector(pt - self.pos)) / self.rot.len()
            if r < r_min:
                r_min = r
                min_pt = pt
        
        return min_pt


class BoundedPlane(Plane):
    """
    Ограниченная плоскость
    
    delta_t
    delta_s
    
    -delta_t <= t <= delta_t - ограничения для параметра t
    -delta_s <= s <= delta_s - ограничения для параметра s
    """

    def __init__(self, pos: Point, rotation: Vector, **params):
        """
        Стандартная инициализация плоскости + поиск двух направляющих
        и ортогональных векторов плоскости.
        
        :param pos:
        :param rotation:
        :param params:
        """
        super().__init__(pos, rotation, **params)
        
        """
        Нахождения направляющих ортогональных векторов плоскости:
        
        1. Выбираем произвольный вектор, лежащий в плоскости.
           Например, можно взять вектор (1, 0, -A / C),
           если C не равно 0.
           Если C равно 0, то можно взять вектор (0, 1, -B / A),
           если A не равно 0.
           
           ?????????????????????????
           A = yp1 * zp2 - zp1 * yp2
           B = -xp1 * zp2 + zp1 * xp2
           C = xp1 * yp2 - yp1 * xp2
           
           xp1 = B, zp1 = 0:
           A(xp1 - x0) + B(yp1 - y0) + -C * z0 = 0
           A * xp1 + B * yp1 = A * x0 + B * y0 + C * z0
           yp1 = (A * x0 + B * y0 + C * z0 - A * B) / B =
               = x0 * (A / B) + y0 + (C / B) * z - A

        2. Находим векторное произведение вектора нормали
           и произвольного вектора:

           v11 = [n(A, B, C) x (1, 0, -A / C)] =
           = (B * (-A / C), A^2 / C + C, -B)
           
           v21 = [n(A, B, C) x (0, 1, -B / A)] = (-B^2 / A - C, B, A)
           
           v2 = [n(A, B, C) x v1(B, -A, 0)] =
           = (A * C, -C * B, -A^2 - B^2)
           
           v1 = (B, -A, 0)
           v2 = (C * B, -C * A, A^2 + B^2)

        3. Нормализуем полученные векторы:

           v1_norm = v1 / |v1|
           v2_norm = v2 / |v2|

        4. Получаем направляющие ортогональные векторы плоскости:

           n1 = v1_norm
           n2 = n1 x n
           где x - векторное произведение векторов.
        """
        a, b, _ = self.rot.point.coords
        self.v1 = Vector(b, -a, 0).norm()
        self.v2 = self.v1 ** self.rot
        self.v2 = self.v2.norm()

    def in_boundaries(self, pt: Point) -> bool:
        """
        Проверка координат точки на соответствие границам плоскости.
        
        :param pt: Точка
        :return:
        """
        corner = self.v1 * self.pr['dv1'] + self.v2 * self.pr['dv2']
        delta_x, delta_y, delta_z = corner.point.coords
        return abs(pt.coords[0] - self.pos.coords[0]) <= delta_x \
            and abs(pt.coords[1] - self.pos.coords[1]) <= delta_y \
            and abs(pt.coords[2] - self.pos.coords[2]) <= delta_z

    def contains(self, pt: Point) -> bool:
        # s = sum(self.rot.point.coords[i] *
        #         (pt.coords[i] - self.pos.coords[i]) for i in range(3))
        if self.in_boundaries(pt):
            return self.rot * Vector(pt - self.pos) == 0
        
        return False

    def intersect(self, v: Vector, pt_begin: Point = Vector.vs.init_pt):
        """
        
        :param v: Радиус-вектор отрезка
        :param pt_begin: Точка начала вектора v
        :return:
        """
        if self.rot * v != 0 and not self.contains(v.point):
            t0 = (self.rot * Vector(self.pos) -
                  self.rot * Vector(pt_begin)) / (self.rot * v)
            if 0 <= t0 <= 1 and self.in_boundaries((v * t0).point):
                return v * t0
            
        elif self.contains(v.point):
            """HeLp mE"""
    
        return Vector.vs.init_pt

    def nearest_point(self, *pts: Point) -> Point:
        """I just go out through the window"""


class Sphere(Object):
    def contains(self, pt: Point) -> bool:
        """
        x**2 + y**2 + z**2 <= params.radius
        
        :param pt:
        :return:
        """
        return pt.coords[0] ** 2 + pt.coords[1] ** 2 + \
            pt.coords[2] ** 2 <= self.pr['radius']
    
    def intersect(self, v: Vector, pt2: Point = Vector.vs.init_pt):
        """
        
        :param pt2:
        :param v:
        :return:
        """
        
        """
        Coming soon...
        """
        pass
    
    def nearest_point(self, *pts: list[Point]) -> Point:
        """We can live"""
        
        
class Cube(Object):
    """
    y = centr.y + params.delta_y
    
    {(centr.x + delta_x) + (centr.z + delta_z) = -y;
     centr.x - delta_x <= pt.x <= centr.x + delta_x
     }
    Или
    
    Уравнение грани (определитель)
     | (x - centr.x) (y - centr.y + delta_y) (z - centr.z) |
     | delta_x       0                       0             |
     | 0             0                       delta_z       |
     
    Ограничения плоскости (расположения точки)
    centr.x - delta_x <= pt.x <= centr.x + delta_x
    
    6 ограниченных плоскостей
    """
    def contains(self, pt: Point) -> bool:
        """
        
        :param pt:
        :return:
        """
        pass
    
    def intersect(self, v: Vector, pt_begin: Point) -> Point:
        """
        God, save us
        
        :param v:
        :param pt_begin:
        :return:
        """
        pass
    
    def nearest_point(self, *pts: list[Point]) -> Point:
        """
        bruh...
        
        :param pts:
        :return:
        """
        pass


if __name__ == "__main__":
    vs = VectorSpace()
    Vector.vs = vs  # Передача векторного пространства в классы
    p1 = Point(1, 2, 3)
    p2 = Point(3, 2, 1)
    v1 = Vector(p1)
    v2 = Vector(p2)
    # print(v1 ** v2)
    # print(v1.len())
    
    pln = Plane(Point(1, 1, 1), Vector(1, 0, 0))
    print(pln.contains(p1))
    """
    2y - z = 11
    x = any
    y = (11 + z) / 2
    
    """
