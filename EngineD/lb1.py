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
        pass
    
    @abstractmethod
    def intersect(self, v: Vector) -> Point:
        """
        Точка пересечения или выходящая за
        поле видимости (дальности прорисовки) (draw_distance)
        
        :param v:
        :return:
        """
        pass
    
    @abstractmethod
    def nearest_point(self, *pts: list[Point]) -> Point:
        pass


class Plane(Object):
    """
    Плоскость

    Координаты, направляющая плоскость
    
    Параметрическое уравнение плоскости
    {x = a * t + b * s + c
     y = f * t + g * s + h
     z = p * t + q * s + v
    }
    
    a, b, f, g, p, q - направляющие вектора
    c, h, v - константы (точки?)
    
    t и s - параметры (константы)
    """
    def contains(self, pt: Point) -> bool:
        return sum(self.rot.point.coords[i] *
                   (pt.coords[i] - self.pos.coords[i])
                   for i in range(3)) == 0
    
    def intersect(self, v: Vector):
        """
        Скалярное произведение вектора нормали к плоскости и вектора v
        не равно нулю и вектор v не лежит на плоскости, то вектор v
        не пересекает плоскость.
        
        Если координаты вектора v соответствуют уравнению плоскости,
        то этот вектор лежит на плоскости.
        
        Иначе, если скалярное произведение вектора нормали к плоскости
        и вектора v равно нулю и вектор v не лежит на плоскости,
        то этот вектор параллелен ей и не пересекает е ни в какой точке.
        
        :param v:
        :return:
        """
        if self.rot * v != 0 and not self.contains(v.point):
            """
            Каноническое уравнение прямой, на которой лежит вектор v:
            x / v.point.coords[0] = y / v.point.coords[1]
            = z / v.point.coords[2]
            
            Параметрическое уравнение той же прямой:
            x = v.point.coords[0] * t0 - 0
            y = v.point.coords[1] * t0 - 0
            z = v.point.coords[2] * t0 - 0
            
            Подставляем параметрические значения в уравнение плоскости:
            self.rot.point.coords[0] * (v.point.coords[0] * t0
                                        - 0 - self.pos.coords[0]) +
            self.rot.point.coords[1] * (v.point.coords[1] * t0
                                        - 0 - self.pos.coords[1]) +
            self.rot.point.coords[2] * (v.point.coords[2] * t0
                                        - 0 - self.pos.coords[2]) = 0
            
            Ищем параметр t0:
            xv, yv, zv - координаты вектора v
            xp, yp, zp - координаты точки, лежащей в плоскости
            A, B, C - координаты вектора нормали к плоскости
            
            A * xv * t0 - A * xp + B * yv * t0 - B * yp +
            C * zv * t0 - C * zp = 0
            
            t0 * (A * xv + B * yv + C * zv) = A * xp + B * yp * C * zp
            
            t0 = (A * xp + B * yp * C * zp) / (A * xv + B * yv + C * zv)
            
            t0 = (self.rot * self.pos) / (self.rot * v)
            
            Подставляем значение параметра в параметрическое
            уравнение прямой и находим координаты точки пересечения:
            Q = self.rot.point * t0 - vs.init_pt
            Q(self.rot.point.coords[0] * t0 - 0,
              self.rot.point.coords[1] * t0 - 0,
              self.rot.point.coords[2] * t0 - 0) - точка пересечения
            """
            t0 = (self.rot * self.pos) / (self.rot * v)
            if 0 <= t0 <= 1:
                return v * t0
        elif self.contains(v.point):
            return v
        
        return False
    
    def nearest_point(self, *pts: Point) -> Point:
        r_min = 10 ** 9
        for pt in pts:
            r = abs(self.rot * (pt - self.pos)) / self.rot.len()
            r_min = min(r_min, r)
        
        return r_min


class BoundedPlane(Plane):
    """
    Ограниченная плоскость
    
    delta_t
    delta_s
    
    -delta_t <= t <= delta_t - ограничения для параметра t
    -delta_s <= s <= delta_s - ограничения для параметра s
    """
    
    def in_boundaries(self, pt: Point) -> bool:
        """
        Проверка координат точки на соответствие границам плоскости.
        
        :param pt:
        :return:
        """
        return self.pr['x_lims'][0] <= pt.coords[0] <= \
            self.pr['x_lims'][1] \
            and self.pr['y_lims'][0] <= pt.coords[1] <= \
            self.pr['y_lims'][1] \
            and self.pr['z_lims'][0] <= pt.coords[2] <= \
            self.pr['z_lims'][1]

    def contains(self, pt: Point) -> bool:
        s = sum(self.rot.point.coords[i] *
                (pt.coords[i] - self.pos.coords[i]) for i in range(3))
        
        if self.in_boundaries(pt):
            return s == 0
        
        return False

    def intersect(self, v: Vector):
        if self.rot * v != 0 and not self.contains(v.point):
            t0 = (self.rot * self.pos) / (self.rot * v)
            if 0 <= t0 <= 1 and self.in_boundaries((v * t0).point):
                return v * t0
        elif self.contains(v.point):
            vb = Vector(v.point)  # копия вектора
            # Дальше идёт какой-то кошмар
            if vb.point.coords[0] < self.pr['x_lims'][0]:
                vb.point.coords[0] = vb.point.coords[0] \
                                     - self.pr['x_lims'][0]
            
            if vb.point.coords[0] > self.pr['x_lims'][1]:
                vb.point.coords[0] = self.pr['x_lims'][1] \
                                     - vb.point.coords[0]

            if vb.point.coords[1] < self.pr['y_lims'][0]:
                vb.point.coords[1] = vb.point.coords[1] \
                                     - self.pr['y_lims'][0]

            if vb.point.coords[1] > self.pr['y_lims'][1]:
                vb.point.coords[1] = self.pr['y_lims'][1] \
                                     - vb.point.coords[1]

            if vb.point.coords[2] < self.pr['z_lims'][0]:
                vb.point.coords[2] = vb.point.coords[2] \
                                     - self.pr['z_lims'][0]

            if vb.point.coords[2] > self.pr['z_lims'][1]:
                vb.point.coords[2] = self.pr['z_lims'][1] \
                                     - vb.point.coords[2]
            
            if vb.point == Point(0, 0, 0):
                if self.contains(vb.vs.init_pt):
                    return vb.vs.init_pt
                elif self.contains(v.point):
                    return v.point
                
                # Ищем, какой из вершин плоскости касается прямая
                # (смотрим 8 или 4 вершины, подставляем
                # в уравнение прямой и учитываем ограничения вектора)
            
            return v
    
        return False

    def nearest_point(self, *pts: Point) -> Point:
        r_min = 10 ** 9
        for pt in pts:
            r = abs(self.rot * (pt - self.pos)) / self.rot.len()
            r_min = min(r_min, r)
    
        return r_min


class Sphere(Object):
    def contains(self, pt: Point) -> bool:
        """
        x**2 + y**2 + z**2 <= params.radius
        
        :param pt:
        :return:
        """
        return pt.coords[0] ** 2 + pt.coords[1] ** 2 + \
            pt.coords[2] ** 2 <= self.pr['radius']
    
    def intersect(self, v: Vector) -> Point:
        """
        
        :param v:
        :return:
        """
        """
        A = (1, 2, 3);
        B = (-9, 6, 7);
        O = (5, 8, -4);
        r = 10;
        / *Расчет
        точек
        пересечения
        сферы(O, r)
        и
        прямой
        AB * /
        print
        "Вектор V нормали к плоскости ABO через векторное произведение
         векторов этой плоскости:";
        print
        "V=", V = (B - O)
        vert(A - O);
        print;
        print
        "Длина отрезка AB и расстояние h от центра O до прямой AB:";
        print
        "AB=", AB = abs(B - A);
        print
        "h=", h = abs
        V / AB;
        print;
        print
        "Единичный вектор OP перпендикуляра из центра O на прямую AB через векторное произведение векторов V и AB:";
        OP = V
        vert(B - A);
        print
        "OP=", OP = OP / abs
        OP; / *нормализация
        вектора
        OP * /
        print;
        print
        "Точка P - основание перпендикуляра из центра O на прямую AB (ближайшая к центру O точка прямой AB):";
        print
        "P=", P = O + OP * h;
        print;
        print
        "Расстояние d от точки P до точек пересечения сферы и прямой, вычисляемое по теореме Пифагора";
        print
        "(в случае мнимого d сфера и прямая не пересекаются):";
        print
        "d=", d = sqrt(r ^ 2 - h ^ 2);
        print;
        print
        "Вектор Pd на прямой AB, имеющий длину d:";
        print
        "Pd=", Pd = (B - A) / AB * d;
        print;
        print
        "Искомые точки P1 и P2 пересечения сферы и прямой:";
        print
        "P1=", P1 = P + Pd;
        print
        "P2=", P2 = P - Pd;
        print;
        / *Проверка
        расчета * /
        print
        "Проверка расстояний от точек P1, P2 до центра O
        (должен дважды получиться радиус сферы r):";
        print
        abs(P1 - O);
        print
        abs(P2 - O);
        print;
        print
        "Проверка принадлежности точек P1, P2 прямой AB
        (должно дважды получиться число 1):";
        print
        abs((P1 - A) / abs(P1 - A) * (B - A) / AB);
        print
        abs((P2 - A) / abs(P2 - A) * (B - A) / AB);
        """


class Cube(Object):
    def contains(self, pt: Point) -> bool:
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
        
        :param pt:
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
