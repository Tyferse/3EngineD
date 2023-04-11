import configparser
import math
import sys
from abc import abstractmethod


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


Vector.vs = VectorSpace()  # Передача векторного пространства в вектор


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


# ______________________________________________________________________
class Map:
    def __init__(self):
        self._obj_list = []
    
    def append(self, *objs):
        # добавление в список объектов
        self._obj_list.extend(objs)
    
    def __getitem__(self, item):
        return self._obj_list[item]
    
    def __iter__(self):
        return iter(self._obj_list)


class Ray:
    def __init__(self, ipt: Point, direction: Vector):
        self.inpt = ipt
        self.dir = direction
    
    def intersect(self, mapping: Map):
        # пересечение с объектами, вызов intersect-ов
        # в объектах внутри mapping
        # intersect возвращает расстояние, а не точку,
        # и принимает Ray, а не Vector и Point
        return [objt.intersect(self) for objt in mapping]


class Object:
    def __init__(self, pos: Point, rotation: Vector):
        self.pos = pos
        self.rot = rotation
    
    @abstractmethod
    def contains(self, pt: Point) -> bool:
        return False
    
    @abstractmethod
    def intersect(self, ray: Ray) -> float or None:
        """
        Точка пересечения или выходящая за
        поле видимости (дальности прорисовки) (draw_distance)

        :param ray:
        :return:
        """
        return None
    
    @abstractmethod
    def nearest_point(self, *pts: list[Point]) -> Point:
        pass


# ______________________________________________________________________
class Parameters:
    def __init__(self, pos: Point, rotation: Vector):
        self.pos = pos
        self.rot = rotation
    
    def move(self, move_to: Point):
        self.pos = self.pos + move_to
    
    def scaling(self, value):
        pass
    
    def rotate(self, x_angle, y_angle, z_angle):
        # Поворот вокруг оси Ox
        y_old = self.rot.point.coords[1]
        z_old = self.rot.point.coords[2]
        self.rot.point.coords[1] = y_old * math.cos(x_angle) \
                                   - z_old * math.sin(x_angle)
        self.rot.point.coords[2] = y_old * math.sin(x_angle) \
                                   + z_old * math.cos(x_angle)
        
        # Поворот вокруг оси Oy
        x_old = self.rot.point.coords[0]
        z_old = self.rot.point.coords[2]
        self.rot.point.coords[0] = x_old * math.cos(y_angle) \
                                   + z_old * math.sin(y_angle)
        self.rot.point.coords[2] = x_old * -math.sin(y_angle) \
                                   + z_old * math.cos(y_angle)
        
        # Поворот вокруг оси Oz
        x_old = self.rot.point.coords[0]
        y_old = self.rot.point.coords[1]
        self.rot.point.coords[0] = x_old * math.cos(z_angle) \
                                   - y_old * math.sin(z_angle)
        self.rot.point.coords[1] = x_old * math.sin(z_angle) \
                                   + y_old * math.cos(z_angle)


class BoundedPlaneParams(Parameters):
    def __init__(self, pos: Point, rotation: Vector,
                 u, v, du, dv):
        super().__init__(pos, rotation)
        self.u = u
        self.v = v
        self.du = du
        self.dv = dv
    
    def scaling(self, value):
        self.du = self.du * value
        self.dv = self.dv * value
    
    def rotate(self, x_angle, y_angle, z_angle):
        tmp = Parameters(self.pos, self.rot)
        tmp.rotate(x_angle, y_angle, z_angle)
        self.rot = tmp.rot
        
        tmp.rot = self.u
        tmp.rotate(x_angle, y_angle, z_angle)
        self.u = tmp.rot
        
        tmp.rot = self.v
        tmp.rotate(x_angle, y_angle, z_angle)
        self.v = tmp.rot


class SphereParams(Parameters):
    def __init__(self, pos: Point, rotation: Vector, radius):
        super().__init__(pos, rotation)
        self.r = radius
    
    def scaling(self, value):
        self.r = self.r * value


class CudeParams(Parameters):
    def __init__(self, pos: Point, limit, rotations: [Vector],
                 edges: '[BoundedPlane]'):
        super().__init__(pos, rotations[0])
        self.rot2, self.rot3 = rotations[1:]
        self.limit = limit
        self.edges = edges
    
    def move(self, move_to: Point):
        self.pos = self.pos + move_to
        
        for edge in self.edges:
            edge.pos = edge.pos + move_to
    
    def scaling(self, value):
        self.rot = self.rot * value
        self.rot2 = self.rot2 * value
        self.rot3 = self.rot3 * value
        rotations = [self.rot, self.rot2, self.rot3]
        self.limit *= value
        
        for i, edge in enumerate(self.edges):
            edge.pr.scaling(value)
            if i % 2 == 0:
                edge.pos = self.pos + rotations[i // 2].point
            else:
                edge.pos = self.pos - rotations[i // 2].point
    
    def rotate(self, x_angle, y_angle, z_angle):
        tmp = Parameters(self.pos, self.rot)
        tmp.rotate(x_angle, y_angle, z_angle)
        self.rot = tmp.rot
        
        tmp.rot = self.rot2
        tmp.rotate(x_angle, y_angle, z_angle)
        self.rot2 = tmp.rot
        
        tmp.rot = self.rot3
        tmp.rotate(x_angle, y_angle, z_angle)
        self.rot3 = tmp.rot
        
        rotations = [self.rot, self.rot2, self.rot3]
        for i, edge in enumerate(self.edges):
            if i % 2 == 0:
                edge.pos = self.pos + rotations[i // 2].point
            else:
                edge.pos = self.pos - rotations[i // 2].point
            
            edge.pr.rotate(x_angle, y_angle, z_angle)


# ______________________________________________________________________
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
    
    def __init__(self, position, rotation):
        super().__init__(position, rotation)
        self.pr = Parameters(self.pos, self.rot)
    
    def _update(self):
        self.pos = self.pr.pos
        self.rot = self.pr.rot
    
    def __str__(self):
        self._update()
        return f'Plane({self.pos}, {str(self.rot)})'
    
    def contains(self, pt: Point) -> bool:
        # return sum(self.rot.point.coords[i] *
        #            (pt.coords[i] - self.pos.coords[i])
        #            for i in range(3)) == 0
        self._update()
        return self.rot * Vector(pt - self.pos) == 0
    
    def intersect(self, ray: Ray) -> float:
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

        :param ray:
        :return: Точка пересечения или точка,
                 ближайшая к центру плоскости.
        """
        self._update()
        if self.rot * ray.dir != 0 and not (self.contains(ray.inpt)
                                            and self.contains(
                ray.dir.point)):
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
                  self.rot * Vector(ray.inpt)) / (self.rot * ray.dir)
            if t0 >= 0:
                return t0 * ray.dir.len()
        
        elif self.contains(ray.inpt):
            """
            Возвращаем ноль, потому что
            """
            return 0
    
    def nearest_point(self, *pts: Point) -> Point:
        self._update()
        r_min = sys.maxsize
        min_pt = Vector.vs.init_pt
        for pt in pts:
            r = abs(self.rot * Vector(pt - self.pos)) / self.rot.len()
            if r == 0:
                return pt
            
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
    
    def __init__(self, pos: Point, rotation: Vector, dv, du):
        """
        Стандартная инициализация плоскости + поиск двух направляющих
        и ортогональных векторов плоскости.

        :param pos:
        :param rotation:
        :param params:
        """
        super().__init__(pos, rotation)
        self.du = du
        self.dv = dv
        
        """
        Нахождения направляющих ортогональных векторов плоскости:

        1. Выбираем произвольный вектор, лежащий в плоскости.
           Например, можно взять вектор (1, 0, -A / C),
           если C не равно 0.
           Если C равно 0, то можно взять вектор (0, 1, -B / A),
           если A не равно 0.

        2. Находим векторное произведение вектора нормали
           и произвольного вектора:

           u = (B, -A, 0)
           v = (A * C, -C * B, -A^2 - B^2)

        3. Нормализуем полученные векторы:

           u_norm = u / |v1|
           v_norm = v / |v2|

        4. Получаем направляющие ортогональные векторы плоскости:

           n1 = u_norm
           n2 = n1 x n
           где x - векторное произведение векторов.
        """
        if abs(self.rot.point.coords[0]) \
            < abs(self.rot.point.coords[1]):
            x_dir = Vector.vs.basis[0]
        else:
            x_dir = Vector.vs.basis[1]
        
        self.u = (self.rot ** x_dir).norm()
        self.v = (self.rot ** self.u).norm()
        
        self.pr = BoundedPlaneParams(self.pos, self.rot,
                                     self.u, self.v, self.du, self.dv)
    
    def _update(self):
        self.pos = self.pr.pos
        self.rot = self.pr.rot
        self.u = self.pr.u
        self.v = self.pr.v
        self.du = self.pr.du
        self.dv = self.pr.dv
    
    def __str__(self):
        self._update()
        return f'Plane({self.pos}, {self.rot},' \
               f' dv1={self.du}, dv2={self.dv})'
    
    def in_boundaries(self, pt: Point) -> bool:
        """
        Проверка координат точки на соответствие границам плоскости.

        :param pt: Точка
        :return:
        """
        self._update()
        corner = self.u * self.du + self.v * self.dv
        delta_x, delta_y, delta_z = corner.point.coords
        return abs(pt.coords[0] - self.pos.coords[0]) <= abs(delta_x) \
            and abs(pt.coords[1] - self.pos.coords[1]) <= abs(delta_y) \
            and abs(pt.coords[2] - self.pos.coords[2]) <= abs(delta_z)
    
    def contains(self, pt: Point) -> bool:
        self._update()
        if self.in_boundaries(pt):
            return self.rot * Vector(pt - self.pos) == 0
        
        return False
    
    def intersect(self, v: Vector, pt_begin: Point = Vector.vs.init_pt):
        """

        :param v: Радиус-вектор отрезка
        :param pt_begin: Точка начала вектора v
        :return:
        """
        self._update()
        if self.rot * v != 0 and not (self.contains(pt_begin)
                                      and self.contains(v.point)):
            t0 = (self.rot * Vector(self.pos) -
                  self.rot * Vector(pt_begin)) / (self.rot * v)
            int_pt = v.point * t0 + pt_begin
            if 0 <= t0 <= 1 and self.in_boundaries(int_pt):
                return int_pt
        
        elif self.rot * Vector(v.point - self.pos) == 0:
            # Проекции вектора из точки центра плоскости
            # к точке начала вектора v на направляющие вектора плоскости
            r_begin = Vector(pt_begin - self.pos)
            # Если начало вектора совпадает с центром плоскости
            if r_begin.len() == 0:
                return self.pos
            
            begin_pr1 = r_begin * self.u / r_begin.len()
            begin_pr2 = r_begin * self.v / r_begin.len()
            
            # Проекции вектора из точки центра плоскости
            # к точке конца вектора v на направляющие вектора плоскости
            r_end = r_begin + v
            if r_end.len() == 0:
                return self.pos
            
            end_pr1 = r_end * self.u / r_end.len()
            end_pr2 = r_end * self.v / r_end.len()
            
            # Возвращаем координаты точки, ближайшей к центру,
            # если хотя бы часть вектора лежит в границах плоскости
            if begin_pr1 > self.du and end_pr1 > self.du \
                or begin_pr2 > self.dv and end_pr2 > self.dv:
                return Vector.vs.init_pt
            
            # Ограничение вектора плоскостью
            def value_limit(value, lim):
                if value < -lim:
                    value = -lim
                elif value > lim:
                    value = lim
                
                return value
            
            begin_pr1 = value_limit(begin_pr1, self.du)
            begin_pr2 = value_limit(begin_pr2, self.dv)
            end_pr1 = value_limit(end_pr1, self.du)
            end_pr2 = value_limit(end_pr2, self.dv)
            
            r_begin = self.u * begin_pr1 + self.v * begin_pr2 \
                      + Vector(self.pos)
            r_end = self.u * end_pr1 + self.v * end_pr2 \
                    + Vector(self.pos)
            # Вектор v, ограниченный плоскостью
            v_tmp = r_end - r_begin - Vector(pt_begin)
            return Plane(self.pos, self.rot).intersect(v_tmp,
                                                       r_begin.point)
        
        return Vector.vs.init_pt
    
    def nearest_point(self, *pts: Point) -> Point:
        """
        Строим два перпендикуляра от точки к плоскости,
         и от этого перпендикуляра до ребра плоскости.
        Если ближайшая точка на плоскости - вершина,
        то строим три перпендикуляра, первый к плоскости,
        второй от первого к первой границе
        (продолжению прямой, содержащей границу плоскости),
        третий - от второго перпендикуляра до второй границы плоскости.

        :param pts:
        :return:
        """
        self._update()
        r_min = sys.maxsize
        min_pt = Vector.vs.init_pt
        r = 0
        for pt in pts:
            r_begin = Vector(pt - self.pos)
            # Если начало вектора совпадает с центром плоскости
            if r_begin.len() == 0:
                return pt
            
            projection1 = r_begin * self.rot / r_begin.len()
            projection2 = r_begin * self.u * self.du / r_begin.len()
            projection3 = r_begin * self.v * self.dv / r_begin.len()
            sign = lambda x: 1 if x > 0 else -1
            if abs(projection2) <= 1 and abs(projection3) <= 1:
                r = projection1 * self.rot.len()
            elif abs(projection2) > 1 and abs(projection3) > 1:
                proj2 = projection2 - sign(projection2)
                proj3 = projection3 - sign(projection3)
                r = self.rot * -projection1 + self.u * proj2 \
                    + self.v * proj3 + Vector(pt)
                r = r.len()
            elif abs(projection2) > 1:
                proj2 = projection2 - sign(projection2)
                r = self.rot * -projection1 + self.u * proj2 \
                    + Vector(pt)
                r = r.len()
            elif abs(projection3) > 1:
                proj3 = projection3 - sign(projection3)
                r = self.rot * -projection1 + self.v * proj3 \
                    + Vector(pt)
                r = r.len()
            
            if r < r_min:
                r_min = r
                min_pt = pt
        
        return min_pt


class Sphere(Object):
    def __init__(self, pos: Point, rotation: Vector, radius):
        super().__init__(pos, rotation)
        self.pr = SphereParams(self.pos, self.rot.norm() * radius,
                               radius)
    
    def _update(self):
        self.pos = self.pr.pos
        self.rot = self.pr.rot
        self.r = self.pr.r
    
    def __str__(self):
        self._update()
        return f'Sphere({self.pos}, {str(self.rot)}, radius={self.r})'
    
    def contains(self, pt: Point) -> bool:
        """
        x**2 + y**2 + z**2 <= params.radius**2

        :param pt:
        :return:
        """
        self._update()
        return self.pos.distance(pt) <= self.r
        # Vector(pt - self.pos) * Vector(pt - self.pos) <= self.r**2
    
    def intersect(self, v: Vector, pt_begin: Point = Vector.vs.init_pt):
        """

        :param pt_begin:
        :param v:
        :return:
        """
        
        """
        Для определения координат точек пересечения сферы и вектора
        необходимо решить систему уравнений,
        состоящую из уравнения сферы и уравнения прямой,
        заданной вектором.
        Уравнение сферы имеет вид:

        (x - x0)**2 + (y - y0)**2 + (z - z0)**2 = r**2

        Уравнение прямой, заданной вектором,
        имеет параметрическое представление:

        x = t * xv + x0
        y = t * yv + y0
        z = t * zv + z0

        где (x0, y0, z0) - координаты начальной точки вектора,
        (xv, yv, zv) - координаты направляющего вектора, t - параметр.

        Подставив параметрическое представление прямой
        в уравнение сферы, получим квадратное уравнение
        относительно параметра t:

        xc, yc, zc - координаты центра сферы

        (t * xv + x0 - xc) * (t * xv + x0 - xc) = (t * xv)**2
        + t * xv * x0 - t * xv * xc + t * xv * x0 + x0**2
        - x0 * xc - t * xv * xc - x0 * xc + xc**2 = xv**2 * t**2
        + (2 * xv * x0 - 2 * xv * xc) * t
        + (x0**2 - 2 * x0 * xc + xc**2) = xv**2 * t**2 +
        2 * (xv * (x0 - xc)) * t + (x0**2 + xc**2 - 2 * x0 * xc)

        (xv**22 + yv**2 + zv**2) * t**2 +
        + 2 * (xv * (x0 - xc) + yv * (y0 - yc) + zv * (z0 - zc)) * t +
        + (x0**2 + y0**2 + z0**2 + xc**2 + yc**2 + zc**2 -
         - 2 * (x0 * xc + y0 * yc + z0 * zc) - r**2) = 0

        Решив это уравнение относительно параметра t,
        найдем координаты точек пересечения сферы и вектора:

        t1 = (-b + sqrt(b**2 - 4 * a * c)) / 2 * a
        t2 = (-b - sqrt(b**2 - 4 * a * c)) / 2 * a

        где a = xv**2 + yv**2 + zv**2,
            b = 2 * (xv * (x0 - xc) + yv * (y0 - yc) + zv * (z0 - zc)),
            c = x0**2 + y0**2 + z0**2 + xc**2 + yc**2 + zc**2 -
                - 2 * (x0 * xc + y0 * yc + z0 * zc) - r**2.

        Подставив найденные значения параметра t
        в параметрическое представление прямой,
        получим координаты точек пересечения:

        x1 = x0 + t1 * vx
        y1 = y0 + t1 * vy
        z1 = z0 + t1 * vz

        x2 = x0 + t2 * vx
        y2 = y0 + t2 * vy
        z2 = z0 + t2 * vz

        Таким образом, для определения координат точек пересечения
        сферы и вектора необходимо решить квадратное уравнение
        относительно параметра t и подставить найденные значения
        параметра в параметрическое представление прямой.
        """
        self._update()
        a = v * v
        b = 2 * v * Vector(pt_begin - self.pos)
        c = Vector(self.pos) * Vector(self.pos) + \
            Vector(pt_begin) * Vector(pt_begin) \
            - 2 * Vector(self.pos) * Vector(pt_begin) - self.r ** 2
        
        d = b ** 2 - 4 * a * c
        if d > 0:
            t1 = (-b + math.sqrt(d)) / (2 * a)
            t2 = (-b - math.sqrt(d)) / (2 * a)
            # Смотрим пересечения с поверхностью сферы
            if 0 <= t1 <= 1:
                return v.point * t1 + pt_begin
            elif 0 <= t2 <= 1:
                return v.point * t2 + pt_begin
            
            # Если вектор лежит внутри сферы
            if (0 <= t1) != (0 <= t2) and (t1 <= 1) != (t2 <= 1):
                # Вектор, соединяющий точку центра
                # и точку начала вектора v
                v_tmp = Vector(self.pos - pt_begin)
                # Проекция вектора v_temp на вектор V
                projection = v * v_tmp / v_tmp.len()
                # Возвращаем координаты точки, ближайшей к центру
                if 0 <= projection <= 1:
                    return projection * v.point + pt_begin
                elif projection > 1:
                    return v.point
                else:
                    return pt_begin
        
        elif d == 0:
            t0 = -b / (2 * a)
            if 0 <= t0 <= 1:
                return v.point * t0 + pt_begin
        
        return Vector.vs.init_pt
    
    def nearest_point(self, *pts: Point) -> Point:
        self._update()
        r_min = sys.maxsize
        min_pt = Vector.vs.init_pt
        for pt in pts:
            r = self.pos.distance(pt)
            if r == 0:
                return pt
            
            if r < r_min:
                r_min = r
                min_pt = pt
        
        return min_pt


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
    
    def __init__(self, pos: Point, rotation: Vector):
        super().__init__(pos, rotation)
        # Ограничения размеров куба (половина длина ребра)
        self.limit = self.rot.len()
        
        # Ещё два ортогональных вектора из центра куба длины self.limit
        if abs(self.rot.point.coords[0]) \
            < abs(self.rot.point.coords[1]):
            x_dir = Vector.vs.basis[0]
        else:
            x_dir = Vector.vs.basis[1]
        
        self.rot2 = (self.rot ** x_dir).norm()
        self.rot3 = (self.rot ** self.rot2).norm()
        
        # Создание граней куба
        self.edges = []
        for v in self.rot, self.rot2, self.rot3:
            self.edges.append(BoundedPlane(v.point + self.pos, v,
                                           du=self.limit,
                                           dv=self.limit))
            self.edges.append(BoundedPlane(-1 * v.point + self.pos,
                                           -1 * v, du=self.limit,
                                           dv=self.limit))
        
        self.pr = CudeParams(self.pos, self.limit,
                             [self.rot, self.rot2, self.rot3],
                             self.edges)
        self.pos = self.pr.pos
        self.rot = self.pr.rot
        self.rot2 = self.pr.rot2
        self.rot3 = self.pr.rot3
        self.edges = self.pr.edges
    
    def __str__(self):
        s = ", ".join(map(str, [self.rot, self.rot2, self.rot3]))
        return f'Cube({self.pos}, ({s}), limit={self.limit * 2:.4f})'
    
    def contains(self, pt: Point) -> bool:
        # Радиус-вектор из центра куба к точке
        v_tmp = Vector(pt - self.pos)
        # Если точка является центром куба
        if v_tmp.len() == 0:
            return True
        
        # Проекции вектора v_tmp на направляющие вектора куба
        rot1_pr = self.rot * v_tmp / v_tmp.len()
        rot2_pr = self.rot2 * v_tmp / v_tmp.len()
        rot3_pr = self.rot3 * v_tmp / v_tmp.len()
        return all(abs(pr) <= 1 for pr in (rot1_pr, rot2_pr, rot3_pr))
    
    def intersect(self, v: Vector, pt_begin: Point) -> Point:
        # Пересечения куба с прямой, имеющей направляющий вектор v
        # и начальную точку pt_begin
        int_pts = []
        
        for edge in self.edges:
            # Пересечения прямой с гранями куба
            if edge.rot * v != 0 and not (edge.contains(pt_begin)
                                          and edge.contains(v.point)):
                t0 = (edge.rot * Vector(edge.pos) -
                      edge.rot * Vector(pt_begin)) / (edge.rot * v)
                int_pt = v.point * t0 + pt_begin
                if 0 <= t0 <= 1 and edge.in_boundaries(int_pt):
                    int_pts.append(int_pt)
                    break
            
            elif edge.rot * Vector(v.point - edge.pos) == 0:
                # Проекции вектора из точки центра плоскости
                # к точке начала вектора v
                # на направляющие вектора плоскости
                r_begin = Vector(pt_begin - edge.pos)
                # Если начало вектора совпадает с центром плоскости
                if r_begin.len() == 0:
                    int_pts.append(edge.pos)
                    break
                
                begin_pr1 = r_begin * edge.v1 / r_begin.len()
                begin_pr2 = r_begin * edge.v2 / r_begin.len()
                
                # Проекции вектора из точки центра плоскости
                # к точке конца вектора v
                # на направляющие вектора плоскости
                r_end = r_begin + v
                if r_end.len() == 0:
                    int_pts.append(edge.pos)
                    break
                
                end_pr1 = r_end * edge.v1 / r_end.len()
                end_pr2 = r_end * edge.v2 / r_end.len()
                
                # Возвращаем координаты точки, ближайшей к центру,
                # если хотя бы часть вектора лежит в границах плоскости
                if begin_pr1 > self.limit and end_pr1 > self.limit \
                    or begin_pr2 > self.limit \
                    and end_pr2 > self.limit:
                    break
                
                # Ограничение вектора плоскостью
                def value_limit(value, lim):
                    if value < -lim:
                        value = -lim
                    elif value > lim:
                        value = lim
                    
                    return value
                
                begin_pr1 = value_limit(begin_pr1, self.limit)
                begin_pr2 = value_limit(begin_pr2, self.limit)
                end_pr1 = value_limit(end_pr1, self.limit)
                end_pr2 = value_limit(end_pr2, self.limit)
                
                r_begin = edge.v1 * begin_pr1 + edge.v2 * begin_pr2 \
                          + Vector(edge.pos)
                r_end = edge.v1 * end_pr1 + edge.v2 * end_pr2 \
                        + Vector(edge.pos)
                # Вектор v, ограниченный плоскостью
                v_tmp = r_end - r_begin - Vector(pt_begin)
                return Plane(edge.pos, edge.rot) \
                    .intersect(v_tmp, r_begin.point)
        
        if len(int_pts):
            pass
        
        return Vector.vs.init_pt
    
    def nearest_point(self, *pts: list[Point]) -> Point:
        pass


class Canvas:
    """
    map
    camera
    vectorspace

    def draw() # первая отрисовка, вызов update

    def update() # возвращает матрицу расстояний из send_rays

    o - min
    draw_distance - max
    count - длина списка символов
    delta = (max - min) / count
    """
    pass


class Console(Canvas):
    """
    Отрисовка символами матрицы
    Список символов [#, @, &, ?, j, i, ,, .]
    Конвертация матрицы расстояний в символы
    """