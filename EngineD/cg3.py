import configparser
import math
import numpy as np
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
        if self.len() == 0:
            return self
        
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


Vector.vs = VectorSpace()


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
    
    def __str__(self):
        return f"Ray({self.inpt}, {self.dir})"
    
    def intersect(self, mapping: Map) -> list[float]:
        # пересечение с объектами, вызов intersect-ов
        # в объектах внутри mapping
        # intersect возвращает расстояние, а не точку,
        # и принимает Ray, а не Vector и Point
        return [objt.intersect(self) for objt in mapping]


class Camera:
    config = configparser.ConfigParser()
    config.read("config.cfg")
    hight = int(config['SCREEN']['width'])
    width = int(config['SCREEN']['hight'])
    ratio = width / hight
    
    def __init__(self, position: Point, look_dir: Vector,
                 fov, draw_dist):
        """
        vfov = fov * H / W - вертикальный угол наклона

        :param position: Расположение.
        :param look_at: Направление камеры (Point).
        :param look_dir: Углы наклона камеры (Vector).
        :param fov: Горизонтальный угол наклона.
        :param draw_dist: Дистанция прорисовки.
        """
        self.pos = position
        # self.look_at = look_at
        self.look_dir = look_dir.norm()
        self.fov = fov
        self.vfov = fov / self.ratio
        self.screen = BoundedPlane(self.pos + self.look_dir.point,
                                   self.look_dir, 1, self.ratio)
        
        """
        Видимость ограничена сферой с длиной радиуса Draw_distance
        Экран - проекция сферы на плоскость,
        зависящей от self.pos и look_dir.
        """
        self.draw_dist = draw_dist
    
    def send_rays(self) -> list[list[Ray]]:
        # считает расстояние от камеры до пересечения луча с объектами
        rays = []
        # t_step = 2 / self.width
        # s_step = 2 / self.ratio / self.hight
        # t = -self.screen.dv
        # s = -self.screen.du
        # Создаём лучи к каждому пикселю
        for i, t in enumerate(np.linspace(
            -self.screen.du, self.screen.du, self.width)):
            rays.append([])
            for s in np.linspace(-self.screen.dv, self.screen.dv,
                                 self.hight):
                direction = Vector(self.screen.pos) + self.screen.v * s \
                            + self.screen.u * t
                rays[i].append(Ray(
                    direction.point - self.look_dir.point,
                    self.look_dir))
                # t += t_step
                # s += s_step
        
        return rays


class Object:
    def __init__(self, pos: Point, rotation: Vector):
        self.pos = pos
        self.rot = rotation
    
    @abstractmethod
    def contains(self, pt: Point, eps=1e-6) -> bool:
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
        """
        Углы Эйлера

        Матрица поворота
        R = (cos alpha -sin alpha)
            (sin alpha  cos alpha)

        V2 = R * V1 (V1, V2 - вектора в двумерном пространстве)

        Rout - крен (ось Ox)
        Pitch - тангаж (ось Oy)
        Yaw - рыскание (ось Oz, нормаль к "самолёту")

        Ryz = (1    0           0        )            (от y к z)
              (0    cos alpha  -sin alpha)
              (0    sin alpha   cos alpha)

        Rxz = ( cos alpha   0   sin alpha) (от x к z в обратную сторону)
              ( 0           1   0        )
              (-sin alpha   0   cos alpha)

        Rxy = (cos alpha   -sin alpha   0)            (от x к y)
              (sin alpha    cos alpha   0)
              (0            0           1)

        R(x_angle, y_angle, z_angle) =
        = Rx(x_angle) * Ry(y_angle) * Rz(z_angle)

        R(alpha, pi/2, gamma)= (0                 0                1)
                               ( sin(alpha+gamma) cos(alpha+gamma) 0)
                               (-cos(alpha+gamma) sin(alpha+gamma) 0)

        Блокировка осей

        :param x_angle:
        :param y_angle:
        :param z_angle:
        :return:
        """
        x_angle = x_angle / 180 * math.pi
        y_angle = y_angle / 180 * math.pi
        z_angle = z_angle / 180 * math.pi
        
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
                edge.pr.pos = self.pos + rotations[i // 2].point
                edge._update()
            else:
                edge.pr.pos = self.pos - rotations[i // 2].point
                edge._update()
    
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
    
    def contains(self, pt: Point, eps=1e-6) -> bool:
        # return sum(self.rot.point.coords[i] *
        #            (pt.coords[i] - self.pos.coords[i])
        #            for i in range(3)) == 0
        self._update()
        return abs(self.rot * Vector(pt - self.pos)) < eps
    
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
    
    def __init__(self, pos: Point, rotation: Vector, du, dv):
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
        
        if self.rot.point == x_dir.point:
            x_dir = Vector.vs.basis[2]
        
        self.u = (self.rot ** x_dir).norm()
        self.v = (self.u ** self.rot).norm()
        
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
               f' du={self.du}, dv={self.dv})'
    
    def in_boundaries(self, pt: Point) -> bool:
        """
        Проверка координат точки на соответствие границам плоскости.

        :param pt: Точка
        :return:
        """
        self._update()
        corner = self.u * self.du + self.v * self.dv
        delta_x, delta_y, delta_z = corner.point.coords
        # print('This is here', corner.point)
        return abs(pt.coords[0] - self.pos.coords[0]) <= abs(delta_x) \
            and abs(pt.coords[1] - self.pos.coords[1]) <= abs(delta_y) \
            and abs(pt.coords[2] - self.pos.coords[2]) <= abs(delta_z)
    
    def contains(self, pt: Point, eps=1e-6) -> bool:
        self._update()
        if self.in_boundaries(pt):
            return abs(self.rot * Vector(pt - self.pos)) < 0
        
        return False
    
    def intersect(self, ray: Ray) -> float or None:
        """

        :param ray:
        :return:
        """
        self._update()
        if self.rot * ray.dir != 0 and \
            not (self.rot * Vector(ray.inpt - self.pos) == 0
                 and self.rot * Vector(ray.dir.point + ray.inpt
                                       - self.pos) == 0):
            if self.contains(ray.inpt):
                return 0
            
            t0 = (self.rot * Vector(self.pos) -
                  self.rot * Vector(ray.inpt)) / (self.rot * ray.dir)
            int_pt = ray.dir.point * t0 + ray.inpt
            if t0 >= 0 and self.in_boundaries(int_pt):
                return int_pt.distance(ray.inpt)
        
        elif self.rot * Vector(
            ray.dir.point + ray.inpt - self.pos) == 0:
            # Проекции вектора из точки центра плоскости
            # к точке начала вектора v на направляющие вектора плоскости
            r_begin = Vector(ray.inpt - self.pos)
            # Если начало вектора совпадает с центром плоскости
            if r_begin.len() == 0:
                return 0
            
            begin_pr1 = r_begin * self.u * self.du / r_begin.len()
            begin_pr2 = r_begin * self.v * self.dv / r_begin.len()
            if abs(begin_pr1) <= 1 and abs(begin_pr2) <= 1:
                return 0
            
            # Проекции вектора из точки центра плоскости
            # к точке конца вектора v на направляющие вектора плоскости
            r_end = r_begin + ray.dir
            if r_end.len() == 0:
                if abs(begin_pr1) > 1 or abs(begin_pr2) > 1:
                    if begin_pr1 > 1:
                        begin_pr1 -= 1
                    elif begin_pr1 < 1:
                        begin_pr1 += 1
                    
                    if begin_pr2 > 1:
                        begin_pr2 -= 1
                    elif begin_pr2 < 1:
                        begin_pr2 += 1
                    
                    return begin_pr1 * self.du + begin_pr2 * self.dv
                
                return 0
            
            def find_point(ray1: Ray, ray2: Ray):
                """
                для прямой (луча) ray1
                x = t * ux + x0
                y = t * uy + y0
                z = t * uz + z0

                для прямой ray2
                x = s * vx + xr
                y = s * vy + yr
                z = s * vz + zr

                Поиск через x:
                t0 * ux + x0 = s0 * vx + xr
                s0 = (t0 * ux + x0 - xr) / vx

                t0 * uy + y0 = (t0 * ux + x0 - xr) * vy / vx + yr
                t0 * (uy - ux * vy / vx) = (x0 - xr) * vy / vx + yr - y0
                t0 = ((x0 - xr) * vy / vx + yr - y0) / (uy - ux * vy/vx)

                через y:
                s0 = (t0 * uy + y0 - yr) / vy

                to * ux + x0 = (t0 * uy + y0 - yr) * vx / vy + xr
                t0 * (ux - uy * vx / vy) = (y0 - yr) * vx / vy + xr - x0
                t0 = ((y0 - yr) * vx / vy + xr - x0) / (ux - uy * vx/vy)

                через z:
                t0 * uz + z0 = s0 * vz + zr
                s0 = (t0 * uz + z0 - zr) / vz

                t0 * uy + y0 = (t0 * uz + z0 - zr) * vy / vz + yr
                t0 * (uy - uz * vy / vz) = (z0 - zr) * vy / vz + yr - y0
                t0 = ((z0 - zr) * vy / vz + yr - y0) / (uy - uz * vy/vz)

                :param ray1:
                :param ray2:
                :return:
                """
                if ray1.dir.point.coords[0] != 0:
                    x0 = ray1.inpt.coords[0]
                    y0 = ray1.inpt.coords[1]
                    xr = ray2.inpt.coords[0]
                    yr = ray2.inpt.coords[1]
                    vx = ray1.dir.point.coords[0]
                    vy = ray1.dir.point.coords[1]
                    ux = ray2.dir.point.coords[0]
                    uy = ray2.dir.point.coords[1]
                    t1 = ((x0 - xr) * vy / vx + yr - y0) \
                         / (uy - ux * vy / vx)
                    s1 = (t1 * ux + x0 - xr) / vx
                    return t1, s1
                elif ray1.dir.point.coords[1] != 0:
                    x0 = ray1.inpt.coords[0]
                    y0 = ray1.inpt.coords[1]
                    xr = ray2.inpt.coords[0]
                    yr = ray2.inpt.coords[1]
                    vx = ray1.dir.point.coords[0]
                    vy = ray1.dir.point.coords[1]
                    ux = ray2.dir.point.coords[0]
                    uy = ray2.dir.point.coords[1]
                    t1 = ((y0 - yr) * vx / vy + xr - x0) \
                         / (ux - uy * vx / vy)
                    s1 = (t0 * uy + y0 - yr) / vy
                    return t1, s1
                elif ray1.dir.point.coords[2] != 0:
                    z0 = ray1.inpt.coords[2]
                    y0 = ray1.inpt.coords[1]
                    zr = ray2.inpt.coords[2]
                    yr = ray2.inpt.coords[1]
                    vz = ray1.dir.point.coords[2]
                    vy = ray1.dir.point.coords[1]
                    uz = ray2.dir.point.coords[2]
                    uy = ray2.dir.point.coords[1]
                    t1 = ((z0 - zr) * vy / vz + yr - y0) / (
                        uy - uz * vy / vz)
                    s1 = (t0 * uz + z0 - zr) / vz
                    return t1, s1
            
            if abs(begin_pr1) > 1:
                if self.u * ray.dir == 0:
                    return
                
                sign = 1 if begin_pr1 > 1 else -1
                t0, s0 = find_point(
                    Ray(sign * self.du * self.u.point + self.pos,
                        self.dv * self.v), ray)
                if s0 >= 0 and abs(t0) <= 1:
                    return s0 * ray.dir.len()
            
            elif abs(begin_pr2) > 1:
                if self.v * ray.dir == 0:
                    return
                
                sign = 1 if begin_pr2 > 1 else -1
                t0, s0 = find_point(
                    Ray(sign * self.dv * self.v.point + self.pos,
                        self.du * self.u), ray)
                if s0 >= 0 and abs(t0) <= 1:
                    return s0 * ray.dir.len()
            
            """
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
            v_tmp = r_end - r_begin - Vector(ray.inpt)
            return Plane(self.pos, self.rot).intersect(
                Ray(v_tmp, r_begin.point))
            """
    
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
    
    def contains(self, pt: Point, eps=1e-6) -> bool:
        """
        x**2 + y**2 + z**2 <= params.radius**2

        :param pt:
        :return:
        """
        self._update()
        return self.pos.distance(pt) - self.r <= eps
        # Vector(pt - self.pos) * Vector(pt - self.pos) <= self.r**2
    
    def intersect(self, ray: Ray) -> float or None:
        """

        :param ray:
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
        a = ray.dir * ray.dir
        b = 2 * ray.dir * Vector(ray.inpt - self.pos)
        c = Vector(self.pos) * Vector(self.pos) + \
            Vector(ray.inpt) * Vector(ray.inpt) \
            - 2 * Vector(self.pos) * Vector(ray.inpt) - self.r ** 2
        
        d = b ** 2 - 4 * a * c
        if d > 0:
            t1 = (-b + math.sqrt(d)) / (2 * a)
            t2 = (-b - math.sqrt(d)) / (2 * a)
            # Смотрим пересечения с поверхностью сферы
            if t1 < 0 <= t2 or 0 < t2 <= t1:
                return t2 * ray.dir.len()
            elif t2 < 0 <= t1 or 0 < t1 <= t2:
                return t1 * ray.dir.len()
        
        elif d == 0:
            t0 = -b / (2 * a)
            if t0 >= 0:
                return t0 * ray.dir.len()
    
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
        self.rot3 = (self.rot2 ** self.rot).norm()
        
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
    
    def _update(self):
        self.pos = self.pr.pos
        self.rot = self.pr.rot
        self.rot2 = self.pr.rot2
        self.rot3 = self.pr.rot3
        self.limit = self.pr.limit
        self.edges = self.pr.edges
    
    def __str__(self):
        self._update()
        s = ", ".join(map(str, [self.rot, self.rot2, self.rot3]))
        return f'Cube({self.pos}, ({s}), limit={self.limit:.4f})'
    
    def contains(self, pt: Point, eps=1e-6) -> bool:
        self._update()
        # Радиус-вектор из центра куба к точке
        v_tmp = Vector(pt - self.pos)
        # Если точка является центром куба
        if v_tmp.len() == 0:
            return True
        
        # Проекции вектора v_tmp на направляющие вектора куба
        rot1_pr = self.rot * v_tmp / v_tmp.len()
        rot2_pr = self.rot2 * v_tmp / v_tmp.len()
        rot3_pr = self.rot3 * v_tmp / v_tmp.len()
        return all(abs(abs(pr) - 1) <= eps
                   for pr in (rot1_pr, rot2_pr, rot3_pr))
    
    def intersect(self, ray: Ray) -> float or None:
        self._update()
        # Пересечения куба с лучом, имеющей направляющий вектор ray.dir
        # и начальную точку ray.inpt
        # int_pts = list(filter(lambda x: x is not None,
        #                       [edge.intersect(ray)
        #                        for edge in self.edges]))
        int_pts = []
        for edge in self.edges:
            r = edge.intersect(ray)
            if r is not None:
                int_pts.append(r)
        
        if len(int_pts):
            return min(int_pts)
        """
            # Пересечения луча с гранями куба
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
        """
    
    def nearest_point(self, *pts: Point) -> Point:
        r_min = sys.maxsize
        min_pt = Vector.vs.init_pt
        r = 0
        nearest = [edge.nearest_point(*pts) for edge in self.edges]
        print(*nearest)
        for i, near_pt in enumerate(nearest):
            r_begin = Vector(near_pt - self.edges[i].pos)
            # Если начало вектора совпадает с центром плоскости
            if r_begin.len() == 0:
                return near_pt
            
            projection1 = r_begin * self.edges[i].rot / r_begin.len()
            projection2 = r_begin * self.edges[i].u * self.edges[i].du \
                          / r_begin.len()
            projection3 = r_begin * self.edges[i].v * self.edges[i].dv \
                          / r_begin.len()
            sign = lambda x: 1 if x > 0 else -1
            if abs(projection2) <= 1 and abs(projection3) <= 1:
                r = projection1 * self.edges[i].rot.len()
            elif abs(projection2) > 1 and abs(projection3) > 1:
                proj2 = projection2 - sign(projection2)
                proj3 = projection3 - sign(projection3)
                r = self.edges[i].rot * -projection1 \
                    + self.edges[i].u * proj2 \
                    + self.edges[i].v * proj3 + Vector(near_pt)
                r = r.len()
            elif abs(projection2) > 1:
                proj2 = projection2 - sign(projection2)
                r = self.edges[i].rot * -projection1 \
                    + self.edges[i].u * proj2 + Vector(near_pt)
                r = r.len()
            elif abs(projection3) > 1:
                proj3 = projection3 - sign(projection3)
                r = self.edges[i].rot * -projection1 \
                    + self.edges[i].v * proj3 + Vector(near_pt)
                r = r.len()
            
            if r < r_min:
                r_min = r
                min_pt = near_pt
        
        return min_pt


symbols = (' ', '.', ',', '-', '+', '=', '*', '#', '%', '&')


class Canvas:
    """
    map
    camera
    vectorspace

    def draw() # первая отрисовка, вызов update

    def update() # возвращает матрицу расстояний из send_rays

    0 - min
    draw_distance - max
    count - длина списка символов
    delta = (max - min) / count
    """
    
    def __init__(self, objmap: Map, camera: Camera):
        self.map = objmap
        self.cam = camera
    
    def update(self):
        rays = self.cam.send_rays()
        dist_matrix = []
        for i in range(self.cam.width):
            dist_matrix.append([])
            for j in range(self.cam.hight):
                distances = rays[i][j].intersect(self.map)
                if all(d is None or d > self.cam.draw_dist
                       for d in distances):
                    dist_matrix[i].append(None)
                # elif min(filter(None, distances)) > self.cam.draw_dist:
                #     dist_matrix[i].append(None)
                else:
                    dist_matrix[i].append(
                        min(filter(lambda x: x is not None, distances)))
        
        return dist_matrix


class Console(Canvas):
    """
    Отрисовка символами матрицы
    Список символов [#, @, &, ?, j, i, ,, .]
    Конвертация матрицы расстояний в символы
    """
    
    def draw(self):
        dist_matrix = self.update()
        # min_d, max_d = min(elem for d in dist_matrix for elem in d),\
        #     max(elem for d in dist_matrix for elem in d)
        for y in range(len(dist_matrix)):
            # print(dist_matrix[y], end='')
            for x in range(len(dist_matrix[y])):
                if dist_matrix[y][x] is None:
                    print(symbols[0], end='')
                    continue
                
                try:
                    gradient = dist_matrix[y][x] / self.cam.draw_dist \
                               * (len(symbols) - 1)
                    
                    print(symbols[len(symbols) - round(gradient) - 1],
                          end='')
                except (IndexError, TypeError):
                    print(len(symbols) * dist_matrix[y][x]
                          / self.cam.draw_dist, dist_matrix[y][x])
                    raise
            
            print()
