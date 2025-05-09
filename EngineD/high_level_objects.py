import sys
from abc import abstractmethod

from basic_objects import *


class Object:
    """
    Базовый класс объекта в трёхмерном пространстве с точкой расположения (центра),
    вектором вращения (направления), операциями проверки точки
    на принадлежность внутренности (или поверхности) объекта,
    нахождения ближайшей точки пересечения луча с поверхностью,
    возврат точки, ближайшей к поверхности объекта.
    """
    def __init__(self, pos: Point, rotation: Vector):
        self.pos = pos
        self.rot = rotation
    
    @abstractmethod
    def contains(self, pt: Point, eps=1e-6) -> bool:
        """
        Проверка на принадлежность точки поверхности или внутренности объекта.
        
        Args:
            pt (Point): Точка, которая проверяется на принадлежность.
            eps (float): Максимальная ошибка округления (если расстояния
                от точки до объекта меньше eps, то вернёт True).

        Returns:
            bool: True, если точка pt принадлежит объекту, иначе False.
        """
        return False
    
    @abstractmethod
    def intersect(self, ray: Ray) -> float | None:
        """
        Наименьшее расстояние до точки пересечения луча с поверхностью объекта.
        
        Args:
            ray (Ray): Луч, для которого вычисляется пересечение с объектом.

        Returns:
            float: Наименьшее расстояние до поверхности объекта.
                Если нет пересечения с объектом, вернёт расстояние
                до границы области видимости камеры.
        """
        return None
    
    @abstractmethod
    def nearest_point(self, *pts: list[Point]) -> Point:
        """
        Возвращает ближайшую к объекту точку.
        
        Args:
            *pts (list[Point]): Список точек.

        Returns:
            Point: Точка, от которой наименьшее расстояние до объекта.

        """
        pass


# Определение параметров объектов __________________________________________________
class Parameters:
    """
    Класс параметров объекта с операциями перемещения, масштабирования и поворота
    """
    def __init__(self, pos: Point, rotation: Vector):
        self.pos = pos
        self.rot = rotation
    
    def move(self, move_to: Point):
        """
        Перемещает центр объекта покоординатно.
        
        Args:
            move_to (Point): Точка, на координаты которой надо изменить
                позицию объекта.
        """
        self.pos = self.pos + move_to
    
    def scaling(self, value: float | int):
        """
        Масштабирование объекта.
        
        Args:
            value (float | int): Масштабирующая константа, которая изменит
                размеры объекта.
        """
        pass
    
    def rotate(self, x_angle: float = 0, y_angle: float = 0, z_angle: float = 0):
        """
        Вращение объекта вокруг осей на передаваемые углы (в радианах).
        
        Args:
            x_angle: Угол поворота вокруг оси OX
            y_angle: Угол поворота вокруг оси OY
            z_angle: Угол поворота вокруг оси OZ
        """
        self.rot.rotate(x_angle, y_angle, z_angle)


class BoundedPlaneParams(Parameters):
    """Параметры ограниченной плоскости."""
    def __init__(self, pos: Point, rotation: Vector, u, v, du, dv):
        super().__init__(pos, rotation)
        self.u = u
        self.v = v
        self.du = du
        self.dv = dv
    
    def scaling(self, value):
        self.du *= value
        self.dv *= value
    
    def rotate(self, x_angle=0, y_angle=0, z_angle=0):
        self.rot.rotate(x_angle, y_angle, z_angle)
        self.u.rotate(x_angle, y_angle, z_angle)
        self.v.rotate(x_angle, y_angle, z_angle)


class SphereParams(Parameters):
    def __init__(self, pos: Point, rotation: Vector, radius):
        super().__init__(pos, rotation)
        self.r = radius
    
    def scaling(self, value):
        self.r = self.r * value


class CubeParams(Parameters):
    """Параметры куба"""
    def __init__(self, pos: Point, limit, rotations: [Vector],
                 edges: '[BoundedPlane]'):
        super().__init__(pos, rotations[0])
        self.rot2, self.rot3 = rotations[1:]
        self.limit = limit
        self.edges = edges
    
    def move(self, move_to: Point):
        self.pos = self.pos + move_to
        
        for i in range(len(self.edges)):
            self.edges[i].pos = self.edges[i].pos + move_to
    
    def scaling(self, value):
        self.rot = self.rot * value
        self.rot2 = self.rot2 * value
        self.rot3 = self.rot3 * value
        rotations = [self.rot, self.rot2, self.rot3]
        self.limit *= value
        
        for i in range(len(self.edges)):
            self.edges[i].pr.scaling(value)
            if i % 2 == 0:
                self.edges[i].pr.pos = self.pos + rotations[i // 2].point
            else:
                self.edges[i].pr.pos = self.pos - rotations[i // 2].point
    
    def rotate(self, x_angle=0, y_angle=0, z_angle=0):
        self.rot.rotate(x_angle, y_angle, z_angle)
        self.rot2.rotate(x_angle, y_angle, z_angle)
        self.rot3.rotate(x_angle, y_angle, z_angle)
        
        for i in range(len(self.edges)):
            self.edges[i].pr.rotate(x_angle, y_angle, z_angle)
            if i % 2 == 0:
                self.edges[i].pr.pos = self.pos + self.edges[i].pr.rot.point
            else:
                self.edges[i].pr.pos = self.pos + self.edges[i].pr.rot.point
            
            self.edges[i]._update()
            

# Определение объектов _____________________________________________________________
class Plane(Object):
    def __init__(self, position, rotation):
        """
        Плоскость без границ.
        
        Args:
            position (Point): Точка, принадлежащая плоскости.
            rotation (Vector): Вектор нормали к плоскости.
        """
        super().__init__(position, rotation)
        self.pr = Parameters(self.pos, self.rot)
    
    def _update(self):
        """Обновление текущих позиции и направления."""
        self.pos = self.pr.pos
        self.rot = self.pr.rot
    
    def __str__(self):
        self._update()
        return f'Plane({self.pos}, {str(self.rot)})'
    
    def contains(self, pt: Point, eps=1e-6) -> bool:
        self._update()
        return abs(self.rot * Vector(pt - self.pos)) < eps
    
    def intersect(self, ray: Ray) -> float:
        self._update()
        if (self.rot * ray.dir != 0
           and not (self.contains(ray.inpt) and self.contains(ray.dir.point))):
            t0 = ((self.rot * Vector(self.pos) - self.rot * Vector(ray.inpt))
                  / (self.rot * ray.dir))
            if t0 >= 0:
                return t0 * ray.dir.len()
        
        elif self.contains(ray.inpt):
            # Возвращаем ноль, потому что точка начала луча уже принадлежит плоскости
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
    def __init__(self, pos: Point, rotation: Vector, du, dv):
        """
        Плоскость, ограниченная длиной и шириной по двум лежащим на ней векторам.
        
        Args:
            pos (Point): Точка центра ограниченной плоскости, откуда исходят
                направляющие вектора
            rotation (Vector): Вектор нормали к плоскости.
            du (float | int): Длина первого (вертикального) вектора плоскости,
                отвечающего за длину.
            dv (float | int): Длина второго (горизонтального) вектора плоскости,
                отвечающего за ширину.
        """
        super().__init__(pos, rotation)
        self.du = du
        self.dv = dv
        
        y_dir = Vector.vs.basis[1]
        if self.rot.point == y_dir.point or self.rot.point == -1 * y_dir.point:
            y_dir = Vector.vs.basis[0]
        
        self.u = (self.rot ** y_dir).norm()
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
               f' du={self.du}, dv={self.dv})'
    
    def in_boundaries(self, pt: Point) -> bool:
        """
        Проверка координат точки на соответствие границам плоскости.
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
            return abs(self.rot * Vector(pt - self.pos)) < eps
        
        return False
    
    def intersect(self, ray: Ray) -> float or None:
        self._update()
        
        if self.rot * ray.dir != 0:
            if self.contains(ray.inpt):
                return 0
            
            t0 = (self.rot * Vector(self.pos) -
                  self.rot * Vector(ray.inpt)) / (self.rot * ray.dir)
            int_pt = ray.dir.point * t0 + ray.inpt
            if t0 >= 0 and self.in_boundaries(int_pt):
                return int_pt.distance(ray.inpt)
        
        elif self.rot * Vector(ray.dir.point
                               + ray.inpt - self.pos) == 0:
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
                    elif begin_pr1 < -1:
                        begin_pr1 += 1
                    
                    if begin_pr2 > 1:
                        begin_pr2 -= 1
                    elif begin_pr2 < -1:
                        begin_pr2 += 1
                    
                    return begin_pr1 * self.du + begin_pr2 * self.dv
                
                return 0
            
            def find_point(ray1: Ray, ray2: Ray):
                """
                Поиск точки пересечения двух прямых
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
                    
                    t1 = ((x0 - xr) * vy / vx + yr - y0) / (uy - ux * vy / vx)
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
                    
                    t1 = ((y0 - yr) * vx / vy + xr - x0) / (ux - uy * vx / vy)
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
                    
                    t1 = ((z0 - zr) * vy / vz + yr - y0) / (uy - uz * vy / vz)
                    s1 = (t0 * uz + z0 - zr) / vz
                    return t1, s1
            
            if abs(begin_pr1) > self.du:
                if self.u * ray.dir == 0:
                    return None
                
                sign = 1 if begin_pr1 > 0 else -1
                t0, s0 = find_point(
                    Ray(sign * self.du * self.u.point + self.pos, self.dv * self.v),
                    ray
                )
                if s0 >= 0 and abs(t0) <= 1:
                    return s0 * ray.dir.len()
            
            elif abs(begin_pr2) > self.dv:
                if self.v * ray.dir == 0:
                    return None
                
                sign = 1 if begin_pr2 > 0 else -1
                t0, s0 = find_point(
                    Ray(sign * self.dv * self.v.point + self.pos, self.du * self.u),
                    ray
                )
                if s0 >= 0 and abs(t0) <= 1:
                    return s0 * ray.dir.len()
    
    def nearest_point(self, *pts: Point) -> Point:
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
                r = (self.rot * -projection1 + self.u * proj2
                     + self.v * proj3 + Vector(pt))
                r = r.len()
            elif abs(projection2) > 1:
                proj2 = projection2 - sign(projection2)
                r = self.rot * -projection1 + self.u * proj2 + Vector(pt)
                r = r.len()
            elif abs(projection3) > 1:
                proj3 = projection3 - sign(projection3)
                r = self.rot * -projection1 + self.v * proj3 + Vector(pt)
                r = r.len()
            
            if r < r_min:
                r_min = r
                min_pt = pt
        
        return min_pt


class Sphere(Object):
    def __init__(self, pos: Point, rotation: Vector, radius: float | int):
        """
        Сфера с центром и радиусом.
        
        Args:
            pos (Point): Точка центра.
            rotation (Vector): Вектор направления (любой, так как сфера симметрична).
            radius (float): Радиус сферы.
        """
        super().__init__(pos, rotation)
        self.pr = SphereParams(self.pos, self.rot.norm() * radius, radius)
    
    def _update(self):
        self.pos = self.pr.pos
        self.rot = self.pr.rot
        self.r = self.pr.r
    
    def __str__(self):
        self._update()
        return f'Sphere({self.pos}, {str(self.rot)}, radius={self.r})'
    
    def contains(self, pt: Point, eps=1e-6) -> bool:
        self._update()
        return self.pos.distance(pt) - self.r <= eps
    
    def intersect(self, ray: Ray) -> float or None:
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
            if t2 < 0 <= t1 or 0 < t1 <= t2:
                return t1 * ray.dir.len()
            elif t1 < 0 <= t2 or 0 < t2 <= t1:
                return t2 * ray.dir.len()
        
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
    def __init__(self, pos: Point, rotation: Vector, size: float):
        """
        Куб с точкой центра и вектором направления.
        
        Args:
            pos (Point): Точка центра.
            rotation (Vector): Вектор направления.
            size (float): Длина ребра куба.
        """
        super().__init__(pos, rotation)
        # Ограничения размеров куба (половина длина ребра)
        self.limit = size / 2
        self.rot = rotation.norm() * self.limit
        
        # Ещё два ортогональных вектора из центра куба длины self.limit
        y_dir = Vector.vs.basis[1]
        if self.rot.point == y_dir.point \
           or self.rot.point == -1 * y_dir.point:
            y_dir = Vector.vs.basis[0]
        
        self.rot2 = (self.rot ** y_dir).norm() * self.limit
        self.rot3 = (self.rot ** self.rot2).norm() * self.limit
        
        # Создание граней куба
        self.edges = []
        for v in self.rot, self.rot2, self.rot3:
            self.edges.append(BoundedPlane(self.pos + v.point, v,
                                           du=self.limit, dv=self.limit))
            self.edges.append(BoundedPlane(self.pos - v.point, -1 * v,
                                           du=self.limit, dv=self.limit))
        
        self.pr = CubeParams(self.pos, self.limit,
                             [self.rot, self.rot2, self.rot3], self.edges)
    
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
        return all(abs(abs(pr) - 1) <= eps for pr in (rot1_pr, rot2_pr, rot3_pr))
    
    def intersect(self, ray: Ray, eps=1e-6) -> float or None:
        self._update()
        
        int_pts = []
        for edge in self.edges:
            r = edge.intersect(ray)
            if r is not None:
                int_pts.append(r)
        
        if len(int_pts):
            return min(int_pts)
    
    def nearest_point(self, *pts: Point) -> Point:
        self._update()
        
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
                r = (self.edges[i].rot * -projection1 + self.edges[i].u * proj2
                     + self.edges[i].v * proj3 + Vector(near_pt))
                r = r.len()
            elif abs(projection2) > 1:
                proj2 = projection2 - sign(projection2)
                r = (self.edges[i].rot * -projection1 + self.edges[i].u * proj2
                     + Vector(near_pt))
                r = r.len()
            elif abs(projection3) > 1:
                proj3 = projection3 - sign(projection3)
                r = (self.edges[i].rot * -projection1 + self.edges[i].v * proj3
                     + Vector(near_pt))
                r = r.len()
            
            if r < r_min:
                r_min = r
                min_pt = near_pt
        
        return min_pt
