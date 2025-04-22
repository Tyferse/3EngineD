import math


class Point:
    """
    Класс точки в трёхмерном пространстве с операциями сложения, сравнения точек,
    нахождение расстояния между двумя точками (метод distance),
    умножения и деления на константу.
    """
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
        return Point(*[self.coords[i] * other for i in range(3)])
    
    def __sub__(self, other):
        return self.__add__(-1 * other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other: [int, float]):
        assert other != 0
        return self.__mul__(1 / other)
    
    def distance(self, pt):
        return math.sqrt(sum((self.coords[i] - pt.coords[i])**2 for i in range(3)))


class Vector:
    """
    Класс свободного вектора (радиус-вектора) в трёхмерном пространстве
    c операциями получения длины (расстояния от точки начала координат),
    нормирования, сложения с другим вектором и умножения на константу,
    скалярного и векторного произведения,
    вращения (относительно точки начала координат).
    """
    def __init__(self, *args):
        if len(args) == 1:
            assert isinstance(args[0], Point), ("1 argument must be Point, "
                                                f"got {type(args[0])} instead")
            self.point = args[0]
        elif len(args) == 3:
            assert all(map(isinstance, args, [(int, float)] * 3)), \
                ("3 arguments must be numbers, "
                 f"got {''.join(type(args[i]) for i in range(3))}")
            self.point = Point(*args)
        else:
            raise TypeError("expected 1 Point argument or 3 numeric argument, "
                            f"got {len(args)} arguments")
        
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
    
    def __eq__(self, other: "Vector"):
        return self.point == other.point
    
    def __ne__(self, other: "Vector"):
        return self.point != other.point
    
    def __add__(self, other: "Vector"):
        return Vector(self.point + other.point)
    
    def __sub__(self, other):
        return Vector(self.point - other.point)
    
    def __mul__(self, other):
        """Умножение на число, скалярное произведение векторов."""
        if isinstance(other, Vector):
            return sum(self.point.coords[i] * other.point.coords[i]
                       for i in range(3))
        else:
            return Vector(self.point * other)
    
    def __rmul__(self, other):
        assert isinstance(other, (int, float))
        
        return Vector(self.point * other)
    
    def __truediv__(self, other):
        assert isinstance(other, (int, float))
        
        return Vector(self.point / other)
    
    def __pow__(self, other):
        """Векторное произведение."""
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
    
    def rotate(self, x_angle: float = 0, y_angle: float = 0, z_angle: float = 0):
        x_angle = math.pi * x_angle / 360
        y_angle = math.pi * y_angle / 360
        z_angle = math.pi * z_angle / 360
        
        # Поворот вокруг оси Ox
        y_old = self.point.coords[1]
        z_old = self.point.coords[2]
        self.point.coords[1] = y_old * math.cos(x_angle) \
                               - z_old * math.sin(x_angle)
        self.point.coords[2] = y_old * math.sin(x_angle) \
                               + z_old * math.cos(x_angle)
        
        # Поворот вокруг оси Oy
        x_old = self.point.coords[0]
        z_old = self.point.coords[2]
        self.point.coords[0] = x_old * math.cos(y_angle) \
                               + z_old * math.sin(y_angle)
        self.point.coords[2] = x_old * -math.sin(y_angle) \
                               + z_old * math.cos(y_angle)
        
        # Поворот вокруг оси Oz
        x_old = self.point.coords[0]
        y_old = self.point.coords[1]
        self.point.coords[0] = x_old * math.cos(z_angle) \
                               - y_old * math.sin(z_angle)
        self.point.coords[1] = x_old * math.sin(z_angle) \
                               + y_old * math.cos(z_angle)


class VectorSpace:
    """
    Класс векторного пространства с точкой начала координат и базисными векторами.
    """
    init_pt = Point(0, 0, 0)
    basis = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
    
    def __init__(self, init_pt: Point = init_pt, dir1: Vector = None,
                 dir2: Vector = None, dir3: Vector = None):
        self.init_pt = init_pt
        for i, d in enumerate((dir1, dir2, dir3)):
            if d is not None:
                VectorSpace.basis[i] = d.norm()


Vector.vs = VectorSpace()


class Map:
    """Класс карты, на которой располагаются все отображаемые объекты."""
    def __init__(self, *args):
        self._obj_list = [*args]
    
    def append(self, *objs):
        self._obj_list.extend(objs)
    
    def __getitem__(self, item):
        return self._obj_list[item]
    
    def __iter__(self):
        return iter(self._obj_list)


class Ray:
    """Класс луча с точкой начала и направляющим вектором"""
    def __init__(self, ipt: Point, direction: Vector):
        self.inpt = ipt
        self.dir = direction
    
    def __str__(self):
        return f"Ray({self.inpt}, {self.dir})"
    
    def intersect(self, mapping: Map) -> list[float]:
        """
        Вычисляет расстояния до пересечения луча со всеми объектами на карте mapping.
        
        Args:
            mapping (Map): Карта с объектами.

        Returns:
            list[float]: Список расстояний до пересечения луча
            с каждым объектом на карте.
        """
        return [objt.intersect(self) for objt in mapping]
