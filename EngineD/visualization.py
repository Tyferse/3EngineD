import configparser
import numpy as np

from high_level_objects import *


config = configparser.ConfigParser()
config.read("config.cfg")
hight = int(config['SCREEN']['screen_hight'])
width = int(config['SCREEN']['screen_width'])
ratio = width / hight


class Camera:
    def __init__(self, position: Point, look_dir: Vector,
                 fov, draw_dist):
        """

        :param position: Расположение.
        :param look_dir: Направление камеры (Vector).
        :param fov: Горизонтальный угол обзора.
        vfov = fov * H / W - вертикальный угол обзора
        :param draw_dist: Дистанция прорисовки.
        """
        self.pos = position
        self.look_dir = look_dir.norm()
        
        self.fov = (fov / 180 * math.pi) / 2
        self.vfov = self.fov / ratio
        self.screen = BoundedPlane(
            self.pos + self.look_dir.point / math.tan(self.fov),
            self.look_dir, math.tan(self.fov), math.tan(self.vfov))
        
        """
        Видимость ограничена сферой с длиной радиуса Draw_distance
        Экран - проекция сферы на плоскость,
        зависящей от self.pos и look_dir.
        """
        self.draw_dist = draw_dist
    
    def send_rays(self) -> list[list[Ray]]:
        # Считаем расстояние от камеры до пересечения луча с объектами
        rays = []
        # Создаём лучи к каждому пикселю
        for i, s in enumerate(np.linspace(
            -self.screen.dv, self.screen.dv, hight)):
            rays.append([])
            for t in np.linspace(-self.screen.du, self.screen.du,
                                 width):
                direction = Vector(self.screen.pos) \
                            + self.screen.v * s + self.screen.u * t
                
                direction = direction - Vector(self.pos)
                direction.point.coords[1] /= 14 / 48
                rays[i].append(Ray(self.pos, direction.norm()))
        
        return rays
    
    def rotate(self, x_angle=0, y_angle=0, z_angle=0):
        self.look_dir.rotate(x_angle, y_angle, z_angle)
        self.screen.pr.rotate(x_angle, y_angle, z_angle)
        self.screen.pr.pos = self.pos + self.look_dir.point
        self.screen._update()


# Список символов отрисовки, расположенных по "яркости"
symbols = " .:!/r(l1Z4H9W8$@"


class Canvas:
    def __init__(self, objmap: Map, camera: Camera):
        self.map = objmap
        self.cam = camera
    
    def update(self):
        rays = self.cam.send_rays()
        dist_matrix = []
        for i in range(hight):
            dist_matrix.append([])
            for j in range(width):
                # rays[i][j].dir /= ((rays[i][j].dir * self.cam.look_dir)
                #                    / rays[hight // 2][j].dir.len() ** 2)
                
                distances = rays[i][j].intersect(self.map)
                if all(d is None or d > self.cam.draw_dist
                       for d in distances):
                    dist_matrix[i].append(None)
                else:
                    dist_matrix[i].append(
                        min(filter(lambda x: x is not None, distances)))
        
        return dist_matrix


class Console(Canvas):
    def draw(self):
        dist_matrix = self.update()
        output_screen = ''
        for y in range(len(dist_matrix)):
            for x in range(len(dist_matrix[y])):
                if dist_matrix[y][x] is None:
                    output_screen += symbols[0]
                    continue
                
                try:
                    gradient = dist_matrix[y][x] / self.cam.draw_dist \
                               * (len(symbols) - 1)
                    
                    output_screen += symbols[len(symbols)
                                             - round(gradient) - 1]
                except (IndexError, TypeError):
                    print(len(symbols) * dist_matrix[y][x]
                          / self.cam.draw_dist, dist_matrix[y][x])
                    raise
            
            output_screen += '\n'
        
        print(output_screen)
