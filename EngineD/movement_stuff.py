import keyboard
import pyautogui as pag
from cg4_test import *


def move_forward(cam: Camera, speed=1):
    if cam.look_dir != Vector(0, 1, 0) \
        and cam.look_dir != Vector(0, -1, 0):
        y = cam.look_dir.point.coords[1]
        cam.look_dir.point.coords[1] = 0
        cam.pos = cam.pos + cam.look_dir.point * speed
        cam.look_dir.point.coords[1] = y
        cam.screen.pos = cam.pos + cam.look_dir.point
    else:
        cam.pos = cam.pos - cam.screen.u.point * speed
        cam.screen.pos = cam.pos + cam.look_dir.point
    
    return cam


def move_backward(cam: Camera, speed=1):
    if cam.look_dir != Vector(0, 1, 0) \
        and cam.look_dir != Vector(0, -1, 0):
        y = cam.look_dir.point.coords[1]
        cam.look_dir.point.coords[1] = 0
        cam.pos = cam.pos - cam.look_dir.point * speed
        cam.look_dir.point.coords[1] = y
        cam.screen.pos = cam.pos + cam.look_dir.point
    else:
        cam.pos = cam.pos + cam.screen.v.point * speed
        cam.screen.pos = cam.pos + cam.look_dir.point
    
    return cam


def move_left(cam: Camera, speed=1):
    y = cam.screen.v.point.coords[1]
    cam.screen.v.point.coords[1] = 0
    cam.pos = cam.pos + cam.screen.u.point * speed
    cam.screen.v.point.coords[1] = y
    return cam


def move_right(cam: Camera, speed=1):
    y = cam.screen.v.point.coords[1]
    cam.screen.v.point.coords[1] = 0
    cam.pos = cam.pos - cam.screen.u.point * speed
    cam.screen.v.point.coords[1] = y
    return cam


def move_to_viewpoint(cam: Camera, speed=1):
    cam.pos = cam.pos + cam.look_dir.point * speed
    return cam


def move_from_viewpoint(cam: Camera, speed=1):
    cam.pos = cam.pos - cam.look_dir.point * speed
    return cam


Events.add("w")
Events.add("s")
Events.add('a')
Events.add('d')
Events.handle('w', move_forward)
Events.handle('s', move_backward)
Events.handle('a', move_left)
Events.handle('d', move_right)


class Spectator(Camera):
    Events.add('shift + w')
    Events.add('shift + s')
    Events.handle('shift + w', move_to_viewpoint)
    Events.handle('shift + s', move_from_viewpoint)


class Player(Camera):
    """
    Реализовать перемещение так, чтобы не было движения сквозь объектов.
    Как?
    """


def launch(console: Console, camera_type: str = 'spectator',
          sensitivity=1, move_speed=1):
    """
    Функция, запускающая бесконечный цикл,
    в котором будут регистрироваться все действия пользователя
    (нажатие клавиш и перемещение мыши).

    :param console: Консоль
    :param camera_type: тип камеры ('spectator' или 'player')
    :param sensitivity: чувствительность мыши
    :param move_speed: скорость перемещения
    """
    assert camera_type in ['spectator', 'player']
    if camera_type == 'spectator':
        tmp = console.cam
        console.cam = Spectator(tmp.pos, tmp.look_dir,
                                tmp.fov * 360 / math.pi, tmp.draw_dist)
        
        # Задача 1:
        # Создать методы вызова определённых функций
        # в зависимости от нажатых клавиш (изменение положения камеры)
        # (возможно (можно быть первым)
        # перемещу в предыдущие два класса)
        
        # print(console.cam.screen.u, console.cam.screen.v)
        
        def close_console():
            raise SystemExit("Work was stopped with exit code 1")
        
        def mv1(action: str):
            console.cam = Events.trigger(action,
                                         console.cam, move_speed)
            # print(console.cam.pos)
            console.draw()
        
        keyboard.add_hotkey('ctrl+q', close_console)
        keyboard.add_hotkey('w', lambda: mv1('w'))
        keyboard.add_hotkey('s', lambda: mv1('s'))
        keyboard.add_hotkey('a', lambda: mv1('a'))
        keyboard.add_hotkey('d', lambda: mv1('d'))
        keyboard.add_hotkey('shift+w', lambda: mv1('shift + w'))
        keyboard.add_hotkey('shift+s', lambda: mv1('shift + s'))
        
        curr_pos = pag.position()
        pag.moveTo(pag.size()[0] // 2, pag.size()[1] // 2)
        pag.click()
        while True:
            something_happened = False
            new_pos = pag.position()
            if new_pos != curr_pos:
                something_happened = True
                difference = [(new_pos[0] - curr_pos[0]) * sensitivity,
                              (new_pos[1] - curr_pos[1]) * sensitivity]
                difference[0] /= (pag.size()[0] // 2)
                difference[1] /= (pag.size()[1] // 2)
                t, s = difference
                # print(t, s)
                
                console.cam.look_dir = t * console.cam.screen.u \
                                       + s * console.cam.screen.v \
                                       + Vector(console.cam.screen.pos) \
                                       - Vector(console.cam.pos)
                console.cam.look_dir = console.cam.look_dir.norm()
                
                # print(console.cam.look_dir)
                # print(console.cam.pos)
                console.cam.screen = BoundedPlane(
                    console.cam.pos + console.cam.look_dir.point,
                    console.cam.look_dir,
                    console.cam.screen.du, console.cam.screen.dv)
                
                curr_pos = new_pos
                pag.PAUSE = 0.15
            
            if something_happened:
                console.draw()
    
    else:
        # Здесь должно быть присвоение к классу Player,
        # и перемещение камеры в соответствии с ограничениями
        # (отсутствия возможности проходить сквозь объекты),
        # но разработчику лень это делать.
        pass


if __name__ == '__main__':
    # keyboard.add_hotkey("w", lambda: Events.trigger('w', cam))
    input()
    print(pag.size())
    
    pag.moveTo(1, 1, 1)
    pag.moveTo(100, 500, 1)
    pag.moveTo(500, 500, 1)
    pag.moveTo(1000, 650, 1)
    pag.moveTo(1900, 10, 1)
    pag.click()
