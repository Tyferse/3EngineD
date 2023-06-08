import keyboard
import pyautogui as pag

from visualization import *


class Events:
    """
    Стек событий
    """
    evdata = {}
    
    @classmethod
    def add(cls, ev: str):
        """
        Добавление событий по названию.

        Event.add("OnRender")
        func(screen):
            screen.update()

        Event.handle("OnRender", func)
        """
        cls.evdata[ev] = []
    
    @classmethod
    def handle(cls, ev: str, func: type(add)):
        """
        "Бронирует" функцию func на какое-то событие ev.
        Хранение функций, соответствующих событию (списку событий).

        :param ev:
        :param func:
        :return:
        """
        cls.evdata[ev].append(func)
    
    @classmethod
    def remove(cls, ev: str, func: type(add)):
        if len(cls.evdata[ev]) < 2:
            del cls.evdata[ev]
            return
            
        for i in range(len(cls.evdata[ev])):
            if cls.evdata[ev][i] is func:
                del cls.evdata[ev][i]
                break
    
    @classmethod
    def __getitem__(cls, item):
        return cls.evdata[item]
    
    @classmethod
    def __iter__(cls):
        return iter(cls.evdata.keys())
    
    @classmethod
    def trigger(cls, ev, *args):
        """
        Обрабатываются функции,
        которые исполняются в соответствии с стеком событий.

        sc1 = Canvas()
        Trigger.trigger("OnRender", sc1)
        """
        try:
            for i in range(len(cls.evdata[ev])):
                called = cls.evdata[ev][i](*args)
                if called is not None:
                    return called
        except KeyError:
            print("Something went wrong:", end=' ')
            print(cls.evdata[ev])


def move_forward(cam: Camera, speed=1):
    if cam.look_dir != Vector(0, 1, 0) \
       and cam.look_dir != Vector(0, -1, 0):
        tmp = cam.look_dir
        tmp.point.coords[1] = 0
        tmp = tmp.norm()
        cam.pos = cam.pos + tmp.point * speed
        cam.screen.pos = cam.pos + cam.look_dir.point
    else:
        tmp = cam.screen.v
        tmp.point.coords[1] = 0
        tmp = tmp.norm()
        cam.pos = cam.pos - tmp.point * speed
        cam.screen.pos = cam.pos + cam.look_dir.point
        
        # cam.pos = cam.pos - cam.screen.u.point * speed
        # cam.screen.pos = cam.pos - cam.screen.u.point
    
    return cam


def move_backward(cam: Camera, speed=1):
    if cam.look_dir != Vector(0, 1, 0) \
       and cam.look_dir != Vector(0, -1, 0):
        tmp = cam.look_dir
        tmp.point.coords[1] = 0
        tmp = tmp.norm()
        cam.pos = cam.pos - tmp.point * speed
        cam.screen.pos = cam.pos + cam.look_dir.point
    else:
        tmp = cam.screen.v
        tmp.point.coords[1] = 0
        tmp = tmp.norm()
        cam.pos = cam.pos + tmp.point * speed
        cam.screen.pos = cam.pos + cam.look_dir.point
        
        # cam.pos = cam.pos + cam.screen.u.point * speed
        # cam.screen.pos = cam.pos + cam.screen.u.point
    
    return cam


def move_left(cam: Camera, speed=1):
    tmp = cam.screen.u
    # tmp.point.coords[1] = 0
    # tmp = tmp.norm()
    cam.pos = cam.pos - tmp.point * speed
    cam.screen.pos = cam.pos + cam.look_dir.point
    return cam


def move_right(cam: Camera, speed=1):
    tmp = cam.screen.u
    # tmp.point.coords[1] = 0
    # tmp = tmp.norm()
    cam.pos = cam.pos + tmp.point * speed
    cam.screen.pos = cam.pos + cam.look_dir.point
    return cam


def move_to_viewpoint(cam: Camera, speed=1):
    cam.pos = cam.pos + cam.look_dir.point * speed
    cam.screen.pos = cam.pos + cam.look_dir.point
    return cam


def move_from_viewpoint(cam: Camera, speed=1):
    cam.pos = cam.pos - cam.look_dir.point * speed
    cam.screen.pos = cam.pos + cam.look_dir.point
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
    pass


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
    assert camera_type in ('spectator', 'player')
    
    is_alive = True
    if camera_type == 'spectator':
        tmp = console.cam
        console.cam = Spectator(tmp.pos, tmp.look_dir,
                                tmp.fov * 360 / math.pi, tmp.draw_dist)
        del tmp
        
        def close_console():
            nonlocal is_alive
            is_alive = False
            print("Work was stopped with exit code 1")
        
        def mv1(action: str):
            console.cam = Events.trigger(action,
                                         console.cam, move_speed)
            console.draw()
        
        keyboard.add_hotkey('ctrl+q', close_console)
        keyboard.add_hotkey('w', lambda: mv1('w'))
        keyboard.add_hotkey('s', lambda: mv1('s'))
        keyboard.add_hotkey('a', lambda: mv1('a'))
        keyboard.add_hotkey('d', lambda: mv1('d'))
        keyboard.add_hotkey('shift+w', lambda: mv1('shift + w'))
        keyboard.add_hotkey('shift+s', lambda: mv1('shift + s'))
        
        pag.moveTo(pag.size()[0] // 2, pag.size()[1] // 2)
        pag.click()
        curr_pos = pag.position()
        while is_alive:
            something_happened = False
            new_pos = pag.position()
            if new_pos != curr_pos:
                something_happened = True
                difference = [(new_pos[0] - curr_pos[0]) * sensitivity,
                              (new_pos[1] - curr_pos[1]) * sensitivity]
                difference[0] /= (pag.size()[0] // 2)
                difference[1] /= (pag.size()[1] // 2)
                t, s = difference
                
                console.cam.look_dir = t * console.cam.screen.u \
                    + s * console.cam.screen.v \
                    + Vector(console.cam.screen.pos) \
                    - Vector(console.cam.pos)
                console.cam.look_dir = console.cam.look_dir.norm()
                
                console.cam.screen = BoundedPlane(
                    console.cam.pos + console.cam.look_dir.point,
                    console.cam.look_dir,
                    console.cam.screen.du, console.cam.screen.dv)
                
                curr_pos = new_pos
                # pag.PAUSE = 0.15
            
            if something_happened:
                console.draw()
            
            if new_pos[0] in (0, 1, 1919, 1920) \
               or new_pos[1] in (0, 1, 1079, 1080):
                pag.moveTo(pag.size()[0] // 2, pag.size()[1] // 2)
                curr_pos = pag.position()
    
    else:
        # Здесь должно быть присвоение к классу Player,
        # и перемещение камеры в соответствии с ограничениями
        # (отсутствия возможности проходить сквозь объекты),
        # но времени на раскачку нет
        pass
