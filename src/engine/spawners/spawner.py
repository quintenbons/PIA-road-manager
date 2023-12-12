import random
from engine.movable.movable import Movable
from engine.node import Node
from engine.road import Road
from typing import Callable, List

class Spawner:
    sources: List[Road]
    destinations: List[Road]
    get_rate: Callable[[int], int]

    movables: List[Movable] = []

    # source: List[Road] - The source of the traffic, car will spawn from this source
    # destinations: List[Road] - The destination of the traffic, car will go to this destination
    # get_rate: callable[int] - A function that returns the rate of spawn for a given time, we spawn |get_rate(time)| cars per second
    # initial_rate: int - The initial rate of spawn
    def __init__(self, sources: List[Road], destinations: List[Road], get_rate: Callable[[int], int], initial_rate: int = 0):
        self.sources = sources
        self.destinations = destinations
        self.get_rate = get_rate
        self.movables = []

        for _ in range(initial_rate):
            self.spawn()

    def update(self, time: int):
        rate = self.get_rate(time)

        for _ in range(rate):
            self.spawn()
        # n = len(self.movables)
        # if n < 100:
        #     for _ in range(100 - n):
        #         self.spawn()

        remove_list = []
        for m in self.movables:
            if not m.update():
                remove_list.append(m)
        for m in remove_list:
            self.movables.remove(m)

    def spawn(self):
        source = self.sources[random.randint(0, len(self.sources) - 1)]
        new_movable = Movable(5, 2, random.random(), random.random() * source.road_len, 2)
        # Add the movable to the road and the road to the movable
        if source.spawn_movable(new_movable, random.randint(0, len(source.lanes) - 1)):

        # Get a random destination
            # new_movable.get_path(self.destinations[random.randint(0, len(self.destinations) - 1)])
            destination = self.destinations[random.randint(0, len(self.destinations) - 1)]

            if destination == source:
                remaining = destination.road_len - new_movable.pos
                pos = new_movable.pos + remaining * (random.random() * 0.8)
                # print("unlucky ?")
            else:
                pos = destination.road_len * (random.random() * 0.5 + 0.25)

            new_movable.set_road_goal(destination, pos)
            self.movables.append(new_movable)

        