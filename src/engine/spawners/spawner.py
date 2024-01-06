import random
from engine.constants import TIME
from engine.movable.movable import Movable
from engine.road import Road
from typing import Callable, List
import build.engine_ia as engine_ia

class Spawner:
    sources: List[Road]
    destinations: List[Road]
    get_rate: Callable[[int], int]
    _total_despawned_score: int = 0

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
            self.spawn(current_tick=0)

    def update(self, current_tick: int):

        cmov_list = [m.cmovable for m in self.movables]
        # for m in self.movables:
        #     # print(m.pos)
        #     if not m.cmovable.update():
        #         remove_list.append(m)
        index_list = engine_ia.spawner_update(cmov_list)
        remove_list: List[Movable] = [self.movables[i] for i in index_list]
        for m in remove_list:
            self._total_despawned_score += m.cmovable.get_score(current_tick)
            self.movables.remove(m)
        
        rate = self.get_rate(current_tick * TIME)

        # if len(self.movables) < 200:
        for _ in range(rate):
            self.spawn(current_tick=current_tick)
            #TODO
            pass

    def get_total_score(self, current_tick: int) -> int:
        """Get total simulation score (not node specific)"""
        total_active_score = 0
        for m in self.movables:
            total_active_score += m.get_score(current_tick)
        return self._total_despawned_score + total_active_score

    def reset_score(self):
        """Reset score of previously despawned movables (still maintains ongoing movables)"""
        self._total_despawned_score = 0

    def spawn(self, current_tick: int):
        source = self.sources[random.randint(0, len(self.sources) - 1)]
        new_movable = Movable(random.random()*4 + 1, random.random()*2.5 + 0.5, random.random() * source.croad.get_road_len(), 2, spawn_tick=current_tick)
        # Add the movable to the road and the road to the movable
        # print(new_movable.cmovable)
        if source.croad.spawn_movable(new_movable.cmovable):

        # Get a random destination
            # new_movable.get_path(self.destinations[random.randint(0, len(self.destinations) - 1)])
            destination = self.destinations[random.randint(0, len(self.destinations) - 1)]

            if destination == source:
                remaining = destination.croad.get_road_len() - new_movable.cmovable.get_pos()
                pos = new_movable.cmovable.get_pos() + remaining * (random.random() * 0.8)
                # print("unlucky ?")
            else:
                pos = destination.croad.get_road_len() * (random.random() * 0.5 + 0.25)

            new_movable.cmovable.set_road_goal(destination.croad, pos)
            self.movables.append(new_movable)

        