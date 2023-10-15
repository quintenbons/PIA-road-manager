class Car:
    def __init__(self, x, y, speed=1):
        self.x = x
        self.y = y
        self.speed = speed

    def move(self):
        self.x += self.speed
