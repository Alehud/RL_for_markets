from abc import ABC, abstractmethod


class Animal(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def make_noise(self):
        pass


class Dog(Animal):

    def __init__(self):
        super().__init__()

    def make_noise(self):
        print("Bark! Bark!")


class Cat(Animal):

    def __init__(self):
        super().__init__()

    def make_noise(self):
        print("Meow! Meow!")


Neri = Dog()
Karin = Cat()

Neri.make_noise()
Karin.make_noise()

# Would return an error
# Alex = Animal()
# Alex.make_noise()

