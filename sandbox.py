from abc import ABC, abstractmethod


class Animal(ABC):

    def __init__(self, value):
        self.value = value
        super().__init__()

    @abstractmethod
    def make_noise(self):
        pass


class Dog(Animal):

    def __init__(self, value):
        super().__init__(value)

    def make_(self):
        print(self.value*100)


a = SubclassFromAbstract1(42)
a.do_something()
