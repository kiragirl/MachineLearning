a = 200
b = 66
print("A") if a > b else print("=") if a == b else print("B")

x = 52

if x > 10:
    print("Above ten,")
    if x > 20:
        print("and also above 20!")
    else:
        print("but not above 20.")

i = 1
while i < 6:
    print(i)
    i += 1
else:
    print("i is no longer less than 6")

fruits = ["apple", "banana", "cherry"]
for x in fruits:
    print(x)

for x in "banana":
    print(x)

for x in range(3, 10):
    print(x)

for x in range(3, 50, 6):
    print(x)

for x in range(10):
    print(x)
else:
    print("Finally finished!")


def my_function(child3, child2, child1):
    print("The youngest child is " + child3)


my_function(child1="Phoebe", child2="Jennifer", child3="Rory")


def my_function(*kids):
    print("The youngest child is " + kids[2])


x = lambda a: a + 10
print(x(5))

x = lambda a, b, c: a + b + c
print(x(5, 6, 2))


def myfunc(n):
    return lambda a: a * n


mydoubler = myfunc(2)

print(mydoubler(11))
print("-------------------")

print(10 >> 1)


class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def myfunc(self):
        print("Hello my name is " + self.name)


p1 = Person("Bill", 63)
p1.myfunc()


class Student(Person):
    pass


x = Student("Elon", "Musk")
x.myfunc()

import platform

x = dir(platform)
print(x)