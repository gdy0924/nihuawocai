from django.test import TestCase


# Create your tests here.

def t2():
    yield [1, 2, 30]
    yield [5, 6, 7]


a = t2()
print(a.__next__())
print(a.__next__())
print(a.__next__())
