# 封裝 #繼承 #多型
class sum():
    def __init__(self,total):
        self.__total=total
    def show(self):
        return(self.__total)
    def save(self,amount):
        self.__total+=amount
    def cost(self,amount):
        self.__total-=amount
        
b=sum(1000)
b.__total=9999999 #封裝無法外部改變
b.save(5000)
b.cost(6000)
print(b.show())

class sum2():
    def __init__(self,total):
        self.total=total
    def show(self):
        return(self.total)
    def save(self,amount):
        self.total+=amount
    def cost(self,amount):
        self.total-=amount
        
b=sum2(1000)
b.total=9999999 #非封裝可以外部改變
b.save(5000)
b.cost(6000)
print(b.show())

print('封裝----------------------------------\n')
class Person():
    def __init__(self,name):
        self.name = name
    def hello(self):
        print(self.name + "人-hello\n")

class Son(Person):
    def __init__(self,name):
        super().__init__(name)
    def talk(self):
        super().hello()
        print(self.name + "學生-talk\n")

class Son2(Son):
    def __init__(self,name):
        super().__init__(name)
    def talk2(self):
        super().talk()
        print(self.name + "女性-talk2\n")

A1 = Person('Tony')
A1.hello()
A2 = Son('Hank')
A2.talk()
A3 = Son2('Apple')
A3.talk2()

print('繼承----------------------------------\n')
class Father():
    def say(self):
        print("我是你爹")
class Mother():
    def say(self):
        print("我是你媽")
class Me(Father,Mother):
    pass

Hank=Me()
Hank.say()
