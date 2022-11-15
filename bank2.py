class Customer():
    def __init__(self, name):
        self.rat = ''
        self.name = name
        self.amount = 0

    def add_amount(self, money):
        self.amount += money
        print('현재 잔액: ', self.amount)

    def set_name(self, name):
        self.name = name

    def set_rat(self):
        a = self.amount
        if a>100000:
            self.rat = 'vvip'
        elif a>10000:
            self.rat = 'vip'
        elif a>1000:
            self.rat = 'gold'
        elif a>1000:
            self.rat = 'silver'
        else:
            self.rat = 'bronze'

    def sub_amount(self, money):
        self.amount -= money
        print('등급: ', self.rat)

    def get_amount(self):
        return self.amount

    def get_name(self):
        return self.name

    def get_rat(self):
        return self.rat

def make_customer():
    print('아이디 생성!')
    name = input('이름 입력:')

    return Customer(name)

if __name__ == '__main__':
    Customer_list  = []

    while True:
        print('--'*30)
        input1 = input('옵션을 입력해 주세요: ')
        if input1 == '1':
            Customer_list.append(make_customer())

        elif input1 == '2':
            print('돈 입금!')
            c_id = input('아이디 입력: ')
            for cu in Customer_list:
                if cu.get_name() == c_id:
                    money = input('입금할 금액 입력: ')
                    cu.add_amount(int(money))

        elif input1 == '3':
            for cu in Customer_list:
                print(cu.get_name())

        elif input1 == '4':
            print('등급확인!')
            c_id = input('아이디 입력: ')
            for cu in Customer_list:
                if cu.get_name() == c_id:
                    cu.set_rat()
                    print('등급은 ', cu.get_rat(),'입니다.')