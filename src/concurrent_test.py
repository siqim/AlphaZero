import threading
import time
from queue import Queue

# 商品队列
q = Queue(3)
event = threading.Event()
base_money = {'shoes': 3, 'computers': 30, 'hat': 15, 'headphones': 20, 'phones': 55}
price_dict = {}


# 生产方法
def produce(product_name):
    global price_dict
    cnt = 1
    while True:

        if not q.full():

            product = product_name + '_' + str(cnt)
            q.put(product, block=True)
            event.wait()
            if event.is_set():
                price = price_dict[product]
                print('---生产产品:%s---，---价格:%d---' % (product, price))
                cnt += 1


        time.sleep(0.5)


def get_price(product_name):
    product_type, amount = product_name.split('_')
    return base_money[product_type] * int(amount)


# 消费方法
def consume():
    global price_dict
    while True:
        if q.full():

            while not q.empty():
                product = q.get(block=True)
                price_dict[product] = get_price(product)
                event.set()

            print('***卖出产品:%s***' % price_dict)


if __name__ == '__main__':
    t1 = threading.Thread(target=consume)
    t2 = threading.Thread(target=produce, args=('shoes',), name='producer_1')
    t3 = threading.Thread(target=produce, args=('computers',), name='producer_2')
    t4 = threading.Thread(target=produce, args=('hat',), name='producer_2')
    t5 = threading.Thread(target=produce, args=('headphones',), name='producer_2')
    t6 = threading.Thread(target=produce, args=('phones',), name='producer_2')

    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()
    t6.start()
#  10个生产者，共计生产100个不同的商品后，一次性给1个消费者，该消费者随即返回各个生产者各个商品的费用
