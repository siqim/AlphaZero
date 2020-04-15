import threading
import time
from queue import Queue

condition = threading.Condition()
q = Queue(10)


class Producer(threading.Thread):
    """ 生产者 ，生产商品，5个后等待消费"""

    def __init__(self, product_name):
        super().__init__()
        self.product_name = product_name
        self.cnt = 0

    def run(self):

        while True:
            condition.acquire()
            if q.full():

                print('已达到10个，停止生产')
                # 唤醒消费者费线程
                condition.notify()
                # 等待-释放锁 或者 被唤醒-获取锁
                condition.wait()

            else:

                print('---执行，produce--')
                product = self.product_name + '_' + str(self.cnt)
                q.put(product, block=True)
                print('---生产产品:%s---' % product)
                self.cnt += 1

                condition.release()
                time.sleep(0.5)


class Customer(threading.Thread):

    def run(self):

        while True:
            condition.acquire()

            if q.full():
                bought_products = []
                while not q.empty():
                    bought_products.append(q.get(block=True))
                print('***卖出产品:%s***' % bought_products)

                condition.notify()
                condition.wait()
            else:
                condition.release()


if __name__ == '__main__':

    p1 = Producer('袜子')
    p2 = Producer('电脑')
    c1 = Customer(name='Customer-1')

    p1.start()
    p2.start()
    c1.start()

    p1.join()
    p2.join()
    c1.join()
