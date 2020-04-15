import threading
import time
from queue import Queue

q = Queue(5)
condition = threading.Condition()


class Producer(threading.Thread):
    """ 生产者 ，生产商品，5个后等待消费"""

    def run(self):

        while True:
            print(f'{threading.current_thread().name}, check lock')
            condition.acquire()
            print(f'{threading.current_thread().name}, continue')
            if q.qsize() >= 5:
                print('已达到5个，停止生产')
                # 唤醒消费者费线程
                condition.notifyAll()
                print('producer notify')
                # 等待-释放锁 或者 被唤醒-获取锁
                condition.wait()
                print('producer wait')
            else:
                q.put(1, block=True)
                print('生产了1个，现在有{0}个'.format(q.qsize()))
                time.sleep(0.5)


class Customer(threading.Thread):
    """ 消费者 抢购商品，每人初始10元，商品单价1元"""

    def run(self):
        while 1:
            print(f'{threading.current_thread().name}, check lock')
            condition.acquire()
            print(f'{threading.current_thread().name}, continue')
            if q.empty():
                print('没货了，{0}通知生产者'.format(threading.current_thread().name))
                condition.notify()
                print('{0} notify'.format(threading.current_thread().name))
                condition.wait()
                print('{0} wait'.format(threading.current_thread().name))

            else:
                print(q.qsize())
                q.get(block=True)
                print('{0}消费了1个, 剩余{1}个'.format(threading.current_thread().name, q.qsize()))
                condition.release()
                time.sleep(0.5)


if __name__ == '__main__':
    p = Producer(daemon=True, name='Producer')
    c1 = Customer(name='Customer-1')
    c2 = Customer(name='Customer-2')
    p.start()
    c1.start()
    c2.start()