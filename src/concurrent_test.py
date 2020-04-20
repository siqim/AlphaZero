import threading
import time
from queue import Queue
import os


def multithreading_produce():
    t_list = [threading.Thread(target=produce, args=(key,),
                               name='producer_{idx}'.format(idx=key), daemon=True) for key in base_money.keys()]
    for t in t_list:
        t.start()
    for t in t_list:
        t.join()


# 生产方法
def produce(product_name):
    global total_num
    cnt = 1
    while 1:

        product = product_name + '_' + str(cnt)
        q.put(product, block=True)

        while product not in price_dict and total_num < max_num:
            time.sleep(0.00000001)

        if total_num >= max_num:
            break

        price = price_dict[product]
        assert price == base_money[product_name] * cnt
        del price_dict[product]
        cnt += 1

        lock.acquire()
        total_num += 1
        if total_num % 100 == 0:
            print(threading.current_thread().name, total_num)
        lock.release()


def get_price(product_name):
    product_type, amount = product_name.split('_')
    return base_money[product_type] * int(amount)


# 消费方法
def consume():
    while True:
        if total_num >= max_num:
            break
        if q.full():
            while not q.empty():
                product = q.get(block=True)
                price_dict[product] = get_price(product)
            time.sleep(0.01)


if __name__ == '__main__':
    num_threads = 512
    max_buffer = 512

    base_money = {str(i): i+1 for i in range(num_threads)}

    price_dict = {}
    q = Queue(max_buffer)
    total_num = 0
    max_num = num_threads * 10
    lock = threading.Lock()

    print('start')
    tik = time.time()

    consume_thread = threading.Thread(target=consume)
    consume_thread.start()
    multithreading_produce()
    consume_thread.join()

    tok = time.time()
    print((tok - tik)/total_num)
