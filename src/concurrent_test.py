import threading
import time
from multiprocessing import Manager, Pool, Lock, Process, Queue
import os


def multithreading_produce(price_dict, q, total_num, max_num, lock, base_money):
    t_list = [threading.Thread(target=produce, args=(key, price_dict, q, total_num, max_num, lock, base_money),
                               name='producer_{idx}'.format(idx=key), daemon=True) for key in base_money.keys()]
    for t in t_list:
        t.start()
    for t in t_list:
        t.join()


# 生产方法
def produce(product_name, price_dict, q, total_num, max_num, lock, base_money):
    cnt = os.getpid()
    while 1:

        product = product_name + '_' + str(cnt)
        q.put(product, block=True)

        while product not in price_dict and total_num.value < max_num.value:
            time.sleep(0.00000001)

        if total_num.value >= max_num.value:
            break

        price = price_dict[product]
        assert price == base_money[product_name] * cnt
        del price_dict[product]

        lock.acquire()
        total_num.value += 1
        lock.release()

        cnt += os.getpid()
        if total_num.value % 100 == 0:
            print(os.getpid(), threading.current_thread().name, total_num.value)


def get_price(product_name, base_money):
    product_type, amount = product_name.split('_')
    return base_money[product_type] * int(amount)


# 消费方法
def consume(price_dict, q, total_num, max_num, base_money):
    while True:
        if total_num.value >= max_num.value:
            break
        if q.full():
            while not q.empty():
                product = q.get(block=True)
                price_dict[product] = get_price(product, base_money)
            time.sleep(0.01)


if __name__ == '__main__':
    num_processes = 2
    num_threads_per_process = 256
    max_buffer = 512

    base_money = {str(i): i+1 for i in range(num_threads_per_process)}
    manager = Manager()

    price_dict = manager.dict()
    q = Queue(max_buffer)
    total_num = manager.Value('i', 0)
    max_num = manager.Value('i', 5000)
    lock = Lock()

    consume_process = Process(target=consume, args=(price_dict, q, total_num, max_num, base_money,))
    produce_process = [Process(target=multithreading_produce,
                               args=(price_dict, q, total_num, max_num, lock, base_money,))
                       for _ in range(num_processes)]

    print('start')
    consume_process.start()
    for p in produce_process:
        p.start()

    tik = time.time()
    consume_process.join()
    for p in produce_process:
        p.join()
    tok = time.time()
    print((tok - tik)/total_num.value)
