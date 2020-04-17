import threading
import time
from queue import Queue

event = threading.Event()
base_money = {str(i): i+1 for i in range(50)}
q = Queue(len(base_money))
price_dict = {}
total_num = 0
max_num = 5000
lock = threading.Lock()


# 生产方法
def produce(product_name):
    global price_dict, total_num
    cnt = 1
    while 1:

        product = product_name + '_' + str(cnt)
        q.put(product, block=True)

        while 1:
            if product in price_dict or total_num >= max_num:
                break
            else:
                time.sleep(0.00000000001)

        if total_num >= max_num:
            break

        price = price_dict[product]
        assert price == base_money[product_name] * cnt
        del price_dict[product]

        lock.acquire()
        total_num += 1
        lock.release()
        cnt += 1
        if total_num % 100 == 0:
            print(total_num)


def get_price(product_name):

    product_type, amount = product_name.split('_')
    return base_money[product_type] * int(amount)


# 消费方法
def consume():
    global price_dict
    while True:
        if total_num >= max_num:
            break
        if q.full():
            while not q.empty():
                product = q.get(block=True)
                price_dict[product] = get_price(product)
            time.sleep(0.01)


if __name__ == '__main__':

    t1 = threading.Thread(target=consume, daemon=True, name='consumer')
    t_list = [threading.Thread(target=produce, args=(key,), name='producer_{idx}'.format(idx=key), daemon=True)
              for key in base_money.keys()]
    print('start')
    tik = time.time()
    t1.start()
    for t in t_list:
        t.start()

    for t in t_list:
        t.join()
    t1.join()
    tok = time.time()
    print((tok - tik)/total_num)
