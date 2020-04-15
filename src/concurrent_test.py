import threading
import time
from queue import Queue


# 生产方法
def produce(product_name):
    global price_dict, total_num
    cnt = 1
    while True:
        if total_num >= max_num:
            break

        if not q.full():
            product = product_name + '_' + str(cnt)
            q.put(product, block=True)
            event.wait()

            while 1:
                if product in price_dict:
                    break
                else:
                    event.clear()
                    event.wait()

            price = price_dict[product]
            assert price == base_money[product_name] * cnt
            del price_dict[product]
            # print('---生产产品:%s---，---价格:%d---' % (product, price))
            total_num += 1
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
            time.sleep(0.001)
            event.set()
        else:
            event.clear()


if __name__ == '__main__':

    event = threading.Event()
    base_money = {str(i): i+1 for i in range(512)}
    q = Queue(len(base_money))
    price_dict = {}
    total_num = 0
    max_num = 30000

    t1 = threading.Thread(target=consume, daemon=True)
    t1.start()

    t_list = [threading.Thread(target=produce, args=(key,), name='producer_{idx}'.format(idx=key), daemon=True)
              for key in base_money.keys()]

    tik = time.time()
    for t in t_list:
        t.start()

    for t in t_list:
        t.join()
    t1.join()
    tok = time.time()
    print(tok - tik)
