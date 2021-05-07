# Multiprocessing didn't work right in the notebook so using .py

import concurrent.futures
import time

start = time.perf_counter()


def do_something(seconds):
    print(f'Sleeping {seconds} second(s)...')
    time.sleep(seconds)
    return f'Done Sleeping...{seconds}'


if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        secs = [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

        results = executor.map(do_something, secs)
        for result in results:
            print(result)

        #results = [executor.submit(do_something, sec) for sec in secs]
        # for f in concurrent.futures.as_completed(results):
        #   print(f.result())

        #f1 = executor.submit(do_something, 1)
        #f2 = executor.submit(do_something, 1)
        # print(f1.result())
        # print(f2.result())

    finish = time.perf_counter()

    print(f'Finished in {round(finish-start, 2)} second(s)')
