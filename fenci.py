import jsonlines
from collections import Counter
import jieba
from jieba import cut
from multiprocessing import Pool
from sys import argv
from functools import partial

def count_on(file: str, rank: int, all: int):
    if len(argv) > 1:
        jieba.set_dictionary(argv[1])
        jieba.initialize()
        print(f"using dict {argv[1]}")
    counter: Counter[str] = Counter()
    with jsonlines.open(file) as reader:
        n = 0
        for index, object in enumerate(reader):
            if index % all != rank:
                continue
            content = object["content"]
            words = cut(content, HMM=False)
            counter.update(words)
            n += 1
            if n % 1000 == 0:
                print(f"processed {n} articles")
    return counter

if __name__ == "__main__":
    counter = Counter()
    all = 10
    with Pool(all) as p:
        for file in ["zhihu_train", "zhihu_test", "zhihu_valid"]:
            counters = p.map(partial(count_on, f"data/{file}.jsonl", all=all), range(all))
            for c in counters:
                counter.update(c)

    with open("words-simple.txt", "w") as f:
        for word, count in counter.most_common():
            f.write(f"{word}\t{count}\n")
