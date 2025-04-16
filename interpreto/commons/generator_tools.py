"""
Tools for working with generators
"""

from collections.abc import Collection, Iterable, Iterator


def enumerate_generator(generator):
    """
    Enumerate a generator without generating all the elements
    """
    index = 0
    for elem in iter(generator):
        yield index, elem
        index += 1

class SubIterator(Iterator):
    def __init__(self, main_iterator, position:int | slice):
        self.main_iterator = main_iterator
        self.position = position

    def __iter__(self) -> Iterator:
        return self

    def __next__(self):
        return self.main_iterator.sub_iterator_next(self.position)

class IteratorSplit(Iterable):
    # TODO : eventually allow split size != 1, sections or custom indexations (useless for now)
    def __init__(self, iterator: Iterator[Collection]):
        self.iterator = iterator
        try:
            first_item = next(iterator)
            item_length = len(first_item)
        except StopIteration as e:
            raise ValueError("Iterator is empty. Can't split an empty iterator") from e
        self.n_splits = item_length
        self.buffers = [[a] for a in first_item]
        self.subiterators = [SubIterator(self, i) for i in range(self.n_splits)]

    def __len__(self):
        return self.n_splits

    def __getitem__(self, index: int):
        return self.subiterators[index]

    def __generate_next_element(self):
        try:
            item = next(self.iterator)
        except StopIteration as e:
            raise StopIteration() from e
        for index, element in enumerate(item):
            self.buffers[index].append(element)

    def sub_iterator_next(self, iterator_index):
        if self.buffers[iterator_index] == []:
            self.__generate_next_element()
        return self.buffers[iterator_index].pop(0)

    def __iter__(self):
        return iter(self.subiterators)

def split_iterator(iterator: Iterator[Collection]):
    return IteratorSplit(iterator)
