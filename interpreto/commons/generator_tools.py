"""
Tools for working with generators
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Generator, Iterable, Iterator
from functools import singledispatchmethod
from types import EllipsisType
from typing import Any


def enumerate_generator(generator: Iterable[Any]):
    """
    Enumerate a generator without generating all the elements
    """
    index = 0
    for elem in iter(generator):
        yield index, elem
        index += 1


class IteratorSplit(Iterable["SubIterator"]):
    # TODO : eventually allow split size != 1, sections or custom indexations (useless for now)
    def __init__(self, iterator: Iterator[Collection[Any]]):
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

    def sub_iterator_next(self, iterator_index: int):
        if self.buffers[iterator_index] == []:
            self.__generate_next_element()
        return self.buffers[iterator_index].pop(0)

    def __iter__(self):
        return iter(self.subiterators)


class SubIterator(Iterator[Any]):
    def __init__(self, main_iterator: IteratorSplit, position: int):
        self.main_iterator = main_iterator
        self.position = position

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self):
        return self.main_iterator.sub_iterator_next(self.position)


def split_iterator(iterator: Iterator[Collection[Any]]):
    return IteratorSplit(iterator)


def allow_nested_iterables_of(*types: type | EllipsisType) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    # TODO : check if Iterable or Generator in types
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def error_implementation(self: object, item: Any, *args: Any, **kwargs: Any) -> Any:
            raise TypeError(
                f"Unsupported type {type(item)} for method {func.__name__} in class {self.__class__.__name__}"
            )

        # Any, are you ok ? So, Any are you ok ? Are you ok, Any ?
        if Any in types or Ellipsis in types or len(types) == 0:
            res = singledispatchmethod(func)
        else:
            res = singledispatchmethod(error_implementation)
            for t in types:
                res.register(t, func)  # type: ignore : t can't be an EllipsisType here

        def generator_func(self: object, item: Iterator[Any], *args: Any, **kwargs: Any) -> Generator[Any, None, None]:
            yield from (res.dispatcher.dispatch(type(element))(self, element, *args, **kwargs) for element in item)

        res.register(Generator, generator_func)

        def iterable_func(self: object, item: Iterable[Any], *args: Any, **kwargs: Any) -> Iterable[Any]:
            result_generator = generator_func(self, iter(item), *args, **kwargs)
            try:
                return type(item)(result_generator)  # type: ignore
            except TypeError:
                return list(result_generator)

        res.register(Iterable, iterable_func)
        return res

    return decorator
