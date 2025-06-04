# MIT License
#
# Copyright (c) 2025 IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL and FOR are research programs operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Tools for working with generators
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Generator, Iterable, Iterator
from functools import singledispatchmethod
from types import EllipsisType
from typing import Any


class IteratorSplit(Iterable["SubIterator"]):
    """Split an iterator of collections into independent iterators.

    Each element yielded by ``iterator`` must be a collection with the same
    length. ``IteratorSplit`` will expose one ``SubIterator`` per element of the
    collection, effectively allowing iteration over each column separately.

    Args:
        iterator: Iterator that yields collections of equal length.

    Raises:
        ValueError: If the provided ``iterator`` is empty.
    """

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
    """Iterator over a single column of an :class:`IteratorSplit`."""

    def __init__(self, main_iterator: IteratorSplit, position: int):
        """Instantiate a sub iterator.

        Args:
            main_iterator: The :class:`IteratorSplit` that manages the buffers.
            position: Index of the column to iterate over.
        """

        self.main_iterator = main_iterator
        self.position = position

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self):
        return self.main_iterator.sub_iterator_next(self.position)


def split_iterator(iterator: Iterator[Collection[Any]]) -> IteratorSplit:
    """Create an :class:`IteratorSplit` from ``iterator``.

    Args:
        iterator: Iterator yielding collections of equal length.

    Returns:
        IteratorSplit: The split iterator exposing as many sub-iterators as
        elements in the yielded collections.
    """

    return IteratorSplit(iterator)


def allow_nested_iterables_of(*types: type | EllipsisType) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to support nested iterables of specific types.

    The returned decorator will dispatch calls to ``func`` based on the type of
    the first argument. If the argument is an ``Iterable`` or ``Generator``, the
    function will be applied recursively to each element while preserving the
    container type when possible.

    Args:
        *types: Accepted types for dispatch. If empty or containing ``Ellipsis``
            or :class:`Any`, every type is accepted.

    Returns:
        Callable: A decorator adding the dispatch logic to ``func``.
    """

    # TODO : check if Iterable or Generator in types
    # TODO : check Unions in types
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
        return res  # type: ignore

    return decorator
