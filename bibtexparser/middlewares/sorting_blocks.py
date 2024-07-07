import abc
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
from functools import cmp_to_key, lru_cache
import operator
from typing import Any, Generic, TypeVar, cast, runtime_checkable
from typing import List
from typing import Protocol
from typing import Tuple
from typing import Type

from bibtexparser.library import Library
from bibtexparser.model import Block
from bibtexparser.model import Entry
from bibtexparser.model import ExplicitComment
from bibtexparser.model import ImplicitComment
from bibtexparser.model import Preamble
from bibtexparser.model import String

from .middleware import LibraryMiddleware


class Key(Protocol):
    """
    Helper protocol for objects that support all comparison operations.

    This is the recommended protocol for a sorting key. Technically, only
    `__lt__` is required for Python's standard sorting methods. However,
    providing all of them is recommended (see PEP 8). If you define `__eq__`
    and one of the remaining operations, then the `functools.total_ordering`
    decorator can define the remaining operations.
    """
    def __eq__(self, obj, /) -> bool:
        ...

    def __lt__(self, obj, /) -> bool:
        ...

    def __le__(self, obj, /) -> bool:
        ...

    def __gt__(self, obj, /) -> bool:
        ...

    def __ge__(self, obj, /) -> bool:
        ...


_BA = TypeVar("_BA", bound=Block, contravariant=True)


@runtime_checkable
class KeyGeneratorTemplate(Protocol, Generic[_BA]):
    def as_key_gen(self) -> Callable[[_BA], Key]:
        ...


KeyGeneratorDescription = (
    Callable[[_BA], Key]
    | KeyGeneratorTemplate[_BA]
    | Sequence['KeyGeneratorDescription[_BA]']
)


def make_key_gen(desc: KeyGeneratorDescription[_BA]
                 ) -> Callable[[_BA], Key]:
    # Case 1: Lexicographic key
    if isinstance(desc, Sequence):
        sub_gen = tuple((make_key_gen(sub_desc) for sub_desc in desc))
        def closure(blk: _BA) -> Key:
            return tuple((key_gen(blk) for key_gen in sub_gen))
        return closure
    # Case 2: Key template (convertible to key generator)
    if hasattr(desc, 'as_key_gen'):
        return cast(KeyGeneratorTemplate, desc).as_key_gen()
    # Case 3: Key generator directly given by caller
    return cast(Callable[[_BA], Key], desc)


def type_key(order: Tuple[Type[Block], ...] | None = None,
             sub_keys: Mapping[Type[Block], KeyGeneratorDescription] | None = None,
             fallback_idx: int | None = None
             ) -> Callable[[Block], Key]:
    """
    Key generator to sort blocks by type.
    """
    # Set default type order
    if order is None:
        order = (String, Preamble, Entry, ImplicitComment, ExplicitComment)
    # Set default value for fallback index
    if fallback_idx is None:
        fallback_idx = len(order)

    # Maps block type to order index (with allowance for subclasses)
    def find_type(blk_type: Type[Block]) -> int:
        try:
            return order.index(blk_type)
        except KeyError:
            for idx, check_type in enumerate(order):
                if issubclass(blk_type, check_type):
                    return idx
            return fallback_idx

    if sub_keys is None:
        # Cache type indices for subclasses
        find_type = lru_cache(find_type)

        # Main key generator
        def simple_key_closure(blk: Block) -> int:
            return find_type(type(blk))
        return simple_key_closure
    else:
        # Make subkey generators
        sub_keys = {key: make_key_gen(value) for key, value in sub_keys.items()}

        # Improved find closure
        @lru_cache
        def find_type_and_subkey(blk_type: Type[_BA]) -> tuple[int, Callable[[Any], Key] | None]:
            # Find index in order tuple
            type_idx = find_type(blk_type)

            # Find subkey
            try:
                sub_key = cast(Callable[[_BA], Key], sub_keys[blk_type])
            except KeyError:
                for key_type, key_gen in sub_keys.items():
                    if issubclass(blk_type, key_type):
                        sub_key = cast(Callable[[_BA], Key], key_gen)
                        break
                sub_key = None

            return type_idx, sub_key

        # Main key generator
        def complex_key_closure(blk: Block) -> Key:
            blk_type = type(blk)
            idx, sub_key_gen = find_type_and_subkey(blk_type)

            if sub_key_gen is None:
                return (idx,)
            
            sub_key = sub_key_gen(blk)
            if isinstance(sub_key, Sequence):
                return (idx, *sub_key)
            return (idx, sub_key)
        return complex_key_closure


class BlockComparator(abc.ABC, Generic[_BA]):
    def as_key(self) -> Callable[[_BA], Key]:
        return cmp_to_key(self.__call__)

    @abc.abstractmethod
    def __call__(self, left: _BA, right: _BA) -> int:
        ...


@dataclass
class _BlockJunk:
    """Data-Structure reflecting zero or more comments together with a block."""

    # The blocks (comments and the main block) are stored in the order they were parsed.
    blocks: List[Block] = field(default_factory=list)

    @property
    def main_block(self) -> Block:
        """Returns the main (i.e., non-comment) block."""
        try:
            return self.blocks[-1]
        except IndexError:
            raise RuntimeError(
                "Block junk must contain at least one block. "
                "This is a bug in bibtexparser, please report it."
            )


class SortBlocksMiddleware(LibraryMiddleware):
    """
    TODO
    """
    _key: Callable[[Block], Key]
    _junk: bool

    def __init__(
        self,
        key: KeyGeneratorDescription,
        preserve_comments_on_top: bool = True,
        allow_inplace_modification: bool = True
    ):
        self._key = make_key_gen(key)
        self._junk = preserve_comments_on_top
        super().__init__(
            allow_inplace_modification=allow_inplace_modification
        )

    @staticmethod
    def _gather_junk(blocks: list[Block]) -> list[_BlockJunk]:
        # Find indices of non-comment blocks
        main_blk_idx = [
            idx
            for idx, blk in enumerate(blocks)
            if not isinstance(blk, (ExplicitComment, ImplicitComment))
        ]

        # Assemble block junks by slicing between non-comment blocks
        # NOTE: Makes use of left-to-right execution order in sub-expressions
        first_pos = 0
        return [
            _BlockJunk(blocks=blocks[first_pos:(first_pos := last_pos + 1)])
            for last_pos in main_blk_idx
        ]

    @staticmethod
    def _junk_key(key: Callable[[Block], Key]) -> Callable[[_BlockJunk], Key]:
        """
        Adapt block sorting key for use on block junks.
        """
        def key_closure(junk: _BlockJunk) -> Key:
            return key(junk.main_block)
        return key_closure

    def transform(self, library: Library) -> Library:
        blocks = library.blocks
        if not (inplace := self.allow_inplace_modification):
            blocks = deepcopy(blocks)

        if self._junk:
            junks = self._gather_junk(blocks)
            junks.sort(key=self._junk_key(self._key))
            blocks[:] = [blk for junk in junks for blk in junk.blocks]
        else:
            blocks.sort(key=self._key)

        if inplace:
            return library
        else:
            return Library(blocks)


class SortBlocksByTypeAndKeyMiddleware(SortBlocksMiddleware):
    """Sorts the blocks of a library by type and key. Optionally, comments remain above same block."""

    def __init__(
        self,
        block_type_order: Tuple[Type[Block], ...] | None = None,
        preserve_comments_on_top: bool = True,
        allow_inplace_modification: bool = False
    ):
        super().__init__(
            key=type_key(
                order=block_type_order,
                sub_keys={
                    Entry: operator.attrgetter('entry_type')
                }
            ),
            preserve_comments_on_top=preserve_comments_on_top,
            allow_inplace_modification=allow_inplace_modification
        )
