import abc
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
from functools import cmp_to_key, lru_cache
import operator
from typing import Any, Generic, TypeVar, cast, overload, override
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
    Protocol for sorting keys.

    Sorting keys must implement the rich comparison operations `__lt__`,
    `__le__`, `__gt__`, and `__ge__`, as well as `__eq__`. These must
    follow the criteria for a total order relation and must not have any
    side effects.

    It is possible to derive the remaining operations from `__eq__` and one
    of the remaining comparisons with the `@functools.total_ordering`
    decorator.
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
_K = TypeVar("_K", bound=Key)
_TK = TypeVar("_TK")
_V = TypeVar("_V")



class _ReversedKey(Key, Generic[_K]):
    """
    Special key that reverses order.

    By default, Python sorting functions sort in ascending order. By wrapping
    a key in one of these objects, the order relation is reversed and Python
    sorts in descending order.

    This can only be compared to other reversed keys. It is used internally by
    `descending_key`.`
    """
    base: _K

    def __init__(self, base: _K):
        self.base = base

    def __eq__(self, other, /) -> bool:
        return self.base == other.base

    def __lt__(self, other, /) -> bool:
        return self.base > other.base

    def __le__(self, other, /) -> bool:
        return self.base >= other.base

    def __gt__(self, other, /) -> bool:
        return self.base < other.base

    def __ge__(self, other, /) -> bool:
        return self.base <= other.base


_KeyGenDesc = (
    Callable[[_BA], Key]
    | Sequence['_KeyGenDesc[_BA]']
)


def _keygen(
    desc: _KeyGenDesc[_BA]
) -> Callable[[_BA], Key]:
    """
    Make key generator from a key description.
    """
    # Lexicographic key
    if isinstance(desc, Sequence):
        sub_gen = tuple((_keygen(sub_desc) for sub_desc in desc))
        def closure(blk: _BA) -> Key:
            return tuple((key_gen(blk) for key_gen in sub_gen))
        return closure

    # Key generator directly given by caller
    return cast(Callable[[_BA], Key], desc)


def compare_key(cmp: Callable[[_BA, _BA], int]) -> Callable[[_BA], Key]:
    """
    Make key generator from comparison function.

    Comparison functions take two arguments and return an integer. For inputs
    `a` and `b`, an input equal to `0` indicates that `a == b`, a negative
    integer indicates `a < b`, and a positive integer indicates `a > b`.

    In order to yield a valid sorting key, the comparison function must not
    have any side effects and must satisfy the following axioms:

    1. `cmp(a, a) == 0` for all `a`;
    2. `cmp(a, b) == 0` if and only if `cmp(b, a) == 0`;
    3. if `cmp(a, b) == 0` and `cmp(b, c) == 0`, then `cmp(a, c) == 0`;
    4. if `cmp(a, b) < 0` and `cmp(b, c) < 0`, then `cmp(a, c) < 0`.

    Internally, this function calls `functools.cmp_to_key()` to wrap the
    compare function in a key object.

    Args:
        cmp: Compare function of signature `(B, B) -> int` where `B` is a
            block type.

    Returns:
        Key generator of signature `(B) -> K` where `B` is the block type
        accepted by `cmp` and `K` is a key object.
    """
    return cmp_to_key(cmp)


def reverse_key(sub: _KeyGenDesc[_BA]) -> Callable[[_BA], Key]:
    """
    Make reversed key generator.

    Args:
        sub: Description of a key generator for a block type `B`.

    Returns:
        Key generator of signature `(B) -> K` where `B` is a block type and
        `K` is a key object type. Key objects returned by this generator
        encode the reversed order from the key described by `sub`.
    """
    sub_gen = _keygen(sub)

    def reverse_key_gen(blk: _BA) -> _ReversedKey:
        return _ReversedKey(sub_gen(blk))

    return reverse_key_gen


def _mro_get(cls: Type[_TK], map: Mapping[Type[_TK], _V],
                default: _V | None = None) -> _V | None:
    """
    Look up values associated with a type with fallback by MRO.

    If the type has no associated value, values associated with its ancestor
    type are looked up according to the method resolution order (MRO). This
    is used by `type_key()` to find sub-keys for subclasses.

    This function can raise multiple exceptions on every invocation. Caching
    its output is recommended.

    Args:
        cls: Type whose associated value is requested.
        map: Mapping in which the value should be looked up.
        default: Default value to return if no type in the MRO has an
            associated value. Defaults to `None`.

    Returns:
        Value associated with the first type in the MRO of `cls` to appear in
        `map` or `default` if none of them occur at all.
    """
    for key in cls.mro():
        try:
            return map[key]
        except KeyError:
            pass
    return default


def _mro_index(cls: Type[_TK], seq: Sequence[Type[_TK]],
               default: int | None = None) -> int:
    """
    Look up index of a type with fallback by MRO.

    If the type does not appear in the sequence, its ancestor type are looked
    up according to the method resolution order (MRO). This is used by
    `type_key()` to find types in the type order.

    This function can raise multiple exceptions on every invocation. Caching
    its output is recommended.

    Args:
        cls: Type whose index is requested
        seq: Type order sequence
        default: Default index to return if no type in the MRO appears in the
            order sequence. Defaults to `len(seq)`.

    Returns:
        Index of the first type in the MRO of `cls` that occurs anywhere in
        `seq` or `default` if no type in the MRO occurs in `seq`.
    """
    for key in cls.mro():
        try:
            return seq.index(key)
        except KeyError:
            pass

    if default is None:
        default = len(seq)
    return default


def type_key(order: Tuple[Type[Block], ...] | None = None,
             sub_keys: Mapping[Type[Block], _KeyGenDesc] | None = None,
             fallback_idx: int | None = None
             ) -> Callable[[Block], Key]:
    """
    Key generator to sort blocks by type.

    Type-specific key generators can be invoked for specified sub-types.
    Sub-keys must be comparable among all object types mapped to the same
    index within the order tuple.

    Args:
        order: Tuple of block types dictating the outermost order of blocks.
            A default order is applied if none is specified.
        sub_keys: Optional dictionary of sub-key descriptions for specific
            block types. Values must accept the key as an input type. Can be
            used to specify keys that only apply, e.g., to entries.
        fallback_idx: Optional index used for types not found in `order`.
            Defaults to the length of `order`.

    Returns:
        Key generator of signature `(Block) -> (int, ?)`, where the integer
        indicates the input block type's index in `order` and `?` is replaced
        with the output of the respective sub-key generator. Blocks of the
        exact same type are guaranteed to always use the same sub-key
        generator.

        For blocks of types not found in `order`, the blocks base types are
        searched in method resolution order (MRO) until a type in `order` is
        found. The result is internally cached for repeated lookups.
    """
    # Set default type order
    if order is None:
        order = (String, Preamble, Entry, ImplicitComment, ExplicitComment)

    if sub_keys is None:
        # Cache type indices for subclasses
        simple_find = lru_cache(
            lambda cls: _mro_index(cls, order, fallback_idx)
        )
        return lambda blk: simple_find(type(blk))

    # More involved closure for index-subkey pair.
    def find(cls: Type[_BA]) -> tuple[int, Callable[[Any], Key] | None]:
        idx = _mro_index(cls, order, fallback_idx)
        key_desc = _mro_get(cls, sub_keys)              # type: ignore
        
        if key_desc is None:
            key_gen = None
        else:
            key_gen = _keygen(key_desc)

        return idx, key_gen
    wrap_find = lru_cache(find)

    # Key generator closure.
    def key_gen(blk: Block) -> Key:
        idx, sub_key = wrap_find(type(blk))
        if sub_key is None:
            return (idx,)
        return idx, sub_key(blk)
    return key_gen


@dataclass
class _BlockJunk:
    """Data-Structure reflecting zero or more comments together with a block."""

    # The blocks (comments and the main block) are stored in the order they were parsed.
    blocks: List[Block] = field(default_factory=list)

    @property
    def main_block(self) -> Block:
        """Return the main (i.e., non-comment) block."""
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
    _reverse: bool
    _junk: bool

    def __init__(
        self,
        key: _KeyGenDesc,
        reverse: bool = False,
        preserve_comments_on_top: bool = True,
        allow_inplace_modification: bool = True
    ):
        self._key = _keygen(key)
        self._reverse = reverse
        self._junk = preserve_comments_on_top
        super().__init__(
            allow_inplace_modification=allow_inplace_modification
        )

    @staticmethod
    def _gather_junk(blocks: list[Block]) -> list[_BlockJunk]:
        """
        Group blocks into junks.

        A junk is a group of a non-comment block and its preceding comments.
        If `preserve_comments_on_top` is set, then they get moved as a unit
        during sorting.

        Internally used by `transform()`.

        Args:
            blocks: List of blocks to group.

        Returns:
            List of block junks in the same order in which they exist in
            `blocks`.
        """
        # Assemble block junks by slicing between non-comment blocks. This
        # is a list comprehension that iterates over an inner generator. The
        # generator yields indices of non-comment blocks. The list
        # comprehension slices from the end of the preceding junk up to the
        # next non-comment block.
        # NOTE: Makes use of left-to-right execution order in sub-expressions
        slc_first = 0
        junks = [
            _BlockJunk(blocks=blocks[slc_first:(slc_first := slc_last + 1)])
            for slc_last in (
                idx for idx, blk in enumerate(blocks)
                if not isinstance(blk, (ExplicitComment, ImplicitComment))
            )
        ]

        # Handle edge case of trailing comments.
        if slc_first < len(blocks):
            junks.append(_BlockJunk(blocks=blocks[slc_first:]))

        return junks
        
    @staticmethod
    def _junk_key(key: Callable[[Block], Key]) -> Callable[[_BlockJunk], Key]:
        """
        Adapt sorting key generator for use on block junks.

        The adapted key generator always returns the key associated with the
        last block in the junk. Note that in rare edge cases, this can be a
        comment block.

        Internally used by `transform()`.

        Args:
            key: Key generator for blocks.

        Returns:
            Adapted key generator for junks.
        """
        return lambda junk: key(junk.main_block)

    @override
    def transform(self, library: Library) -> Library:
        # Get block list and copy if inplace modification is prohibited.
        blocks = library.blocks
        if not (inplace := self.allow_inplace_modification):
            blocks = deepcopy(blocks)

        # Perform sorting. Always writes result to `blocks`.
        if self._junk:
            junks = self._gather_junk(blocks)
            junks.sort(key=self._junk_key(self._key), reverse=self._reverse)
            blocks[:] = (blk for junk in junks for blk in junk.blocks)
        else:
            blocks.sort(key=self._key, reverse=self._reverse)

        # Create new library if a copy was generated.
        # NOTE: If `inplace` is set, then `blocks` should still reference
        #     `library.blocks`.
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
                    Entry: operator.attrgetter('entry_type', 'key')
                }
            ),
            preserve_comments_on_top=preserve_comments_on_top,
            allow_inplace_modification=allow_inplace_modification
        )
