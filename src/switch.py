from abc import abstractmethod, ABC
from copy import copy
from inspect import signature


class SwitchError(Exception):
    pass


class ExhaustiveError(SwitchError):
    """ Raised when a Switch is not exhaustive and it is supposed to be. """
    pass


class FallThroughError(SwitchError):
    """ Raised when it's possible to fall through past the end of a switch. """
    pass


class DefaultError(SwitchError):
    """ Raised when the default case is not at the end. """
    pass


class RepeatedCaseError(SwitchError):
    """ Raised when a case is repeated more than once. """
    pass


class SignatureError(SwitchError):
    """ Raised when the function signatures do not match. """
    pass


class Default:
    """ Specify an unique class so that we don't interfere with other variables."""
    pass


def case(val, *, fallthrough=False):
    def decorator(function):
        return Case(function, val, fallthrough)

    return decorator


def cases(*, match=None, interval=None, fallthrough=False):
    if (match is None) is (interval is None):
        raise ValueError("Must provide exactly one of match, interval")
    if interval:
        def decorator(function):
            return IntervalCase(function, interval[0], interval[1], fallthrough)
    else:
        def decorator(function):
            return SelectCase(function, match, fallthrough)

    return decorator


class MetaCase:
    __slots__ = ("function", "fallthrough")

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    def __repr__(self):
        raise NotImplementedError()

    def __contains__(self, item):
        raise NotImplementedError()

    def __iter__(self):
        raise NotImplementedError()


class Case(MetaCase):
    __slots__ = ("case",)

    def __init__(self, function, case=None, fallthrough=False):
        self.function = function
        self.case = case
        self.fallthrough = fallthrough

    def __repr__(self):
        return f"Case({self.function!r}, case={self.case!r}, fallthrough={self.fallthrough!r})"

    def __contains__(self, item):
        return item == self.case

    def __iter__(self):
        yield self.case


class IntervalCase(MetaCase):
    __slots__ = ("type", "start", "end")

    def __init__(self, function, start=None, end=None, fallthrough=False):
        if start > end:
            raise ValueError(f"start ({start}) must be less than end ({end}).")

        self.function = function
        self.start = start
        self.end = end
        self.type = type(start)
        self.fallthrough = fallthrough

    def __repr__(self):
        return f"IntervalCase({self.function!r}, start={self.start!r}, end={self.end!r}, fallthrough={self.fallthrough!r})"

    def __contains__(self, item):
        try:
            return self.start <= item <= self.end
        except TypeError:
            return False

    def __iter__(self):
        yield from range(self.start, self.end)
        yield self.end


class SelectCase(MetaCase):
    __slots__ = ("selections",)

    def __init__(self, function, selections=None, fallthrough=False):
        self.function = function
        self.selections = selections
        self.fallthrough = fallthrough

    def __repr__(self):
        return f"Case({self.function!r}, selections={self.selections!r}, fallthrough={self.fallthrough!r})"

    def __contains__(self, item):
        return item in self.selections

    def __iter__(self):
        yield from self.selections


class MetaSwitch:
    _fn_signature = None

    @classmethod
    def match(cls, value, *args, **kwargs):
        """
        Will run the matched switch code for value - class level function for doing Switch.match()
        :param value:   The value / case the switch statement is matching.
        :param args:    Any arguments for the function that will execute on the match.
        :param kwargs:  Any kwargs for the function that will execute on the match.
        :return:
        """
        return cls.generic_match(cls, value, *args, **kwargs)

    def instance_match(self, value, *args, **kwargs):
        """
        Runs the matching code for value, passing along any args and kwargs which are needed.
        Instance level function for doing Switch().match()
        :return:        The output of the code block that was run.
        """
        return self.generic_match(self, value, *args, **kwargs)

    @staticmethod
    def generic_match(obj, value, *args, **kwargs):
        """
        A generic matching function that enables match() to work for both classes and instances.
        :param obj:
        :param value:
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError()

    @staticmethod
    def pack_args(*args, **kwargs):
        """ Returns all specified args and kwargs as a list of positional
        arguments and a dictionary of keyword arguments. """
        return args, kwargs

    @classmethod
    def _default(cls, *args, **kwargs):
        """
        The code that runs when the default case occurs.
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError(f"{cls.__name__} has not specified a default case with @case(Default)")

    @classmethod
    def list_cases(cls):
        return cls.generic_list_cases(cls)

    def instance_list_cases(self):
        return self.generic_list_cases(self)

    @staticmethod
    def generic_list_cases(obj):
        raise NotImplementedError()

    @classmethod
    def check_fn_signature(cls, fn):
        if cls._fn_signature is None:
            cls._fn_signature = signature(fn)
        elif cls._fn_signature != signature(fn):
            raise SignatureError("All cased functions must have matching signatures.")


class FallThroughSwitch(MetaSwitch, ABC):
    _fn_table = None
    _fall_throughs = None

    @staticmethod
    @abstractmethod
    def register_fallthroughs(cls):
        raise NotImplementedError()


class Switch(FallThroughSwitch):
    _fn_table = {}
    _fall_throughs = {}

    def __init__(self):
        self._fn_table = copy(self._fn_table)
        self._fall_throughs = copy(self._fall_throughs)
        self._fn_signature = copy(self._fn_signature)
        self.match = self.instance_match
        self.list_cases = self.instance_list_cases

    @staticmethod
    def register_fallthroughs(cls):
        falling_through = False
        last_fn = None
        for fn in cls._fn_table.values():
            if falling_through:
                cls._fall_throughs[last_fn] = fn

            if fn.fallthrough:
                last_fn = fn
                falling_through = True
            else:
                falling_through = False

        if falling_through:
            raise FallThroughError("Last fn has fallthrough specified, but has no remaining cases to fall into.")

    @staticmethod
    def __Switch_init_subclass__(cls):
        orig_table = cls._fn_table
        cls._fn_table = copy(cls._fn_table)
        cls._fall_throughs = copy(cls._fall_throughs)
        cls._fn_signature = copy(cls._fn_signature)
        have_default = False

        for fn in cls.__dict__.values():
            if not isinstance(fn, MetaCase):
                continue

            if not isinstance(fn, Case):
                raise ValueError("Switch does not support interval or selection cases.")

            if have_default:
                raise DefaultError("Default must occur at the end.")

            try:
                # We're allowed to override the statements of parent switch cases.
                # Note that order will be defined by the parent first.
                if fn.case in cls._fn_table and fn.case not in orig_table:
                    raise RepeatedCaseError(f"{fn.case} was specified more than once.")
                cls._fn_table[fn.case] = fn
            except TypeError:
                raise TypeError("Switch requires hashable values. To use unhashable values, subclass UnhashableSwitch.")

            cls.check_fn_signature(fn.function)

            if Default in fn:
                have_default = True
                cls._default = fn

        cls.register_fallthroughs(cls)

    def __init_subclass__(cls):
        Switch.__Switch_init_subclass__(cls)

    @staticmethod
    def generic_match(obj, value, *args, **kwargs):
        args = [None, *args]
        fn = obj._fn_table.get(value, obj._default)
        while fn in obj._fall_throughs:
            args, kwargs = fn(*args, **kwargs)
            fn = obj._fall_throughs[fn]
        return fn(*args, **kwargs)

    @staticmethod
    def generic_list_cases(obj):
        return list(obj._fn_table.values())


class UnhashableSwitch(FallThroughSwitch):
    _fn_table = []
    _fall_throughs = []

    @staticmethod
    def register_fallthroughs(cls):
        falling_through = False
        last_fn = None
        for fn in cls._fn_table:
            if falling_through:
                cls._fall_throughs[last_fn] = fn
            if fn.fallthrough:
                last_fn = fn
                falling_through = True
            else:
                falling_through = False

        if falling_through:
            raise FallThroughError("Last fn has fallthrough specified, but has no remaining cases to fall into.")

    @staticmethod
    def __UnhashableSwitch_init_subclass__(cls):
        cls._fn_table = copy(cls._fn_table)
        cls._fall_throughs = cls._fall_throughs.copy()
        cls._fn_signature = copy(cls._fn_signature)
        have_default = False

        for fn in cls.__dict__.values():
            if not isinstance(fn, MetaCase):
                continue

            if have_default:
                raise DefaultError("Default must occur at the end.")

            cls._fn_table.append(fn)
            cls.check_fn_signature(fn.function)

            if Default in fn:
                have_default = True
                cls._default = fn
        cls.register_fallthroughs(cls)

    def __init_subclass__(cls):
        UnhashableSwitch.__UnhashableSwitch_init_subclass__(cls)

    @staticmethod
    def generic_match(obj, value, *args, **kwargs):
        for fn in obj._fn_table:
            if value not in fn:
                continue

            args = [None, *args]
            while fn in obj._fall_throughs:
                args, kwargs = fn(*args, **kwargs)
                fn = obj._fall_throughs[fn]
            return fn(*args, **kwargs)
        return obj._default(None, *args, **kwargs)

    @staticmethod
    def generic_list_cases(obj):
        return [x for fn in obj._fn_table for x in fn]


class AutoSwitch(FallThroughSwitch):
    _fn_signature = None
    _fn_table = None

    @classmethod
    def __init_as_unhashable_switch(cls):
        cls._fn_table = cls._fn_table or []
        cls.__init_as(UnhashableSwitch)

    @classmethod
    def __init_as_switch(cls):
        cls._fn_table = cls._fn_table or {}
        cls.__init_as(Switch)

    @classmethod
    def __init_as(cls, type_):
        cls._fn_table = copy(cls._fn_table)
        cls._fall_throughs = {}
        cls._fn_signature = copy(cls._fn_signature)
        cls.register_fallthroughs = type_.register_fallthroughs
        cls.generic_match = type_.generic_match
        cls.generic_list_cases = type_.generic_list_cases
        getattr(type_, f"__{type_.__name__}_init_subclass__")(cls)

    def __init_subclass__(cls):
        """
        Will attempt to create a switch which utilizes Switch's case mechanism if possible, defaulting
        to UnhashableSwitch if an unhashable value is encountered.
        :return: A valid switch.
        """
        if cls.__name__ == "ExhaustiveSwitch":
            return

        cls._default = MetaSwitch._default

        if cls._fn_table and isinstance(cls._fn_table, list):
            cls.__init_as_unhashable_switch()
            return

        try:
            cls.__init_as_switch()
        except (ValueError, TypeError):
            cls._fn_table = []
            cls.__init_as_unhashable_switch()


class ExhaustiveSwitch(AutoSwitch):
    """
    A Switch statement which is exhaustive over enum.
    """

    def __init_subclass__(cls, enum=None, **kwargs):
        """
        Will attempt to create a Switch which covers all values of cls.enum, and raises TypeError if cls.enum is not
        fully covered.
        :return: A valid ExhaustiveSwitch
        """
        if enum is None:
            raise ValueError("Enum must be specified.")
        cls.enum = enum

        super(ExhaustiveSwitch, cls).__init_subclass__()
        if callable(cls._default) and cls._default != MetaSwitch._default:
            return

        # Get all defined values.
        if len(set(cls.list_cases())) < len(cls.list_cases()):
            raise RepeatedCaseError("Cases appear multiple times.")

        # Allow cls to include more cases, but it must cover all cases in cls.enum
        if set(cls.enum).difference(cls.list_cases()):
            raise ExhaustiveError(
                "ExhaustiveSwitch requires all enum values to be defined or @case(Default) to be specified."
            )


class MagicExhaustiveSwitch(MetaSwitch):
    _default = None

    def __init_subclass__(cls, enum=None, **kwargs):
        """
        Will attempt to create a Switch which covers all values of cls.enum, and raises TypeError if cls.enum is not
        fully covered.
        :return: A valid ExhaustiveSwitch
        """
        if enum is None:
            raise ValueError("Enum must be specified.")
        cls.enum = enum

        cls._fn_signature = None
        size_enum = len(cls.enum.__members__)
        common_functions = set(cls.enum.__members__).intersection(cls.__dict__)
        if cls._default is None and size_enum != len(common_functions):
            raise ExhaustiveError(
                f"Each item in {cls.enum.__name__} must have a corresponding function in {cls.__name__}")
        for name in common_functions:
            fn = getattr(cls, name)
            if not callable(fn):
                raise SignatureError(f"All items named after values in {cls.enum.__name__} must be callable.")

            cls.check_fn_signature(fn)

        if cls._default:
            cls.check_fn_signature(cls._default)

    @staticmethod
    def generic_match(obj, value, *args, **kwargs):
        getattr(obj, obj.enum(value).name, obj._default)(None, *args, **kwargs)


class ContextSwitch:
    """ A switch which allows code like
        a = 3
        with ContextSwitch(x) as ctx:
            if ctx(1):
                return 1
                ctx.escape
            if ctx(2):
                a = 2
            if ctx(3, fallthrough=False):
                a += 3
            if ctx.default or ctx(Default):
                return a
        """

    def __init__(self, value):
        self.value = value
        self._broken = False
        self._matched = False
        self._default_matched = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._matched and not self._broken:
            raise FallThroughError(
                "Can not fallthrough past end of switch. Try adding a ctx.escape or fallthrough=False."
            )

    def __break__(self):
        self._broken = True

    @property
    def default(self):
        if self._default_matched:
            raise DefaultError("Can not match default multiple times.")
        self._default_matched = True
        return not (self._matched and self._broken)

    @property
    def escape(self):
        self._broken = True
        return True

    def __call__(self, *args, fallthrough=True, **kwargs):
        """
        ContextSwitches can not have any of the following:
        repeated defaults, defaults before the end, fallthroughs at the end.
        :param args:
        :param fallthrough:
        :param kwargs:
        :return:
        """
        if self._default_matched:
            raise DefaultError("Can only match default value once, at the end of the switch.")
        if Default in args:
            self._default_matched = True

        if self._broken:
            return False

        if not self._matched and self.value in args:
            self._matched = True

        if self._matched and not self._broken:
            self._broken |= not fallthrough
            return True

        if self._default_matched and not self._matched:
            return True

        return False

    def __iter__(self):
        yield self
        yield self.__break__


class ExhaustiveContextSwitch(ContextSwitch):
    """ A switch which allows code like
        a = 3
        with ContextSwitch(x) as ctx:
            if ctx(1):
                return 1
            if ctx(2):
                a = 2
            if ctx(3, fallthrough=False):
                a += 3
            # alternative, ctx.default()
            if ctx.default:
                return a
        """

    def __init__(self, value, enum):
        super().__init__(value)
        self.enum = enum
        self.visited = []

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        if not (self._default_matched or set(self.enum).issubset(self.visited)):
            raise ExhaustiveError(f"Switch is not exhaustive over {self.enum}")

    def __call__(self, *args, fallthrough=True, **kwargs):
        self.visited.extend(args)
        return super(ExhaustiveContextSwitch, self).__call__(*args, fallthrough=fallthrough, **kwargs)
