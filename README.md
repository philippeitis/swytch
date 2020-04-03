# xswitch
If you've ever felt like you really, really need a switch statement in Python, then this might be the thing for you.

There's a little bit of everything for everyone: Exhaustive switches, default cases, switch cases for hashable types, switch cases for unhashable types, context-based switches, and switches over enums.

```python
import enum
from xswitch import case, Default, ExhaustiveSwitch
class PowersOfTwo(enum.IntEnum):
  zero = 1
  one = 2
  two = 4
  
class PowersOfTwoSwitch(ExhaustiveSwitch, enum=PowersofTwo):
  @case(PowersOfTwo.zero)
  def zero(self):
    print("2^0")
  
  @case(PowersOfTwo.one)
  def one(self):
    print("2^one")
  
  @case(Default)
  def default(self):
    print("2^?")

if __name__ == "__main__":
  PowersOfTwoSwitch.match(0)
  PowersOfTwoSwitch.match(PowersOfTwo.one)
  PowersOfTwoSwitch.match(5)

>>> 2^0
>>> 2^one
>>> 2^?
```


