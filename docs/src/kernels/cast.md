# Cast

- ✓* Indicates reintepreted casts

## Into Signed Integers

|⟍ Into <br /> ⟍   <br /> From⟍   |  Int8 | Int16 | Int32 |
|-|-|-|-|
| Int8 | | ✓ | ✓ |
| Int16 | | | ✓ |
| Int32 | | | |
| UInt8 | ✓* | ✓ | ✓ |
| UInt16 | | ✓* | ✓ |
| UInt32 | | |
| Float32 | | |
| Date32 | | |

## Into Unsigned Integers

|⟍ Into <br /> ⟍   <br /> From⟍   |  UInt8 | UInt16 | UInt32 |
|-|-|-|-|
| Int8 | ✓* | ✓* | ✓* |
| Int16 |  | ✓* | ✓* |
| Int32 |  |  |  |
| UInt8 | | ✓ | ✓ |
| UInt16 | | | ✓ |
| UInt32 | | |
| Float32 | ✓ | |
| Date32 | | |

caveats
- Float32 -> UInt8
    1. underflow -> `0`
    2. overflow -> `% 256`

## Into Floats
|⟍ Into <br /> ⟍   <br /> From⟍   |  Float32 |
|-|-|
| Int8 | ✓ |
| Int16 | ✓ |
| Int32 |  |
| UInt8 | ✓ |
| UInt16 | ✓ |
| UInt32 | |
| Float32 | |
| Date32 | |

## Into Boolean

|⟍ Into <br /> ⟍   <br /> From⟍   |  boolean |
|-|-|
| Int8 | |
| Int16 | |
| Int32 |  |
| UInt8 | |
| UInt16 | |
| UInt32 | |
| Float32 | ✓ |
| Date32 | |