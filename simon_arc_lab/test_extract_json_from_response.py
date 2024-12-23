import unittest
from .extract_json_from_response import extract_json_from_response

RESPONSE_VALID_A = """
text before

```json
["ignore", "this", "list"]
```


```
[
  [
    1,
    2,
    3
  ],
  [
    4,
    5,
    6
  ]
]
```

text after
"""

RESPONSE_VALID_B = """
text before

```json
{
    "ignore": "this dictionary"
}
```

The result is:

```json
[
  [1, 2, 3],
  [4, 5, 6]
]
```

text after
"""

RESPONSE_VALID_C = """
text before

```json
{
    "ignore": "this dictionary"
}
```

The result is:

```json
{
  "return": "this dictionary"
}
```

text after
"""

RESPONSE_INVALID_A = """
text before

No tripple backticks anywhere. I don't contain a valid JSON block.

{
  "return": "this dictionary"
}

text after
"""

RESPONSE_INVALID_B = """
text before

No starting tripple backticks. I don't contain a valid JSON block.

{
  "return": "this dictionary"
}
```

text after
"""

class TestJsonFromResponse(unittest.TestCase):
    def test_10000_response_valid_a(self):
        actual = extract_json_from_response(RESPONSE_VALID_A)
        expected = [
            [1, 2, 3], 
            [4, 5, 6]
        ]
        self.assertEqual(actual, expected)

    def test_10001_response_valid_b(self):
        actual = extract_json_from_response(RESPONSE_VALID_B)
        expected = [
            [1, 2, 3], 
            [4, 5, 6]
        ]
        self.assertEqual(actual, expected)

    def test_10002_response_valid_c(self):
        actual = extract_json_from_response(RESPONSE_VALID_C)
        expected = {
            "return": "this dictionary"
        }
        self.assertEqual(actual, expected)

    def test_20000_response_invalid_a(self):
        with self.assertRaises(ValueError):
            extract_json_from_response(RESPONSE_INVALID_A)

    def test_20001_response_invalid_a(self):
        with self.assertRaises(ValueError):
            extract_json_from_response(RESPONSE_INVALID_B)

