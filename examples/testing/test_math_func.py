import math_func
import pytest
import sys


# Skip example with condition
@pytest.mark.skipif(sys.version_info < (3, 3), reason='Do not run test_add.')
def test_add():
    assert math_func.add(7, 3) == 10
    assert math_func.add(7) == 9


def test_add_float():
    result = math_func.add(10.5, 25.5)
    assert result == 36


@pytest.mark.strings
def test_add_strings():
    result = math_func.add('Hello ', 'World')
    assert result == 'Hello World'
    assert type(result) is str
    assert 'Heldlo' not in result

#


@pytest.mark.parametrize('x, y, result',
                         [(7, 3, 10),
                          ('Hello ', 'World', 'Hello World'),
                          (10.5, 25.5, 36)]
                         )
def test_add(x, y, result):
    assert math_func.add(x, y) == result


@ pytest.mark.number
def test_product():
    assert math_func.product(5, 3) == 15
    assert math_func.product(5) == 10


@ pytest.mark.strings
def test_product_strings():
    assert math_func.product('Hello ', 3) == 'Hello Hello Hello '
    result = math_func.product('Hello ')
    assert result == 'Hello Hello '
    assert type(result) is str
    assert 'Hello ' in result
