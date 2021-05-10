from json_func import StudentDB
import pytest
import os

file_path = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope='module')
def db():
    print('*****Setup*****')
    db = StudentDB()
    db.connect(f'{file_path}/data.json')
    yield db
    print('*****Teardown*****')
    db.close()

# db = None

# Can use setup and teardown methods or the fixuture above.
# def setup_module(module):
#     print('*****Setup*****')
#     global db
#     db = StudentDB()
#     db.connect('data.json')


# def teardown_module(module):
#     print('*****Teardown*****')
#     db.close()


def test_scott_data(db):
    scott_data = db.get_data('Scott')
    assert scott_data['id'] == 1
    assert scott_data['name'] == 'Scott'
    assert scott_data['result'] == 'pass'


def test_mark_data(db):
    scott_data = db.get_data('Mark')
    assert scott_data['id'] == 2
    assert scott_data['name'] == 'Mark'
    assert scott_data['result'] == 'fail'
