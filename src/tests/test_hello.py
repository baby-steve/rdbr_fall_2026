# I haven't written unit tests in Python before so this
# is a sort of trial run of sorts. I'll delete/rename this
# file once we have actual unit tests.
import unittest


class TestHello(unittest.TestCase):
    def test_hello(self):
        self.assertEqual(2, 1)


if __name__ == "__main__":
    unittest.main()
