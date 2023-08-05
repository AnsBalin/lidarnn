from model.model import Model
import unittest


class TestSum(unittest.TestCase):
    def test_sum(self):
        model = Model()
        print(model.add(1, 2))


if __name__ == "__main__":
    unittest.main()
