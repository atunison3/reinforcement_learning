import unittest


class TestMain(unittest.TestCase):
    def test_main(self):
        from reinforcement_learning.main import foo

        message = foo()

        self.assertEqual(message, "Hello World")


if __name__ == "__main__":
    unittest.main()
