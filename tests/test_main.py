import unittest


class TestMain(unittest.TestCase):
    def test_main(self):

        message = "Hello World"

        self.assertEqual(message, "Hello World")


if __name__ == "__main__":
    unittest.main()
