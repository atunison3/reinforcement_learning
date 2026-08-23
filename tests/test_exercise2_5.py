import unittest

from reinforcement_learning.exercise2_5 import initialize_action_values


class TestInitializeActionValues(unittest.TestCase):
    def test_default_n_returns_ten_zero_valued_actions(self) -> None:
        result = initialize_action_values()

        self.assertEqual(result, {i: 0.0 for i in range(1, 11)})

    def test_custom_n_returns_expected_keys_and_zero_values(self) -> None:
        n = 5

        result = initialize_action_values(n)

        self.assertEqual(set(result.keys()), set(range(1, n + 1)))
        self.assertTrue(all(value == 0.0 for value in result.values()))
        self.assertEqual(len(result), n)

    def test_n_of_one_returns_single_action(self) -> None:
        result = initialize_action_values(1)

        self.assertEqual(result, {1: 0.0})

    def test_non_integer_n_raises_type_error(self) -> None:
        invalid_values = (10.0, "10", None, [10], 10.5)

        for n in invalid_values:
            with self.subTest(n=n):
                with self.assertRaises(TypeError) as context:
                    initialize_action_values(n)  # type: ignore[arg-type]

                self.assertIn("n must be integer", str(context.exception))

    def test_n_less_than_one_raises_value_error(self) -> None:
        for n in (0, -1, -100):
            with self.subTest(n=n):
                with self.assertRaises(ValueError) as context:
                    initialize_action_values(n)

                self.assertIn("n must be > 0", str(context.exception))


if __name__ == "__main__":
    unittest.main()
