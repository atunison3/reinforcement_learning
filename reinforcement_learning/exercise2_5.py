def initialize_action_values(n: int = 10) -> dict[int, float]:
    """Initialize the action-value vector"""

    if not isinstance(n, int):
        raise TypeError(f"n must be integer, received type {type(n)}")

    if n < 1:
        raise ValueError(f"n must be > 0, received {n}")

    q: dict[int, float] = {1: 0.0}
    for i in range(2, n + 1):
        q[i] = 0.0

    return q


def main() -> None:
    """Main function"""

    return


if __name__ == "__main__":
    main()
