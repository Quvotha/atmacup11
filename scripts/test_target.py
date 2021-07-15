import unittest

from target import soring_date2target


class TestSortingDate2Target(unittest.TestCase):

    def test_soring_date2target(self):
        test_data = (
            (1599.99999999, 0),
            (1600, 0),
            (1600.00000000, 0),
            (1600.00000001, 1),
            (1699.99999999, 1),
            (1700, 1),
            (1700.00000000, 1),
            (1700.00000001, 2),
            (1799.99999999, 2),
            (1800, 2),
            (1800.00000000, 2),
            (1800.00000001, 3),
        )
        for sorting_date, expected in test_data:
            output = soring_date2target(sorting_date)
            self.assertEqual(output, expected, f'{sorting_date}, {expected}, {output}')


if __name__ == '__main__':
    unittest.main()
