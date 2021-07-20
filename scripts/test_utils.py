from typing import List
import unittest

from utils import should_stop_earlier


class TestShouldStopEarly(unittest.TestCase):

    LOSSES: List[float] = [1., 0.8, 0.6, 0.55, 0.5, 0.45]
    SCORES: List[float] = [0., 0.2, 0.4, 0.45, 0.5, 0.55]

    def test_loss_improve(self):
        output = should_stop_earlier(
            self.LOSSES,
            patience=3,
            min_delta=0.05
        )
        expected = True
        self.assertIs(output, expected,  'ロスが改善していないので中断する')
        output = should_stop_earlier(
            self.LOSSES,
            patience=3,
            min_delta=0.05 - 1e-5
        )
        expected = False
        self.assertIs(output, expected,  'ロスが改善しているので中断しない')

        output = should_stop_earlier(
            self.LOSSES,
            patience=7,
            min_delta=0.05 - 1e-5
        )
        expected = False
        self.assertIs(output, expected,  '学習回数が足りていないので中断しない')

        output = should_stop_earlier(
            self.LOSSES,
            patience=3,
            min_delta=0.05 - 1e-5,
            greater_is_better=True
        )
        expected = True
        self.assertIs(output, expected,  '`greater_than_better`が誤っているのでスコア扱い、中断になる')

    def test_score_improve(self):
        output = should_stop_earlier(
            self.SCORES,
            patience=3,
            min_delta=0.05,
            greater_is_better=True
        )
        expected = True
        self.assertIs(output, expected,  'スコアが改善していないので中断する')

        output = should_stop_earlier(
            self.SCORES,
            patience=3,
            min_delta=0.05 - 1e-5,
            greater_is_better=True
        )
        expected = False
        self.assertIs(output, expected,  'スコアが改善しているので中断しない')

        output = should_stop_earlier(
            self.SCORES,
            patience=7,
            min_delta=0.05 - 1e-5,
            greater_is_better=True
        )
        expected = False
        self.assertIs(output, expected,  '学習回数が足りていないので中断しない')

        output = should_stop_earlier(
            self.SCORES,
            patience=3,
            min_delta=0.05 - 1e-5,
        )
        expected = True
        self.assertIs(output, expected,  '`greater_than_better`が誤っているのでロス扱い、中断になる')

    def test_patience(self):
        should_stop_earlier(
            self.LOSSES,
            patience=2,
            min_delta=0.05
        )
        with self.assertRaises(ValueError, msg='`patience` が 1 なのでエラー'):
            should_stop_earlier(
                self.LOSSES,
                patience=1,
                min_delta=0.05
            )

    def test_min_delta(self):

        should_stop_earlier(
            self.LOSSES,
            patience=2,
            min_delta=0.
        )
        with self.assertRaises(ValueError, msg='`min_delta` が 0 未満なのでエラー'):
            should_stop_earlier(
                self.LOSSES,
                patience=2,
                min_delta=-1e-5
            )


if __name__ == '__main__':
    unittest.main()
