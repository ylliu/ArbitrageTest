from unittest import TestCase

from back_test_pair import BackTestPair


class TestBackTestPair(TestCase):
    def test_get_top_50(self):
        back_test = BackTestPair('BTC/USDT', 'ETH/USDT')
        top_50_symbols = back_test.get_top_50()
        print(top_50_symbols)


    def test_get_listing_date(self):
        back_test = BackTestPair('BTC/USDT', 'ETH/USDT')
        res = back_test.get_listing_date('bitcoin')
        print(res)
