import os
from random import randint
from unittest import TestCase
import knowknow.utility as knowknow


class Test(TestCase):
    def test_comb(self):
        self.assertEqual(knowknow.comb('q.w.e', 'e.r.t'), 'e.q.r.t.w')

    def test_gen_tuple_template(self):
        template = knowknow.gen_tuple_template(tuple('qwer'))
        self.assertEqual(template._fields, tuple('qwer'))
        self.assertEqual(template.__name__, '_'.join('qwer'))

    def test_make_cross(self):
        k___v = dict(zip('qwert', '12345'))
        namedtuple = knowknow.make_cross(k___v)
        old_cross = knowknow.make_cross_kwargs(k___v)
        for k, v in k___v.items():
            self.assertEqual(v, namedtuple.__getattribute__(k))
            self.assertEqual(v, old_cross.__getattribute__(k))

    def test_download_file(self):
        url = 'http://example.com/'
        outfn = 'a.test'
        knowknow.download_file(url, outfn)

        avi_url = 'https://file-examples-com.github.io/uploads/2018/04/file_example_AVI_1920_2_3MG.avi'
        knowknow.download_file(avi_url, outfn)
        self.assertAlmostEqual(2.3, os.path.getsize(outfn) / 2 ** 20, delta=0.2)

    def test_named_tupelize(self):
        ctypes = 'q.w.e.r'
        d = {tuple(randint(1, 100) for _ in range(len(ctypes))): 'value' for _ in range(4)}
        tupelized = knowknow.named_tupelize(d, ctypes)
        self.assertEqual(len(d), len(tupelized))
        for key in d:
            self.assertTrue(
                knowknow.make_cross(dict(zip(sorted(ctypes.split('.')), key))) in tupelized)
