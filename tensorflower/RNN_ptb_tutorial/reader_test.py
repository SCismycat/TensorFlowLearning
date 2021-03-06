#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Leslee
# @Email    : leelovesc@gmail.com
# @Time    : 2019.11.25 11:11
from tensorflower.python.platform import gfile
import tensorflower as tf
from com.hiekn.tensorflow.RNN_ptb_tutorial import reader
import os

class PtbReaderTest(tf.test.TestCase):

  def setUp(self):
    self._string_data = "\n".join(
        [" hello there i am",
         " rain as day",
         " want some cheesy puffs ?"])

  def testPtbRawData(self):
    tmpdir = tf.test.get_temp_dir()
    for suffix in "train", "valid", "test":
      filename = os.path.join(tmpdir, "ptb.%s.txt" % suffix)
      with gfile.GFile(filename, "w") as fh:
        fh.write(self._string_data)
    # Smoke test
    output = reader.ptb_raw_data(tmpdir)
    self.assertEqual(len(output), 4)

  def testPtbIterator(self):
    raw_data = [4, 3, 2, 1, 0, 5, 6, 1, 1, 1, 1, 0, 3, 4, 1]
    batch_size = 3
    num_steps = 2
    output = list(reader.ptb_iterator(raw_data, batch_size, num_steps))
    self.assertEqual(len(output), 2)
    o1, o2 = (output[0], output[1])
    self.assertEqual(o1[0].shape, (batch_size, num_steps))
    self.assertEqual(o1[1].shape, (batch_size, num_steps))
    self.assertEqual(o2[0].shape, (batch_size, num_steps))
    self.assertEqual(o2[1].shape, (batch_size, num_steps))

if __name__ == "__main__":
  tf.test.main()