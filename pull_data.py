#from os.path import Path

import zipfile
import urllib2

#Path('tmp').mkdir_p()

import os
os.makedirs('tmp')

for model_name in ('seq2seq','seq2tree'):
  for data_name in ('jobqueries','geoqueries','atis'):
    fn = '%s_%s.zip' % (model_name, data_name)
    link = 'http://dong.li/lang2logic/' + fn
    with open('tmp/' + fn, 'wb') as f_out:
      f_out.write(urllib2.urlopen(link).read())
    with zipfile.ZipFile('tmp/' + fn) as zf:
      zf.extractall('./%s/%s/data/' % (model_name, data_name))

#Path('tmp').rmtree()
