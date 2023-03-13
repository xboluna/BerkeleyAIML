from PIL import Image
import os 
from glob import glob

def verify_image(fn):
    "Confirm that `fn` can be opened"
    try:
        im = Image.open(fn).convert('RGB')
         return True
    except: 
        print(f'removing {fn}')
        os.remove(fn)
    return

import pdb;pdb.set_trace()
subdirs = glob(str('./datascraper/data/train/*'))
for dir in subdirs:
    for file in glob(str(dir)+'/*'):
        print(file)
        verify_image(file)
