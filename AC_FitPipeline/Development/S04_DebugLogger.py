# from Utility_Rendering import *

# from Logger import *
from tqdm.auto import trange
import tqdm
import sys
import time
if __name__ == '__main__':
    outFolder = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\LogginTest'
    # for i in range(10):
    #     outFile = join(outFolder, str(i).zfill(3) + '.txt')
    #     logger = configLogger(outFile, )
    #     logger.info('---------------------------------')
    #     logger.info('\n'.join(['{}'.format(j) for j in range(10)]))

    # for i in tqdm.tqdm(trange(10), desc='Loop 1'):
    #     loop = tqdm.tqdm(trange(100), desc='Loop 2')
    #     for j in loop:
    #         # desc = 'Iteration:{}'.format(j)
    #         # loop.set_description(desc)
    #         time.sleep(0.1)

    outter_bar = tqdm.tqdm(range(20), desc='sleeping')
    outter_loop = range(20)

    inner_bar = tqdm.tqdm(range(15), desc='inside', leave=False)
    inner_loop = range(15)

    for i in outter_loop:
        outter_bar.update(1)
        for j in inner_loop:
            inner_bar.update(1)
            desc = 'Iteration:{}'.format(j)
            inner_bar.set_description(desc)
            time.sleep(0.5)
        inner_bar.reset()