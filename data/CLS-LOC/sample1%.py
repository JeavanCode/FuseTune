import os
import glob
import numpy as np
import shutil
np.random.seed(0)
classList = os.listdir('valid')
for cla in classList:
    os.makedirs('../1%IM/valid/' + cla, exist_ok=True)
    fileList = glob.glob('valid/' + cla + '/*.JPEG')
    chosen = np.random.choice(len(fileList), size=5, replace=False)
    chosenList = list(np.array(fileList)[chosen])
    print(chosenList)
    for choice in chosenList:
        shutil.copy(choice, '../1%IM/valid/' + cla)
