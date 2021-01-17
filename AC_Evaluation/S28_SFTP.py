import paramiko
from Utility import *
import tqdm
# 获取Transport实例
tran = paramiko.Transport(('155.98.68.64', 22))

# 连接SSH服务端，使用password
tran.connect(username="GraphicLab", password='Graphic3335')

sftp = paramiko.SFTPClient.from_transport(tran)

# localpath = "main.py"
# remotepath = "/Z:/main.py"
#
# sftp.put(localpath, remotepath)
#
# tran.close()

inFolder = r'/Z:/shareZ/2019_12_13_Lada_Capture/Converted'
outFolder = r'E:\WorkingCopy\2019_12_12_LadaCapture'

frameNames = [str(fId).zfill(5) for fId in range(6141, 6141+2000)]

camNames = 'ABCDEFGHIJKLMNOP'
#range(len(inCamFolders))
camIds = [0,4,8,12,]
inCamFolders = [join(inFolder, camN) for camN in camNames]
outCamFolders = [join(outFolder, camN) for camN in camNames]

for iCam in camIds:
    os.makedirs(outCamFolders[iCam], exist_ok=True)
    print("Copying from:", inCamFolders[iCam])
    for frame in tqdm.tqdm(frameNames):
        remotepath = join(inCamFolders[iCam], camNames[iCam] + frame+'.pgm' )
        localpath = join(outCamFolders[iCam], camNames[iCam] + frame+'.pgm' )
        sftp.get(remotepath, localpath)

