from Sync import *
from os.path import join

if __name__ == "__main__":
	syncPts = {"A":189 - 2, "K":484 - 2, "L":419 - 2, "M":376 - 2, "N":329 - 2, "O":287 - 2, "P":240 - 2}
	inFolder = r'Z:\2020_03_06_TextureTest\Converted'
	seqFolders = glob.glob(join(inFolder, '*'))

	print('\n'.join(seqFolders))


	for folder in seqFolders:
		folderName = Path(folder).stem
		print("Sync", folder, 'at', syncPts[folderName])
		syncOneSequence(folder, syncPts[folderName], ext = "pgm", addFolderName=True, zfillNum=5)