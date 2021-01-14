import subprocess, json

def ARAPDeformation(inObjMesh, inTargetObjMesh, outObjMesh, corresIds):
    json.dump(corresIds, open('ARAP_Corrs.json', 'w'))
    cmd = ['ARAPDemformationWithCorres_bin', inObjMesh, inTargetObjMesh, outObjMesh, 'ARAP_Corrs.json']
    print(*cmd)
    subprocess.call(cmd)