import pyvista as pv
import numpy as np
import itertools, os



if __name__ == '__main__':
    inCompleteMesh = r'C:\Code\MyRepo\03_capture\BodyTracking\Data\2020_12_27_betterCoarseMesh\Mesh1487\A00003052.obj'
    meshToTest = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487\TestDeletefaces\Test.obj'

    completeMesh = pv.PolyData(inCompleteMesh)
    testMesh = pv.PolyData(meshToTest)

    numRealPts = 1487
    threshold = 0.5

    diff = np.abs(completeMesh.points[:numRealPts, :]-testMesh.points[:numRealPts, :])
    maxDiff = np.max(diff)
    print("Max dfference:", maxDiff, ' happens at vertex: ', np.argmax(diff))

    maxId = np.argmax(diff)
    vId = int(maxId /3)
    print("before: ")

    assert maxDiff < threshold