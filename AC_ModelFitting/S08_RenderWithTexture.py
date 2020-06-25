from S07_ToSilhouetteFitting_MultiFrames import *

if __name__ == '__main__':
    cfg = RenderingCfg()

    cfg.sigma = 0
    cfg.blurRange = 0
    cfg.plotStep = 20
    cfg.numCams = 16
    # low learning rate for pose optimization
    cfg.learningRate = 1e-3
    cfg.batchSize = 4
    # cfg.faces_per_pixel = 6 # for testing
    cfg.faces_per_pixel = 5  # for debugging
    # cfg.imgSize = 2160
    cfg.imgSize = 1080
    cfg.terminateLoss = 0.1
    cfg.lpSmootherW = 0.000001
    # cfg.normalSmootherW = 0.1
    cfg.normalSmootherW = 0.0
    cfg.numIterations = 500
    # cfg.numIterations = 20
    cfg.kpFixingWeight = 0

    # Set paths
    DATA_DIR = r"..\Data\TextureMap"
    obj_filename = os.path.join(DATA_DIR, "SMPLWithSocks.obj")
    fittedMeshFile = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\Final\06550\InterpolatedMesh.ply'

    # Setup
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    inputs = InputBundle()

    # Load obj file
    mesh = load_objs_as_meshes([obj_filename], device=device)
    texture_image=mesh.textures.maps_padded()
    normals = mesh.verts_normals_packed()

    fittedMesh = pv.PolyData(fittedMeshFile)
    pts = torch.tensor(fittedMesh.points, requires_grad=False, device=device)
    mesh = mesh.update_padded(pts[None,...])

    xyz = torch.from_numpy(np.float32([0, 0, 2000]))[None]

    # fixed light brightness
    #         diffuse = 1.2505254745483398
    #         ambient = 0.5083667039871216
    #         specular = 0.0
    diffuse = 0.0
    # ambient = 1.0
    ambient = 1.0

    specular = 0.0
    s = specular * torch.from_numpy(np.ones((1, 3)).astype(np.float32)).to(device)
    d = diffuse * torch.from_numpy(np.ones((1, 3)).astype(np.float32)).to(device)
    a = ambient * torch.from_numpy(np.ones((1, 3)).astype(np.float32)).to(device)
    light = PointLights(device=device, location=xyz, specular_color=s, ambient_color=a, diffuse_color=d)

    actual_img_shape = (2160, 4000)
    cam_params, cams_torch = load_cameras(inputs.camParamF, device, actual_img_shape)
    cams = init_camera_batches(cams_torch, device)
    
    # build renderer
    rendererSynth = RendererWithTexture(device, lights=light, cfg=cfg)
    # rendererSynth = Renderer(device, cfg)

    # Change specular color to green and change material shininess
    # materials = Materials(
    #     device=device,
    #     specular_color=[[0.0, 1.0, 0.0]],
    #     shininess=10.0
    # )
    plt.imshow(texture_image.cpu().numpy()[0, ..., :3])
    plt.show()

    images = []
    with torch.no_grad():
        for iCam in range(len(cams)):
            image_cur = rendererSynth.renderer(mesh,  cameras=cams[iCam]
                # materials=materials,
            )
            plt.imshow(image_cur.cpu().numpy()[0,...,:3])
            plt.show()
            images.append(image_cur.cpu().detach().numpy())
        images = np.concatenate(images, axis=0)
