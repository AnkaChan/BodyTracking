import numpy as np
import pickle
import torch
from torch.nn import Module
import os


class SMPLModel(Module):
  def __init__(self, device=None, model_path='./model.pkl', personalShape=None, unitMM=False):
    
    super(SMPLModel, self).__init__()
    # with open(model_path, 'rb') as f:
    #   params = pickle.load(f, encoding='iso-8859-1')
    # self.J_regressor = torch.from_numpy(
    #   np.array(params['J_regressor'].todense())
    # ).type(torch.float64)
    # self.weights = torch.from_numpy(params['weights']).type(torch.float64)
    # self.posedirs = torch.from_numpy(params['posedirs']).type(torch.float64)
    # self.v_template = torch.from_numpy(params['v_template']).type(torch.float64)
    # self.shapedirs = torch.from_numpy(np.array(params['shapedirs'])).type(torch.float64)
    # self.kintree_table = params['kintree_table']
    # self.faces = params['f']
    data = np.load(model_path)
    self.J_regressor = torch.from_numpy(data['JRegressor']).type(torch.float64)
    self.weights = torch.from_numpy(data['Weights']).type(torch.float64)
    self.posedirs = torch.from_numpy(data['PoseBlendShapes']).type(torch.float64)
    self.v_template = torch.from_numpy(data['VTemplate']).type(torch.float64)
    self.shapedirs = torch.from_numpy(data['ShapeBlendShapes']).type(torch.float64)
    self.faces = data['Faces']
    self.parent = torch.from_numpy(data['ParentTable']).type(torch.int64)
    self.personalShape = personalShape

    if unitMM:
      self.v_template = self.v_template * 1000
      self.posedirs = self.posedirs * 1000
      self.shapedirs = self.shapedirs * 1000

    faces = []
    fId = 0
    while fId < self.faces.shape[0]:
      numFVs = self.faces[fId]
      face = []
      fId += 1
      for i in range(numFVs):
        face.append(self.faces[fId])
        fId += 1

      faces.append(face)

    self.faces = torch.from_numpy(np.array(faces)).type(torch.int64)

    self.numJoints = self.weights.shape[1]

    self.device = device if device is not None else torch.device('cpu')
    for name in ['J_regressor', 'weights', 'posedirs', 'v_template', 'shapedirs']:
      _tensor = getattr(self, name)
      setattr(self, name, _tensor.to(device))

  # @staticmethod
  def rodrigues(self, r):
    """
    Rodrigues' rotation formula that turns axis-angle tensor into rotation
    matrix in a batch-ed manner.

    Parameter:
    ----------
    r: Axis-angle rotation tensor of shape [batch_size, 1, 3].

    Return:
    -------
    Rotation matrix of shape [batch_size, 3, 3].

    """
    #r = r.to(self.device)
    eps = r.clone().normal_(std=1e-8)
    # theta = torch.norm(r + eps, dim=(1, 2), keepdim=True)  # dim cannot be tuple
    theta = torch.norm((r + eps).cpu(), dim=(1, 2), keepdim=True).to(self.device)
    theta_dim = theta.shape[0]
    r_hat = r / theta
    cos = torch.cos(theta)
    z_stick = torch.zeros(theta_dim, dtype=torch.float64).to(r.device)
    m = torch.stack(
      (z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1], r_hat[:, 0, 2], z_stick,
       -r_hat[:, 0, 0], -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick), dim=1)
    m = torch.reshape(m, (-1, 3, 3))
    i_cube = (torch.eye(3, dtype=torch.float64).unsqueeze(dim=0) \
             + torch.zeros((theta_dim, 3, 3), dtype=torch.float64)).to(r.device)
    A = r_hat.permute(0, 2, 1)
    dot = torch.matmul(A, r_hat)
    R = cos * i_cube + (1 - cos) * dot + torch.sin(theta) * m
    return R

  @staticmethod
  def with_zeros(x):
    """
    Append a [0, 0, 0, 1] tensor to a [3, 4] tensor.

    Parameter:
    ---------
    x: Tensor to be appended.

    Return:
    ------
    Tensor after appending of shape [4,4]

    """
    ones = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64).to(x.device)
    ret = torch.cat((x, ones), dim=0)
    return ret

  @staticmethod
  def pack(x):
    """
    Append zero tensors of shape [4, 3] to a batch of [4, 1] shape tensor.

    Parameter:
    ----------
    x: A tensor of shape [batch_size, 4, 1]

    Return:
    ------
    A tensor of shape [batch_size, 4, 4] after appending.

    """
    zeros43 = torch.zeros((x.shape[0], 4, 3), dtype=torch.float64).to(x.device)
    ret = torch.cat((zeros43, x), dim=2)
    return ret

  def write_obj(self, verts, file_name):
    # with open(file_name, 'w') as fp:
    #   for v in verts:
    #     fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
    #
    #   for f in self.faces + 1:
    #     fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
    with open(file_name, 'w') as fp:
      for v in verts:
        fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
      for f in self.faces:
        fp.write('f ')
        for fv in f:
          fp.write('%d ' % (fv + 1))
        fp.write('\n')

  def forward(self, betas, pose, trans, simplify=False, returnDeformedJoints=False):
    """
          Construct a compute graph that takes in parameters and outputs a tensor as
          model vertices. Face indices are also returned as a numpy ndarray.

          Prameters:
          ---------
          pose: Also known as 'theta', a [24,3] tensor indicating child joint rotation
          relative to parent joint. For root joint it's global orientation.
          Represented in a axis-angle format.

          betas: Parameter for model shape. A tensor of shape [10] as coefficients of
          PCA components. Only 10 components were released by SMPL author.

          trans: Global translation tensor of shape [3].

          Return:
          ------
          A tensor for vertices, and a numpy ndarray as face indices.

    """
    # id_to_col = {self.kintree_table[1, i]: i
    #              for i in range(self.numJoints)}
    # parent = {
    #   i: id_to_col[self.kintree_table[0, i]]
    #   for i in range(1, self.kintree_table.shape[1])
    # }
    v_shaped = torch.tensordot(self.shapedirs, betas, dims=([2], [0])) + self.v_template
    J = torch.matmul(self.J_regressor, v_shaped)
    R_cube_big = self.rodrigues(pose.view(-1, 1, 3))

    if simplify:
      v_posed = v_shaped
    else:
      R_cube = R_cube_big[1:]
      I_cube = (torch.eye(3, dtype=torch.float64).unsqueeze(dim=0) + \
        torch.zeros((R_cube.shape[0], 3, 3), dtype=torch.float64)).to(self.device)
      lrotmin = torch.reshape(R_cube - I_cube, (-1, 1)).squeeze()
      v_posed = v_shaped + torch.tensordot(self.posedirs, lrotmin, dims=([2], [0]))

    if self.personalShape is not  None:
      v_posed = v_posed + self.personalShape

    results = []
    results.append(
      self.with_zeros(torch.cat((R_cube_big[0], torch.reshape(J[0, :], (3, 1))), dim=1))
    )
    for i in range(1, self.numJoints):
      results.append(
        torch.matmul(
          results[self.parent[i]],
          self.with_zeros(
            torch.cat(
              (R_cube_big[i], torch.reshape(J[i, :] - J[self.parent[i], :], (3, 1))),
              dim=1
            )
          )
        )
      )

    stacked = torch.stack(results, dim=0)
    results = stacked - \
      self.pack(
        torch.matmul(
          stacked,
          torch.reshape(
            torch.cat((J, torch.zeros((self.numJoints, 1), dtype=torch.float64).to(self.device)), dim=1),
            (self.numJoints, 4, 1)
          )
        )
      )
    T = torch.tensordot(self.weights, results, dims=([1], [0]))
    rest_shape_h = torch.cat(
      (v_posed, torch.ones((v_posed.shape[0], 1), dtype=torch.float64).to(self.device)), dim=1
    )
    v = torch.matmul(T, torch.reshape(rest_shape_h, (-1, 4, 1)))
    v = torch.reshape(v, (-1, 4))[:, :3]
    result = v + torch.reshape(trans, (1, 3))
    if not returnDeformedJoints:
      return result
    else:
      jointsH = torch.cat([
        torch.transpose(J, 0, 1),
        torch.ones((1, J.shape[0]), dtype=torch.float64, requires_grad=False, device=self.device)
      ], dim=0)
      # for i in range(J.shape[0]):
      #   jointsH[:, i] = results[i, ...] @ jointsH[:, i]
      jointsH = torch.matmul(results, torch.transpose(jointsH, 0, 1)[...,None])
      # newJs = torch.transpose(jointsH[:3, :], 0, 1) + torch.reshape(trans, (1, 3))
      newJs = jointsH[:, :3, 0] + torch.reshape(trans, (1, 3))

      return result, newJs

  def getTransformation(self, betas, pose, trans, simplify=False, returnPoseBlendShape=False):
    """
          Construct a compute graph that takes in parameters and outputs a tensor as
          model vertices. Face indices are also returned as a numpy ndarray.

          Prameters:
          ---------
          pose: Also known as 'theta', a [24,3] tensor indicating child joint rotation
          relative to parent joint. For root joint it's global orientation.
          Represented in a axis-angle format.

          betas: Parameter for model shape. A tensor of shape [10] as coefficients of
          PCA components. Only 10 components were released by SMPL author.

          trans: Global translation tensor of shape [3].

          Return:
          ------
          A tensor for vertices, and a numpy ndarray as face indices.

    """
    # id_to_col = {self.kintree_table[1, i]: i
    #              for i in range(self.numJoints)}
    # parent = {
    #   i: id_to_col[self.kintree_table[0, i]]
    #   for i in range(1, self.kintree_table.shape[1])
    # }
    v_shaped = torch.tensordot(self.shapedirs, betas, dims=([2], [0])) + self.v_template
    J = torch.matmul(self.J_regressor, v_shaped)
    R_cube_big = self.rodrigues(pose.view(-1, 1, 3))

    if simplify or not returnPoseBlendShape:
      v_posed = v_shaped
    else:
      R_cube = R_cube_big[1:]
      I_cube = (torch.eye(3, dtype=torch.float64).unsqueeze(dim=0) + \
        torch.zeros((R_cube.shape[0], 3, 3), dtype=torch.float64)).to(self.device)
      lrotmin = torch.reshape(R_cube - I_cube, (-1, 1)).squeeze()
      v_posed = v_shaped + torch.tensordot(self.posedirs, lrotmin, dims=([2], [0]))

    if self.personalShape is not  None:
      v_posed = v_posed + self.personalShape

    results = []
    results.append(
      self.with_zeros(torch.cat((R_cube_big[0], torch.reshape(J[0, :], (3, 1))), dim=1))
    )
    for i in range(1, self.numJoints):
      results.append(
        torch.matmul(
          results[self.parent[i]],
          self.with_zeros(
            torch.cat(
              (R_cube_big[i], torch.reshape(J[i, :] - J[self.parent[i], :], (3, 1))),
              dim=1
            )
          )
        )
      )

    stacked = torch.stack(results, dim=0)
    results = stacked - \
      self.pack(
        torch.matmul(
          stacked,
          torch.reshape(
            torch.cat((J, torch.zeros((self.numJoints, 1), dtype=torch.float64).to(self.device)), dim=1),
            (self.numJoints, 4, 1)
          )
        )
      )
    T = torch.tensordot(self.weights, results, dims=([1], [0]))
    
    T[:, :3, 3] += trans
    
    if returnPoseBlendShape:
      return T, torch.tensordot(self.posedirs, lrotmin, dims=([2], [0])), v_shaped
    else:
      return T

  def getTransformation(self, betas, pose, trans, simplify=False, returnPoseBlendShape=False):
    """
          Construct a compute graph that takes in parameters and outputs a tensor as
          model vertices. Face indices are also returned as a numpy ndarray.

          Prameters:
          ---------
          pose: Also known as 'theta', a [24,3] tensor indicating child joint rotation
          relative to parent joint. For root joint it's global orientation.
          Represented in a axis-angle format.

          betas: Parameter for model shape. A tensor of shape [10] as coefficients of
          PCA components. Only 10 components were released by SMPL author.

          trans: Global translation tensor of shape [3].

          Return:
          ------
          A tensor for vertices, and a numpy ndarray as face indices.

    """
    # id_to_col = {self.kintree_table[1, i]: i
    #              for i in range(self.numJoints)}
    # parent = {
    #   i: id_to_col[self.kintree_table[0, i]]
    #   for i in range(1, self.kintree_table.shape[1])
    # }
    v_shaped = torch.tensordot(self.shapedirs, betas, dims=([2], [0])) + self.v_template
    J = torch.matmul(self.J_regressor, v_shaped)
    R_cube_big = self.rodrigues(pose.view(-1, 1, 3))

    if simplify or not returnPoseBlendShape:
      v_posed = v_shaped
    else:
      R_cube = R_cube_big[1:]
      I_cube = (torch.eye(3, dtype=torch.float64).unsqueeze(dim=0) + \
        torch.zeros((R_cube.shape[0], 3, 3), dtype=torch.float64)).to(self.device)
      lrotmin = torch.reshape(R_cube - I_cube, (-1, 1)).squeeze()
      v_posed = v_shaped + torch.tensordot(self.posedirs, lrotmin, dims=([2], [0]))

    if self.personalShape is not  None:
      v_posed = v_posed + self.personalShape

    results = []
    results.append(
      self.with_zeros(torch.cat((R_cube_big[0], torch.reshape(J[0, :], (3, 1))), dim=1))
    )
    for i in range(1, self.numJoints):
      results.append(
        torch.matmul(
          results[self.parent[i]],
          self.with_zeros(
            torch.cat(
              (R_cube_big[i], torch.reshape(J[i, :] - J[self.parent[i], :], (3, 1))),
              dim=1
            )
          )
        )
      )

    stacked = torch.stack(results, dim=0)
    results = stacked - \
      self.pack(
        torch.matmul(
          stacked,
          torch.reshape(
            torch.cat((J, torch.zeros((self.numJoints, 1), dtype=torch.float64).to(self.device)), dim=1),
            (self.numJoints, 4, 1)
          )
        )
      )
    T = torch.tensordot(self.weights, results, dims=([1], [0]))
    
    T[:, :3, 3] += trans
    
    if returnPoseBlendShape:
      return T, torch.tensordot(self.posedirs, lrotmin, dims=([2], [0])), v_shaped
    else:
      return T

  def getTransformation(self, betas, pose, trans, simplify=False, returnPoseBlendShape=False):
    """
          Construct a compute graph that takes in parameters and outputs a tensor as
          model vertices. Face indices are also returned as a numpy ndarray.

          Prameters:
          ---------
          pose: Also known as 'theta', a [24,3] tensor indicating child joint rotation
          relative to parent joint. For root joint it's global orientation.
          Represented in a axis-angle format.

          betas: Parameter for model shape. A tensor of shape [10] as coefficients of
          PCA components. Only 10 components were released by SMPL author.

          trans: Global translation tensor of shape [3].

          Return:
          ------
          A tensor for vertices, and a numpy ndarray as face indices.

    """
    # id_to_col = {self.kintree_table[1, i]: i
    #              for i in range(self.numJoints)}
    # parent = {
    #   i: id_to_col[self.kintree_table[0, i]]
    #   for i in range(1, self.kintree_table.shape[1])
    # }
    v_shaped = torch.tensordot(self.shapedirs, betas, dims=([2], [0])) + self.v_template
    J = torch.matmul(self.J_regressor, v_shaped)
    R_cube_big = self.rodrigues(pose.view(-1, 1, 3))

    if simplify or not returnPoseBlendShape:
      v_posed = v_shaped
    else:
      R_cube = R_cube_big[1:]
      I_cube = (torch.eye(3, dtype=torch.float64).unsqueeze(dim=0) + \
        torch.zeros((R_cube.shape[0], 3, 3), dtype=torch.float64)).to(self.device)
      lrotmin = torch.reshape(R_cube - I_cube, (-1, 1)).squeeze()
      v_posed = v_shaped + torch.tensordot(self.posedirs, lrotmin, dims=([2], [0]))

    if self.personalShape is not  None:
      v_posed = v_posed + self.personalShape

    results = []
    results.append(
      self.with_zeros(torch.cat((R_cube_big[0], torch.reshape(J[0, :], (3, 1))), dim=1))
    )
    for i in range(1, self.numJoints):
      results.append(
        torch.matmul(
          results[self.parent[i]],
          self.with_zeros(
            torch.cat(
              (R_cube_big[i], torch.reshape(J[i, :] - J[self.parent[i], :], (3, 1))),
              dim=1
            )
          )
        )
      )

    stacked = torch.stack(results, dim=0)
    results = stacked - \
      self.pack(
        torch.matmul(
          stacked,
          torch.reshape(
            torch.cat((J, torch.zeros((self.numJoints, 1), dtype=torch.float64).to(self.device)), dim=1),
            (self.numJoints, 4, 1)
          )
        )
      )
    T = torch.tensordot(self.weights, results, dims=([1], [0]))
    
    T[:, :3, 3] += trans
    
    if returnPoseBlendShape:
      return T, torch.tensordot(self.posedirs, lrotmin, dims=([2], [0])), v_shaped
    else:
      return T


def test_gpu(gpu_id=[1], modelPath = r'C:\Code\MyRepo\ChbCapture\06_Deformation\SMPL_Socks\SMPLSH\SmplshModel.npz'):
  if len(gpu_id) > 0 and torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id[0])
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')
  print(device)

  # pose_size = 72
  pose_size = 3 * 52
  beta_size = 10


  restposeShape = np.zeros((6750, 3))

  restposeShape[0, :] = [0.100, 0.1100, 0.1100]
  restposeShape[3513, :] = [-0.100, 0.1100, 0.1100]
  restposeShape = restposeShape * 1000

  restposeShape = torch.from_numpy(restposeShape) \
    .type(torch.float64).to(device)

  np.random.seed(9608)
  pose = torch.from_numpy((np.random.rand(pose_size) - 0.5) * 0.8)\
          .type(torch.float64).to(device)
  betas = torch.from_numpy((np.random.rand(beta_size) - 0.5) * 0.06) \
          .type(torch.float64).to(device)
  trans = torch.from_numpy(np.zeros(3)).type(torch.float64).to(device)
  outmesh_path = './smplsh_torch_withPersonalShapeMM.obj'

  model = SMPLModel(device=device, model_path=modelPath, personalShape=restposeShape, unitMM=True)
  result = model(betas, pose, trans)
  model.write_obj(result, outmesh_path)


if __name__ == '__main__':
  test_gpu([1])
