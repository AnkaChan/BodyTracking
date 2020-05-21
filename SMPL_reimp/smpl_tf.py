import tensorflow as tf
import numpy as np
import pickle


def rodrigues(r, dtype=tf.float64):
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
  theta = tf.norm(r + tf.random_normal(r.shape, 0, 1e-8, dtype=dtype), axis=(1, 2), keepdims=True)
  # avoid divide by zero
  r_hat = r / theta
  cos = tf.cos(theta)
  z_stick = tf.zeros(theta.get_shape().as_list()[0], dtype=dtype)
  m = tf.stack(
    (z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1], r_hat[:, 0, 2], z_stick,
     -r_hat[:, 0, 0], -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick), axis=1)
  m = tf.reshape(m, (-1, 3, 3))
  i_cube = tf.expand_dims(tf.eye(3, dtype=dtype), axis=0) + tf.zeros(
    (theta.get_shape().as_list()[0], 3, 3), dtype=dtype)
  A = tf.transpose(r_hat, (0, 2, 1))
  B = r_hat
  dot = tf.matmul(A, B)
  R = cos * i_cube + (1 - cos) * dot + tf.sin(theta) * m
  return R

# def rodrigues_batch(r):
#   """
#   Rodrigues' rotation formula that turns axis-angle tensor into rotation
#   matrix in a batch-ed manner.
#
#   Parameter:
#   ----------
#   r: Axis-angle rotation tensor of shape [batch_size, 1, 3].
#
#   Return:
#   -------
#   Rotation matrix of shape [batch_size, 3, 3].
#
#   """
#   theta = tf.norm(r + tf.random_normal(r.shape, 0, 1e-8, dtype=tf.float64), axis=(-2, -1), keepdims=True)
#   # avoid divide by zero
#   r_hat = r / theta
#   cos = tf.cos(theta)
#   z_stick = tf.zeros(theta.get_shape().as_list()[0:2], dtype=tf.float64)
#   m = tf.stack(
#     (z_stick, -r_hat[:, :, 0, 2], r_hat[:, :, 0, 1], r_hat[:, :, 0, 2], z_stick,
#      -r_hat[:, :, 0, 0], -r_hat[:, :, 0, 1], r_hat[:, :, 0, 0], z_stick), axis=-2)
#   m = tf.reshape(m, (m.shape[0], -1, 3, 3))
#   i_cube = tf.expand_dims(tf.eye(3, dtype=tf.float64), axis=0) + tf.zeros(
#     (theta.get_shape().as_list()[0], 3, 3), dtype=tf.float64)
#   A = tf.transpose(r_hat, (0, 2, 1))
#   B = r_hat
#   dot = tf.matmul(A, B)
#   R = cos * i_cube + (1 - cos) * dot + tf.sin(theta) * m
#   return R

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
  ret = tf.concat(
    (x, tf.constant([[0.0, 0.0, 0.0, 1.0]], dtype=tf.float64)),
    axis=0
  )
  return ret


def with_zeros_batch(x, dtype=tf.float64):
  """
  Append a [0, 0, 0, 1] tensor to a [3, 4] tensor in a batch-ed manner.

  Parameter:
  ---------
  x: Tensor to be appended.

  Return:
  ------
  Tensor after appending of shape [4,4]

  """
  ret = tf.concat(
    (x, tf.constant([[[0.0, 0.0, 0.0, 1.0]] for i in range(x.shape[0])], dtype=dtype)),
    axis=1
  )
  return ret

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
  ret = tf.concat(
    (tf.zeros((x.get_shape().as_list()[0], 4, 3), dtype=tf.float64), x),
    axis=2
  )
  return ret

def pack_batch(x, dtype=tf.float64):
  """
  Append zero tensors of shape [4, 3] to a batch of [4, 1] shape tensor.

  Parameter:
  ----------
  x: A tensor of shape [batch_size, 4, 1]

  Return:
  ------
  A tensor of shape [batch_size, 4, 4] after appending.

  """
  ret = tf.concat(
    (tf.zeros((x.get_shape().as_list()[0], x.get_shape().as_list()[1], 4, 3), dtype=dtype), x),
    axis=3
  )
  return ret


def smpl_model(model_path, betas, pose, trans, simplify=False):
  """
  Construct a compute graph that takes in parameters and outputs a tensor as
  model vertices. Face indices are also returned as a numpy ndarray.

  Parameters:
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
  # For detailed comments see smpl_np.py
  with open(model_path, 'rb') as f:
    params = pickle.load(f, encoding='iso-8859-1')

  J_regressor = tf.constant(
    np.array(params['J_regressor'].todense(),
    dtype=np.float64)
  )
  weights = tf.constant(params['weights'], dtype=np.float64)
  posedirs = tf.constant(params['posedirs'], dtype=np.float64)
  v_template = tf.constant(params['v_template'], dtype=np.float64)
  shapedirs = tf.constant(np.array(params['shapedirs']), dtype=np.float64)
  f = params['f']

  kintree_table = params['kintree_table']
  id_to_col = {kintree_table[1, i]: i for i in range(kintree_table.shape[1])}
  parent = {
    i: id_to_col[kintree_table[0, i]]
    for i in range(1, kintree_table.shape[1])
  }
  v_shaped = tf.tensordot(shapedirs, betas, axes=[[2], [0]]) + v_template
  J = tf.matmul(J_regressor, v_shaped)
  pose_cube = tf.reshape(pose, (-1, 1, 3))
  R_cube_big = rodrigues(pose_cube)
  if simplify:
    v_posed = v_shaped
  else:
    R_cube = R_cube_big[1:]
    I_cube = tf.expand_dims(tf.eye(3, dtype=tf.float64), axis=0) + \
             tf.zeros((R_cube.get_shape()[0], 3, 3), dtype=tf.float64)
    lrotmin = tf.squeeze(tf.reshape((R_cube - I_cube), (-1, 1)))
    v_posed = v_shaped + tf.tensordot(posedirs, lrotmin, axes=[[2], [0]])
  results = []
  results.append(
    with_zeros(tf.concat((R_cube_big[0], tf.reshape(J[0, :], (3, 1))), axis=1))
  )
  for i in range(1, kintree_table.shape[1]):
    results.append(
      tf.matmul(
        results[parent[i]],
        with_zeros(
          tf.concat(
            (R_cube_big[i], tf.reshape(J[i, :] - J[parent[i], :], (3, 1))),
            axis=1
          )
        )
      )
    )
  stacked = tf.stack(results, axis=0)
  results = stacked - \
            pack(
              tf.matmul(
                stacked,
                tf.reshape(
                  tf.concat((J, tf.zeros((24, 1), dtype=tf.float64)), axis=1),
                  (24, 4, 1)
                )
              )
            )
  T = tf.tensordot(weights, results, axes=((1), (0)))
  rest_shape_h = tf.concat(
    (v_posed, tf.ones((v_posed.get_shape().as_list()[0], 1), dtype=tf.float64)),
    axis=1
  )
  v = tf.matmul(T, tf.reshape(rest_shape_h, (-1, 4, 1)))
  v = tf.reshape(v, (-1, 4))[:, :3]
  result = v + tf.reshape(trans, (1, 3))
  return result, f

def smpl_model_tensor(model_path, betas, pose, trans, simplify=False, dtype=tf.float64):
  """
  Construct a compute graph that takes in parameters and outputs a tensor as
  model vertices. Face indices are also returned as a numpy ndarray.

  Parameters:
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

  numBatches = int(pose.shape[0])

  # For detailed comments see smpl_np.py
  with open(model_path, 'rb') as f:
    params = pickle.load(f, encoding='iso-8859-1')

  J_regressor = tf.constant(
    np.array(params['J_regressor'].todense(),dtype=np.float64), dtype=dtype)
  weights = tf.constant(params['weights'], dtype=dtype)
  posedirs = tf.constant(params['posedirs'], dtype=dtype)
  v_template = tf.constant(params['v_template'], dtype=dtype)
  shapedirs = tf.constant(np.array(params['shapedirs']), dtype=dtype)
  f = params['f']

  kintree_table = params['kintree_table']
  id_to_col = {kintree_table[1, i]: i for i in range(kintree_table.shape[1])}
  parent = {
    i: id_to_col[kintree_table[0, i]]
    for i in range(1, kintree_table.shape[1])
  }
  if len(betas.shape) == 2:
    # If betas if also a tensor for each frame
    v_shaped = tf.add(tf.tensordot(betas, shapedirs, axes=[[1], [2]]), v_template)
    J = tf.tensordot(v_shaped, J_regressor, axes=[[1], [1]])
    J = tf.transpose(J, [0, 2, 1])
  elif len(betas.shape) == 1:
    v_shaped = tf.tensordot(shapedirs, betas, axes=[[2], [0]]) + v_template
    J = tf.matmul(J_regressor, v_shaped)

  pose_cube = tf.reshape(pose, (-1, 1, 3))
  R_cube_big = rodrigues(pose_cube, dtype=dtype)
  R_cube_big = tf.reshape(R_cube_big, [pose.shape[0], int(int(pose.shape[1]) / 3), 3, 3])
  if simplify:
    v_posed = v_shaped
  else:
    R_cube = R_cube_big[:, 1:]
    R_cube = tf.reshape(R_cube, [-1, 3, 3])
    I_cube = tf.expand_dims(tf.eye(3, dtype=dtype), axis=0) + \
             tf.zeros((R_cube.get_shape()[0], 3, 3), dtype=dtype)
    lrotmin = tf.squeeze(tf.reshape((R_cube - I_cube), (numBatches, -1, 1)))
    v_posed = v_shaped + tf.tensordot(lrotmin, posedirs, axes=[[1], [2]])
  results = []
  results.append(
    with_zeros_batch(tf.concat((R_cube_big[:, 0], tf.reshape(J[:, 0, :], (numBatches, 3, 1))), axis=2), dtype=dtype)
  )
  for i in range(1, kintree_table.shape[1]):
    results.append(
      tf.matmul(
        results[parent[i]],
        with_zeros_batch(
          tf.concat(
            (R_cube_big[:, i], tf.reshape(J[:, i, :] - J[:, parent[i], :], (numBatches, 3, 1))),
            axis=2
          ),
          dtype=dtype
        )
      )
    )
  stacked = tf.stack(results, axis=0)
  stacked = tf.transpose(stacked, [1, 0, 2, 3])
  results = stacked - \
            pack_batch(
              tf.matmul(
                stacked,
                tf.reshape(
                  tf.concat((J, tf.zeros((numBatches, 24, 1), dtype=dtype)), axis=2),
                  (numBatches, 24, 4, 1)
                )
              ),
              dtype=dtype
            )
  T = tf.tensordot(results, weights, axes=((1), (1)))
  T = tf.transpose(T, [0,3,1,2])
  rest_shape_h = tf.concat(
    (v_posed, tf.ones((v_posed.get_shape().as_list()[0], v_posed.get_shape().as_list()[1], 1), dtype=dtype)),
    axis=2
  )
  v = tf.matmul(T, tf.reshape(rest_shape_h, (numBatches, -1, 4, 1)))
  v = tf.reshape(v, (numBatches, -1, 4))[:, :, :3]
  result = v + tf.reshape(trans, (numBatches, 1, 3))
  return result, f

if __name__ == '__main__':
  pose_size = 72
  beta_size = 10

  #np.random.seed(9608)
  pose = (np.random.rand(pose_size) - 0.5) * 1.5
  betas = (np.random.rand(beta_size) - 0.5) * 0.56

  pose = np.zeros(pose_size)
  betas = np.zeros(beta_size)

  trans = np.zeros(3)

  pose = tf.constant(pose, dtype=tf.float64)
  betas = tf.constant(betas, dtype=tf.float64)
  trans = tf.constant(trans, dtype=tf.float64)

  model_path = r'C:/Data/SMPL_python_v.1.0.0/smpl/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl'

  output, faces = smpl_model(model_path, betas, pose, trans, False)
  sess = tf.Session()
  result = sess.run(output)

  outmesh_path = './smpl_tf.obj'
  with open(outmesh_path, 'w') as fp:
    for v in result:
      fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

    for f in faces + 1:
      fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
