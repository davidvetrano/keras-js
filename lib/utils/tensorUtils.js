'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _range2 = require('lodash/range');

var _range3 = _interopRequireDefault(_range2);

exports.checkShape = checkShape;
exports.data3DLayoutForGL = data3DLayoutForGL;
exports.createIndicesFor2DReshaped = createIndicesFor2DReshaped;

var _ndarray = require('ndarray');

var _ndarray2 = _interopRequireDefault(_ndarray);

var _ndarrayOps = require('ndarray-ops');

var _ndarrayOps2 = _interopRequireDefault(_ndarrayOps);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * Function to throw error if specified shape is incompatible with data
 *
 * @param {number[]} data
 * @param {number[]} shape
 */

function checkShape(data, shape) {
  if (data.length && shape.length && data.length !== shape.reduce((a, b) => a * b, 1)) {
    throw new Error('[Tensor] specified shape incompatible with data.');
  }
}

/**
 * Shuffle ndarray data layout for WebGL
 * - data for TEXTURE_2D_ARRAY or TEXTURE_3D laid out sequentially per-slice
 *
 * @param {TypedArray} typedarrayConstructor
 * @param {Object} arr - ndarray tensor
 * @param {number[]} shape
 */
function data3DLayoutForGL(typedarrayConstructor, arr, shape) {
  // must shuffle data layout for webgl
  //
  const data = new typedarrayConstructor(arr.data.length);
  const slice = (0, _ndarray2.default)(new typedarrayConstructor(shape[0] * shape[1]), [shape[0], shape[1]]);
  let offset = 0;
  for (let i = 0; i < shape[2]; i++) {
    _ndarrayOps2.default.assign(slice, arr.pick(null, null, i));
    data.set(slice.data, offset);
    offset += shape[0] * shape[1];
  }

  return data;
}

/**
 * Create indicesForReshaped for 2D reshaped tensor
 *
 * @param {number[]} shape
 * @param {boolean} square
 * @param {number} axis
 */
function createIndicesFor2DReshaped(shape, square = false, axis = -1) {
  const size = shape.reduce((a, b) => a * b, 1);
  const indicesRowArr = (0, _ndarray2.default)(new Int32Array(size), shape);
  const indicesColArr = (0, _ndarray2.default)(new Int32Array(size), shape);

  if (square) {
    // called by Tensor.reshapeTo2DSquare
    const squareDim = Math.ceil(Math.sqrt(size));
    const indicesRowArrReshaped = (0, _ndarray2.default)(new Int32Array(squareDim ** 2), [squareDim, squareDim]);
    const indicesColArrReshaped = (0, _ndarray2.default)(new Int32Array(squareDim ** 2), [squareDim, squareDim]);
    for (let i = 0; i < squareDim; i++) {
      _ndarrayOps2.default.assigns(indicesRowArrReshaped.pick(i, null), i);
    }
    for (let j = 0; j < squareDim; j++) {
      _ndarrayOps2.default.assigns(indicesColArrReshaped.pick(null, j), j);
    }
    indicesRowArr.data.set(indicesRowArrReshaped.data.subarray(0, indicesRowArr.size));
    indicesColArr.data.set(indicesColArrReshaped.data.subarray(0, indicesColArr.size));
  } else {
    // called by Tensor.reshapeTo2D
    if (axis < 0) {
      axis = shape.length + axis;
    }
    const axisSize = shape[axis];
    const otherAxes = [...shape.slice(0, axis), ...shape.slice(axis + 1)];
    const otherAxesSize = otherAxes.reduce((a, b) => a * b, 1);
    const indicesRowArrSlice = (0, _ndarray2.default)(new Int32Array((0, _range3.default)(otherAxesSize)), otherAxes);
    const axisSlices = Array(shape.length).fill(null);
    for (let n = 0; n < axisSize; n++) {
      axisSlices[axis] = n;
      _ndarrayOps2.default.assign(indicesRowArr.pick(...axisSlices), indicesRowArrSlice);
      _ndarrayOps2.default.assigns(indicesColArr.pick(...axisSlices), n);
    }
  }

  const indicesForReshaped = { row: indicesRowArr, col: indicesColArr };
  return indicesForReshaped;
}