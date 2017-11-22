'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _Tensor = require('../../Tensor');

var _Tensor2 = _interopRequireDefault(_Tensor);

var _Layer = require('../../Layer');

var _Layer2 = _interopRequireDefault(_Layer);

var _ndarrayGemm = require('ndarray-gemm');

var _ndarrayGemm2 = _interopRequireDefault(_ndarrayGemm);

var _ndarrayOps = require('ndarray-ops');

var _ndarrayOps2 = _interopRequireDefault(_ndarrayOps);

var _ndarrayUnsqueeze = require('ndarray-unsqueeze');

var _ndarrayUnsqueeze2 = _interopRequireDefault(_ndarrayUnsqueeze);

var _ndarrayConcatRows = require('ndarray-concat-rows');

var _ndarrayConcatRows2 = _interopRequireDefault(_ndarrayConcatRows);

var _isEqual = require('lodash/isEqual');

var _isEqual2 = _interopRequireDefault(_isEqual);

var _range = require('lodash/range');

var _range2 = _interopRequireDefault(_range);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * Merge layer class
 */
class Merge extends _Layer2.default {
  /**
   * Creates a Merge layer
   *
   * @param {Object} [attrs] - layer config attributes
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = 'Merge';
    this.isMergeLayer = true;

    const { mode = 'sum', concat_axis = -1, dot_axes = -1 } = attrs;

    const availableModes = ['sum', 'mul', 'concat', 'ave', 'cos', 'dot', 'max'];
    if (availableModes.indexOf(mode) > -1) {
      this.mode = mode;
    } else {
      this.throwError(`${mode} not available.`);
    }

    // no mini-batch axis here, so we subtract 1 if given axis > 0
    this.concatAxis = concat_axis <= 0 ? concat_axis : concat_axis - 1;
    if (Array.isArray(dot_axes)) {
      this.dotAxes = [dot_axes[0] <= 0 ? dot_axes[0] : dot_axes[0] - 1, dot_axes[1] <= 0 ? dot_axes[1] : dot_axes[1] - 1];
    } else {
      this.dotAxes = [dot_axes <= 0 ? dot_axes : dot_axes - 1, dot_axes <= 0 ? dot_axes : dot_axes - 1];
    }
  }

  /**
   * Internal method for validating inputs
   *
   * @param {Tensor[]} inputs
   * @returns {boolean}
   */
  _validateInputs(inputs) {
    const shapes = inputs.map(x => x.tensor.shape.slice());
    if (['sum', 'mul', 'ave', 'cos', 'max'].indexOf(this.mode) > -1) {
      if (!shapes.every(shape => (0, _isEqual2.default)(shape, shapes[0]))) {
        this.throwError(`All input shapes must be the same for mode ${this.mode}.`);
      }
    }
    if (['cos', 'dot'].indexOf(this.mode) > -1) {
      if (inputs.length !== 2) {
        this.throwError(`Exactly 2 inputs required for mode ${this.mode}.`);
      }
      if (this.dotAxes[0] < 0) {
        this.dotAxes[0] = shapes[0].length + this.dotAxes[0];
      }
      if (this.dotAxes[1] < 0) {
        this.dotAxes[1] = shapes[1].length + this.dotAxes[1];
      }
      if (shapes[0][this.dotAxes[0]] !== shapes[1][this.dotAxes[1]]) {
        this.throwError('Dimensions incompatibility using dot mode.');
      }
    } else if (this.mode === 'concat') {
      let nonConcatShapes = shapes.slice();
      let _concatAxis = this.concatAxis < 0 ? nonConcatShapes[0].length + this.concatAxis : this.concatAxis;
      if (this.concatAxis === 0) _concatAxis = 0;
      (0, _range2.default)(nonConcatShapes.length).forEach(i => {
        nonConcatShapes[i].splice(_concatAxis, 1);
      });
      if (!nonConcatShapes.every(shape => (0, _isEqual2.default)(shape, nonConcatShapes[0]))) {
        this.throwError('In concat mode, all shapes must be the same except along the concat axis.');
      }
    }
    return true;
  }

  /**
   * Method for layer computational logic
   *
   * @param {Tensor[]} inputs
   * @returns {Tensor}
   */
  call(inputs) {
    const valid = this._validateInputs(inputs);
    if (!valid) {
      this.throwError('Invalid inputs to call method.');
    }

    let output;
    let outputShape;
    if (['sum', 'mul', 'ave', 'max'].indexOf(this.mode) > -1) {
      outputShape = inputs[0].tensor.shape.slice();
      output = new _Tensor2.default([], outputShape);
    } else if (this.mode === 'concat') {
      outputShape = inputs[0].tensor.shape.slice();
      let _concatAxis = this.concatAxis < 0 ? outputShape.length + this.concatAxis : this.concatAxis;
      if (this.concatAxis === 0) _concatAxis = 0;
      inputs.slice(1, inputs.length).forEach(x => {
        const d = x.tensor.shape.slice()[_concatAxis];
        outputShape[_concatAxis] += d;
      });
      output = new _Tensor2.default([], outputShape);
    } else if (['cos', 'dot'].indexOf(this.mode) > -1) {
      let shape1 = inputs[0].tensor.shape.slice();
      let shape2 = inputs[1].tensor.shape.slice();
      shape1.splice(this.dotAxes[0], 1);
      shape2.splice(this.dotAxes[1], 1);
      outputShape = shape1.concat(shape2);
      if (outputShape.length === 1) {
        outputShape.push(1);
      }
      output = new _Tensor2.default([], outputShape);
    }

    if (this.mode === 'sum') {
      for (let i = 0; i < inputs.length; i++) {
        _ndarrayOps2.default.addeq(output.tensor, inputs[i].tensor);
      }
    } else if (this.mode === 'mul') {
      _ndarrayOps2.default.assigns(output.tensor, 1.0);
      for (let i = 0; i < inputs.length; i++) {
        _ndarrayOps2.default.muleq(output.tensor, inputs[i].tensor);
      }
    } else if (this.mode === 'ave') {
      for (let i = 0; i < inputs.length; i++) {
        _ndarrayOps2.default.addeq(output.tensor, inputs[i].tensor);
      }
      _ndarrayOps2.default.divseq(output.tensor, inputs.length);
    } else if (this.mode === 'max') {
      _ndarrayOps2.default.assign(output.tensor, inputs[0].tensor);
      for (let i = 1; i < inputs.length; i++) {
        _ndarrayOps2.default.maxeq(output.tensor, inputs[i].tensor);
      }
    } else if (this.mode === 'concat') {
      let _concatAxis = this.concatAxis < 0 ? inputs[0].tensor.shape.length + this.concatAxis : this.concatAxis;
      if (this.concatAxis === 0) _concatAxis = 0;
      if (_concatAxis === 0) {
        (0, _ndarrayConcatRows2.default)(output.tensor, inputs.map(x => x.tensor));
      } else {
        let dimsAxisSwap = [_concatAxis];
        for (let i = 0; i < inputs[0].tensor.shape.length; i++) {
          if (i !== _concatAxis) dimsAxisSwap.push(i);
        }
        (0, _ndarrayConcatRows2.default)(output.tensor.transpose(...dimsAxisSwap), inputs.map(x => x.tensor.transpose(...dimsAxisSwap)));
      }
    } else if (this.mode === 'dot') {
      if (inputs[0].tensor.shape.length === 2 && inputs[1].tensor.shape.length === 2) {
        if (this.dotAxes[0] === 0 && this.dotAxes[1] === 0) {
          (0, _ndarrayGemm2.default)(output.tensor, inputs[0].tensor.transpose(1, 0), inputs[1].tensor);
        } else if (this.dotAxes[0] === 1 && this.dotAxes[1] === 1) {
          (0, _ndarrayGemm2.default)(output.tensor, inputs[0].tensor, inputs[1].tensor.transpose(1, 0));
        }
      } else {
        this.throwError('dot mode for 3+ dim tensors not yet implemented.');
      }
    } else if (this.mode === 'cos') {
      if (inputs[0].tensor.shape.length === 2 && inputs[1].tensor.shape.length === 2) {
        let a = new _Tensor2.default([], output.tensor.shape);
        let b = new _Tensor2.default([], output.tensor.shape);
        if (this.dotAxes[0] === 0 && this.dotAxes[1] === 0) {
          (0, _ndarrayGemm2.default)(a.tensor, inputs[0].tensor.transpose(1, 0), inputs[0].tensor);
          (0, _ndarrayGemm2.default)(b.tensor, inputs[1].tensor.transpose(1, 0), inputs[1].tensor);
          (0, _ndarrayGemm2.default)(output.tensor, inputs[0].tensor.transpose(1, 0), inputs[1].tensor);
        } else if (this.dotAxes[0] === 1 && this.dotAxes[1] === 1) {
          (0, _ndarrayGemm2.default)(a.tensor, inputs[0].tensor, inputs[0].tensor.transpose(1, 0));
          (0, _ndarrayGemm2.default)(b.tensor, inputs[1].tensor, inputs[1].tensor.transpose(1, 0));
          (0, _ndarrayGemm2.default)(output.tensor, inputs[0].tensor, inputs[1].tensor.transpose(1, 0));
        }
        _ndarrayOps2.default.muleq(a.tensor, b.tensor);
        _ndarrayOps2.default.sqrteq(a.tensor);
        _ndarrayOps2.default.diveq(output.tensor, a.tensor);
        output.tensor = (0, _ndarrayUnsqueeze2.default)(output.tensor, 0);
      } else {
        this.throwError('cos mode for 3+ dim tensors not yet implemented.');
      }
    }

    return output;
  }
}
exports.default = Merge;