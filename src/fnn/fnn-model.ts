import { Graph } from "../graph";
import { Mat, MatJson } from "../mat";
import { RandMat } from "../rand-mat";
import { Assertable } from "../utils/assertable";
import { NetOpts } from "../utils/net-opts";
import { ANN } from "./ann";

export interface FNNJson {
  /**
   * hidden
   */
  h: {
    /** Wh */
    w: Array<MatJson>;
    /** bh */
    b: Array<MatJson>;
  };
  /**
   * decoder
   */
  d: {
    /** Wh */
    w: MatJson;
    /** b */
    b: MatJson;
  };
}

export abstract class FNNModel extends Assertable implements ANN {

  protected architecture: { inputSize: number, hiddenUnits: Array<number>, outputSize: number };
  protected training: { alpha: number, lossClamp: number, loss: number };

  public model: { hidden: { Wh: Array<Mat>, bh: Array<Mat> }, decoder: { Wh: Mat, b: Mat } } =
    { hidden: { Wh: [], bh: [] }, decoder: { Wh: null, b: null } };

  protected graph: Graph;
  protected previousOutput: Mat;

  constructor(...args:
    [opt: NetOpts, json: FNNJson] |
    [opt: NetOpts]) {
    super();
    this.graph = new Graph();
    // 初期状態で生成
    if (args.length === 1) {
      this.initializeModelAsFreshInstance(args[0]);
    }
    // JSONオブジェクトから復元
    else {
      this.architecture = this.determineArchitectureProperties(args[0]);
      this.training = this.determineTrainingProperties(args[0]);
      this.model = this.initializeFreshNetworkModel();
      this.initializeModelFromJSONObject(args[1]);
    }
  }

  protected static isFromJSON(opt: any): boolean {
    return FNNModel.has(opt, ['h', 'd'])
      && FNNModel.has(opt.h, ['w', 'b'])
      && FNNModel.has(opt.d, ['w', 'b']);
  }

  protected initializeModelFromJSONObject(json: FNNJson): void {
    this.initializeHiddenLayerFromJSON(json);
    this.model.decoder.Wh = Mat.fromJSON(json.d.w);
    this.model.decoder.b = Mat.fromJSON(json.d.b);
  }

  protected initializeHiddenLayerFromJSON(json: FNNJson): void {
    FNNModel.assert(Array.isArray(json.h.w), 'Wrong JSON Format to recreate Hidden Layer.');
    for (let i = 0; i < json.h.w.length; i++) {
      this.model.hidden.Wh[i] = Mat.fromJSON(json.h.w[i]);
      this.model.hidden.bh[i] = Mat.fromJSON(json.h.b[i]);
    }
  }

  protected static isFreshInstanceCall(opt: any): boolean {
    return FNNModel.has(opt, ['architecture']) && FNNModel.has(opt.architecture, ['inputSize', 'hiddenUnits', 'outputSize']);
  }

  protected initializeModelAsFreshInstance(opt: NetOpts): void {
    this.architecture = this.determineArchitectureProperties(opt);
    this.training = this.determineTrainingProperties(opt);

    const mu = opt['mu'] ? opt['mu'] : 0;
    const std = opt['std'] ? opt['std'] : 0.1;

    this.model = this.initializeFreshNetworkModel();

    this.initializeHiddenLayer(mu, std);

    this.initializeDecoder(mu, std);
  }

  protected determineArchitectureProperties(opt: NetOpts): { inputSize: number, hiddenUnits: Array<number>, outputSize: number } {
    const out = { inputSize: null, hiddenUnits: null, outputSize: null };
    out.inputSize = typeof opt.architecture.inputSize === 'number' ? opt.architecture.inputSize : 1;
    out.hiddenUnits = Array.isArray(opt.architecture.hiddenUnits) ? opt.architecture.hiddenUnits : [1];
    out.outputSize = typeof opt.architecture.outputSize === 'number' ? opt.architecture.outputSize : 1;
    return out;
  }

  protected determineTrainingProperties(opt: NetOpts): { alpha: number, lossClamp: number, loss: number } {
    const out = { alpha: null, lossClamp: null, loss: null };
    if (!opt.training) {
      // patch `opt`
      opt.training = out;
    }

    out.alpha = typeof opt.training.alpha === 'number' ? opt.training.alpha : 0.01;
    out.lossClamp = typeof opt.training.lossClamp === 'number' ? opt.training.lossClamp : 1;
    out.loss = typeof opt.training.loss === 'number' ? opt.training.loss : 1e-6;

    return out;
  }

  protected initializeFreshNetworkModel(): { hidden: { Wh: Array<Mat>; bh: Array<Mat>; }; decoder: { Wh: Mat; b: Mat; }; } {
    return {
      hidden: {
        Wh: new Array<Mat>(this.architecture.hiddenUnits.length),
        bh: new Array<Mat>(this.architecture.hiddenUnits.length)
      },
      decoder: {
        Wh: null,
        b: null
      }
    };
  }

  protected initializeHiddenLayer(mu: number, std: number): void {
    let hiddenSize;
    for (let i = 0; i < this.architecture.hiddenUnits.length; i++) {
      const previousSize = this.getPrecedingLayerSize(i);
      hiddenSize = this.architecture.hiddenUnits[i];
      this.model.hidden.Wh[i] = new RandMat(hiddenSize, previousSize, mu, std);
      this.model.hidden.bh[i] = new Mat(hiddenSize, 1);
    }
  }

  /**
   * According to the given hiddenLayer Index, get the size of preceding layer.
   * @param i current hidden layer index
   */
  private getPrecedingLayerSize(i: number) {
    return i === 0 ? this.architecture.inputSize : this.architecture.hiddenUnits[i - 1];
  }

  protected initializeDecoder(mu: number, std: number): void {
    this.model.decoder.Wh = new RandMat(this.architecture.outputSize, this.architecture.hiddenUnits[this.architecture.hiddenUnits.length - 1], mu, std);
    this.model.decoder.b = new Mat(this.architecture.outputSize, 1);
  }

  /**
   * Sets the neural network into a trainable state.
   * Also cleans the memory of forward pass operations, meaning that the last forward pass cannot be used for backpropagation.
   * @param isTrainable
   */
  public setTrainability(isTrainable: boolean): void {
    this.graph.forgetCurrentSequence();
    this.graph.memorizeOperationSequence(isTrainable);
  }

  /**
   *
   * @param expectedOutput Corresponding target for previous Input of forward-pass
   * @param alpha update factor
   * @returns squared summed loss
   */
  public backward(expectedOutput: Array<number> | Float64Array, alpha?: number): void {
    FNNModel.assert(this.graph.isMemorizingSequence(), '[' + this.constructor.name + '] Trainability is not enabled.');
    FNNModel.assert(typeof this.previousOutput !== 'undefined', '[' + this.constructor.name + '] Please execute `forward()` before calling `backward()`');
    this.propagateLossIntoDecoderLayer(expectedOutput);
    this.backwardGraph();
    this.updateWeights(alpha);
    this.resetGraph();
  }

  private backwardGraph(): void {
    this.graph.backward();
  }

  private resetGraph(): void {
    this.graph.forgetCurrentSequence();
  }

  private propagateLossIntoDecoderLayer(expected: Array<number> | Float64Array): void {
    let loss;
    for (let i = 0; i < this.architecture.outputSize; i++) {
      loss = this.previousOutput.w[i] - expected[i];
      if (Math.abs(loss) <= this.training.loss) {
        continue;
      } else {
        loss = this.clipLoss(loss);
        this.previousOutput.dw[i] = loss;
      }
    }
  }

  private clipLoss(loss: number): number {
    if (loss > this.training.lossClamp) { return this.training.lossClamp; }
    else if (loss < -this.training.lossClamp) { return -this.training.lossClamp; }
    return loss;
  }

  protected updateWeights(alpha?: number): void {
    alpha = alpha ? alpha : this.training.alpha;
    this.updateHiddenLayer(alpha);
    this.updateDecoderLayer(alpha);
  }

  protected updateHiddenLayer(alpha: number): void {
    for (let i = 0; i < this.architecture.hiddenUnits.length; i++) {
      this.model.hidden.Wh[i].update(alpha);
      this.model.hidden.bh[i].update(alpha);
    }
  }

  protected updateDecoderLayer(alpha: number): void {
    this.model.decoder.Wh.update(alpha);
    this.model.decoder.b.update(alpha);
  }

  /**
   * Compute forward pass of Neural Network
   * @param input 1D column vector with observations
   * @param graph optional: inject Graph to append Operations
   * @returns Output of type `Mat`
   */
  public forward(input: Array<number> | Float64Array): Array<number> | Float64Array {
    const mat = this.transformArrayToMat(input);
    const activations = this.specificForwardpass(mat);
    const outputMat = this.computeOutput(activations);
    const output = this.transformMatToArray(outputMat);
    this.previousOutput = outputMat;
    return output;
  }

  private transformArrayToMat(input: Array<number> | Float64Array): Mat {
    const mat = new Mat(this.architecture.inputSize, 1);
    mat.setFrom(input);
    return mat;
  }

  private transformMatToArray(input: Mat): Array<number> | Float64Array {
    const arr = input.w.slice(0);
    return arr;
  }

  protected abstract specificForwardpass(state: Mat): Array<Mat>;

  protected computeOutput(hiddenUnitActivations: Array<Mat>): Mat {
    const weightedInputs = this.graph.mul(this.model.decoder.Wh, hiddenUnitActivations[hiddenUnitActivations.length - 1]);
    return this.graph.add(weightedInputs, this.model.decoder.b);
  }

  public getSquaredLossFor(input: number[] | Float64Array, expectedOutput: number[] | Float64Array): number {
    const trainability = this.graph.isMemorizingSequence();
    this.setTrainability(false);
    const lossSum = this.calculateLossSumByForwardPass(input, expectedOutput);
    this.setTrainability(trainability);
    return lossSum * lossSum;
  }

  private calculateLossSumByForwardPass(input: Array<number> | Float64Array, expected: Array<number> | Float64Array): number {
    let lossSum = 0;
    const actualOutput = this.forward(input);
    for (let i = 0; i < this.architecture.outputSize; i++) {
      const loss = actualOutput[i] - expected[i];
      lossSum += loss;
    }
    return lossSum;
  }

  private static has(obj: any, keys: Array<string>): boolean {
    FNNModel.assert(obj, 'Improper input for DNN.');
    for (const key of keys) {
      if (Object.hasOwnProperty.call(obj, key)) { continue; }
      return false;
    }
    return true;
  }
}
