import { Graph } from "../graph";
import { Mat, MatJson } from "../mat";
import { RandMat } from "../rand-mat";
import { Assertable } from "../utils/assertable";
import { InnerState } from "../utils/inner-state";
import { NetOpts } from "../utils/net-opts";

export abstract class RNNModel extends Assertable {

  protected architecture: { inputSize: number, hiddenUnits: Array<number>, outputSize: number };

  public model: { hidden: any, decoder: { Wh: Mat, b: Mat } };

  protected graph: Graph;

  /**
   * Generates a Neural Net instance from a pre-trained Neural Net JSON.
   * @param {{ hidden: any, decoder: { Wh: Mat, b: Mat } }} opt Specs of the Neural Net.
   */
  constructor(opt: { hidden: any, decoder: { Wh: Mat, b: Mat } });
  /**
   * Generates a Neural Net with given specs.
   * @param {NetOpts} opt Specs of the Neural Net.
   */
  constructor(opt: NetOpts);
  constructor(opt: any) {
    super();
    const needsBackpropagation = opt && opt.needsBackpropagation ? opt.needsBackpropagation : true;

    this.graph = new Graph();
    this.graph.memorizeOperationSequence(true);

    if (this.isFromJSON(opt)) {
      this.initializeModelFromJSONObject(opt);
    } else if (this.isFreshInstanceCall(opt)) {
      this.initializeModelAsFreshInstance(opt);
    } else {
      RNNModel.assert(false, 'Improper input for DNN.');
    }
  }

  protected abstract isFromJSON(opt: any): boolean;

  protected initializeModelFromJSONObject(opt: { hidden: any, decoder: { Wh: MatJson, b: MatJson } }): void {
    this.initializeHiddenLayerFromJSON(opt);
    this.model.decoder.Wh = Mat.fromJSON(opt['decoder']['Wh']);
    this.model.decoder.b = Mat.fromJSON(opt['decoder']['b']);
  }

  protected abstract initializeHiddenLayerFromJSON(opt: { hidden: any, decoder: { Wh: MatJson, b: MatJson } }): void;

  private isFreshInstanceCall(opt: NetOpts): boolean {
    return RNNModel.has(opt, ['architecture']) && RNNModel.has(opt.architecture, ['inputSize', 'hiddenUnits', 'outputSize']);
  }

  private initializeModelAsFreshInstance(opt: NetOpts): void {
    this.architecture = opt.architecture;

    const mu = opt['mu'] ? opt['mu'] : 0;
    const std = opt['std'] ? opt['std'] : 0.01;

    this.model = this.initializeNetworkModel();

    this.initializeHiddenLayer(mu, std);

    this.initializeDecoder(mu, std);
  }

  protected abstract initializeNetworkModel(): { hidden: any; decoder: { Wh: Mat; b: Mat; }; };

  protected abstract initializeHiddenLayer(mu: number, std: number): void;

  protected initializeDecoder(mu: number, std: number): void {
    this.model.decoder.Wh = new RandMat(this.architecture.outputSize, this.architecture.hiddenUnits[this.architecture.hiddenUnits.length - 1], mu, std);
    this.model.decoder.b = new Mat(this.architecture.outputSize, 1);
  }

  public abstract forward(input: Mat, previousActivationState?: InnerState, graph?: Graph): InnerState;

  /**
   * Updates all weights depending on their specific gradients
   * @param alpha discount factor for weight updates
   * @returns {void}
   */
  public update(alpha: number): void {
    this.updateHiddenUnits(alpha);
    this.updateDecoder(alpha);
  }

  protected abstract updateHiddenUnits(alpha: number): void;
  protected abstract updateDecoder(alpha: number): void;

  protected computeOutput(hiddenActivations: Mat[], graph: Graph): Mat {
    const precedingHiddenLayerActivations = hiddenActivations[hiddenActivations.length - 1];
    const weightedInputs = graph.mul(this.model.decoder.Wh, precedingHiddenLayerActivations);
    return graph.add(weightedInputs, this.model.decoder.b);
  }

  protected static has(obj: any, keys: Array<string>): boolean {
    RNNModel.assert(obj, '[class:rnn-model] improper input for instantiation');
    for (const key of keys) {
      if (Object.hasOwnProperty.call(obj, key)) { continue; }
      return false;
    }
    return true;
  }
}
