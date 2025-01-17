import { Graph } from "../graph";
import { Mat, MatJson } from "../mat";
import { RandMat } from "../rand-mat";
import { InnerState } from "../utils/inner-state";
import { NetOpts } from "../utils/net-opts";
import { RNNModel } from "./rnn-model";

export class RNN extends RNNModel {
  /**
   * Generates a Neural Net instance from a pre-trained Neural Net JSON.
   * @param {{ hidden: { Wh, Wx, bh }, decoder: { Wh, b } }} opt Specs of the Neural Net.
   */
  constructor(opt: { hidden: { Wh, Wx, bh }, decoder: { Wh, b } });
  /**
   * Generates a Neural Net with given specs.
   * @param {NetOpts} opt Specs of the Neural Net. [defaults to: needsBackprop = true, mu = 0, std = 0.01]
   */
  constructor(opt: NetOpts);
  constructor(opt: any) {
    super(opt);
  }

  protected isFromJSON(opt: any): boolean {
    return RNNModel.has(opt, ['hidden', 'decoder'])
      && RNNModel.has(opt.hidden, ['Wh', 'Wx', 'bh'])
      && RNNModel.has(opt.decoder, ['Wh', 'b']);
  }

  protected initializeHiddenLayerFromJSON(opt: { hidden: { Wh: MatJson[], Wx: MatJson[], bh: MatJson[] }, decoder: { Wh: MatJson, b: MatJson } }): void {
    RNNModel.assert(!Array.isArray(opt['hidden']['Wh']), 'Wrong JSON Format to recreate Hidden Layer.');
    RNNModel.assert(!Array.isArray(opt['hidden']['Wx']), 'Wrong JSON Format to recreate Hidden Layer.');
    RNNModel.assert(!Array.isArray(opt['hidden']['bh']), 'Wrong JSON Format to recreate Hidden Layer.');
    for (let i = 0; i < opt.hidden.Wh.length; i++) {
      this.model.hidden.Wx[i] = Mat.fromJSON(opt.hidden.Wx[i]);
      this.model.hidden.Wh[i] = Mat.fromJSON(opt.hidden.Wh[i]);
      this.model.hidden.bh[i] = Mat.fromJSON(opt.hidden.bh[i]);
    }
  }

  protected initializeNetworkModel(): { hidden: any; decoder: { Wh: Mat; b: Mat; }; } {
    return {
      hidden: {
        Wx: new Array<Mat>(this.architecture.hiddenUnits.length),
        Wh: new Array<Mat>(this.architecture.hiddenUnits.length),
        bh: new Array<Mat>(this.architecture.hiddenUnits.length)
      },
      decoder: {
        Wh: null,
        b: null
      }
    };
  }

  protected initializeHiddenLayer(): void {
    let hiddenSize;
    for (let i = 0; i < this.architecture.hiddenUnits.length; i++) {
      const previousSize = i === 0 ? this.architecture.inputSize : this.architecture.hiddenUnits[i - 1];
      hiddenSize = this.architecture.hiddenUnits[i];
      this.model.hidden.Wx[i] = new RandMat(hiddenSize, previousSize, 0, 0.08);
      this.model.hidden.Wh[i] = new RandMat(hiddenSize, hiddenSize, 0, 0.08);
      this.model.hidden.bh[i] = new Mat(hiddenSize, 1);
    }
  }

  /**
   * Forward pass for a single tick of Neural Network
   * @param input 1D column vector with observations
   * @param previousActivationState Structure containing hidden representation ['h'] of type `Mat[]` from previous iteration
   * @param graph optional: inject Graph to append Operations
   * @returns Structure containing hidden representation ['h'] of type `Mat[]` and output ['output'] of type `Mat`
   */
  forward(input: Mat, previousActivationState?: InnerState, graph?: Graph): InnerState {
    previousActivationState = previousActivationState ? previousActivationState : null;
    graph = graph ? graph : this.graph;

    const previousHiddenActivations = this.getPreviousHiddenActivationsFrom(previousActivationState);

    const hiddenActivations = this.computeHiddenActivations(input, previousHiddenActivations, graph);

    const output = this.computeOutput(hiddenActivations, graph);

    // return hidden representation and output
    return { 'hiddenActivationState': hiddenActivations, 'output': output };
  }

  private getPreviousHiddenActivationsFrom(previousActivationState: InnerState): Mat[] {
    let previousHiddenActivations;
    if (this.givenPreviousActivationState(previousActivationState)) {
      previousHiddenActivations = previousActivationState.hiddenActivationState;
    } else {
      previousHiddenActivations = new Array<Mat>();
      for (let i = 0; i < this.architecture.hiddenUnits.length; i++) {
        previousHiddenActivations.push(new Mat(this.architecture.hiddenUnits[i], 1));
      }
    }
    return previousHiddenActivations;
  }

  private givenPreviousActivationState(previousActivationState: InnerState) {
    return previousActivationState && typeof previousActivationState.hiddenActivationState !== 'undefined';
  }

  private computeHiddenActivations(input: Mat, previousHiddenActivations: Mat[], graph: Graph): Mat[] {
    const hiddenActivations = new Array<Mat>();
    for (let i = 0; i < this.architecture.hiddenUnits.length; i++) {
      const inputVector = i === 0 ? input : hiddenActivations[i - 1];
      const previousActivations = previousHiddenActivations[i];
      const weightedStatelessInputPortion = graph.mul(this.model.hidden.Wx[i], inputVector);
      const weightedStatefulInputPortion = graph.mul(this.model.hidden.Wh[i], previousActivations);
      const activation = graph.relu(graph.add(graph.add(weightedStatelessInputPortion, weightedStatefulInputPortion), this.model.hidden.bh[i]));
      hiddenActivations.push(activation);
    }
    return hiddenActivations;
  }

  protected updateHiddenUnits(alpha: number): void {
    for (let i = 0; i < this.architecture.hiddenUnits.length; i++) {
      this.model.hidden.Wx[i].update(alpha);
      this.model.hidden.Wh[i].update(alpha);
      this.model.hidden.bh[i].update(alpha);
    }
  }

  protected updateDecoder(alpha: number): void {
    this.model.decoder.Wh.update(alpha);
    this.model.decoder.b.update(alpha);
  }
}
