import { Mat } from "../mat";
import { NetOpts } from "../utils/net-opts";
import { FNNModel } from "./fnn-model";

export class DNN extends FNNModel {

  constructor(...args:
    [opt: NetOpts, json: { hidden: { Wh, bh }, decoder: { Wh: Mat, b: Mat } }] |
    [opt: NetOpts]) {
    super(...args);
  }

  /**
   * Compute forward pass of Neural Network
   * @param state 1D column vector with observations
   * @param graph optional: inject Graph to append Operations
   * @returns Output of type `Mat`
   */
  public specificForwardpass(state: Mat): Mat[] {
    const activations = this.computeHiddenActivations(state);
    return activations;
  }

  protected computeHiddenActivations(state: Mat): Mat[] {
    const hiddenActivations = new Array<Mat>();
    for (let d = 0; d < this.architecture.hiddenUnits.length; d++) {
      const inputVector = d === 0 ? state : hiddenActivations[d - 1];
      const weightedInput = this.graph.mul(this.model.hidden.Wh[d], inputVector);
      const biasedWeightedInput = this.graph.add(weightedInput, this.model.hidden.bh[d]);
      const activation = this.graph.tanh(biasedWeightedInput);
      hiddenActivations.push(activation);
    }
    return hiddenActivations;
  }

  public static toJSON(dnn: DNN): { hidden: { Wh, bh }, decoder: { Wh, b } } {
    const json = { hidden: { Wh: [], bh: [] }, decoder: { Wh: null, b: null } };
    for (let i = 0; i < dnn.model.hidden.Wh.length; i++) {
      json.hidden.Wh[i] = Mat.toJSON(dnn.model.hidden.Wh[i]);
      json.hidden.bh[i] = Mat.toJSON(dnn.model.hidden.bh[i]);
    }
    json.decoder.Wh = Mat.toJSON(dnn.model.decoder.Wh);
    json.decoder.b = Mat.toJSON(dnn.model.decoder.b);
    return json;
  }

  public static fromJSON(initOpt: NetOpts, json: { hidden: { Wh, bh }, decoder: { Wh, b } }): DNN {
    return new DNN(initOpt, json);
  }
}
