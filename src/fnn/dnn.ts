import { Mat, MatJson } from "../mat";
import { NetOpts } from "../utils/net-opts";
import { FNNJson, FNNModel } from "./fnn-model";

export class DNN extends FNNModel {

  constructor(...args:
    [opt: NetOpts, json: FNNJson] |
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

  public static toJSON(dnn: DNN): FNNJson {
    const json: FNNJson = { h: { w: [], b: [] }, d: { w: null, b: null } };
    for (let i = 0; i < dnn.model.hidden.Wh.length; i++) {
      json.h.w[i] = Mat.toJSON(dnn.model.hidden.Wh[i]);
      json.h.b[i] = Mat.toJSON(dnn.model.hidden.bh[i]);
    }
    json.d.w = Mat.toJSON(dnn.model.decoder.Wh);
    json.d.b = Mat.toJSON(dnn.model.decoder.b);
    return json;
  }

  public static fromJSON(initOpt: NetOpts, json: FNNJson): DNN {
    return new DNN(initOpt, json);
  }
}
