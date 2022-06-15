import { Mat } from "../mat";

/**
 * State of inner activations
 */
export interface InnerState {
  hiddenActivationState: Array<Mat>;
  output: Mat;
  cells?: Array<Mat>;
}
