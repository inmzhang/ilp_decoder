import numpy as np
import numpy.typing as npt
import cvxpy as cp
from scipy.sparse import csc_matrix
import stim

from ilp_decoder.utils import DemMatrices, detector_error_model_to_check_matrices


class ILPDecoder:
    def __init__(
        self,
        check_matrix: csc_matrix,
        obs_matrix: csc_matrix,
        error_probs: npt.NDArray[np.float_],
        solver: str = "HIGHS",
        **kwargs,
    ) -> None:
        """Construct a ILPDecoder object from a detector check matrix, observable matrix,
        and error probabilities vector.

        Parameters
        ----------
        check_matrix : csc_matrix
            The detector check matrix of the code. This is a sparse matrix in Compressed Sparse Column format.
            It should have shape `(num_detectors, num_errors)` with dtype `np.uint8`.
        obs_matrix : csc_matrix
            The observable matrix of the code. This is a sparse matrix in Compressed Sparse Column format.
            It should have shape `(num_observables, num_errors)` with dtype `np.uint8`.
        error_probs : np.ndarray
            A 1D numpy array of length `num_errors` containing the error probabilities for each error.

        solver : the solver to use for the ILP. The default is "HIGHS".
        kwargs
            Additional keyword arguments are passed to `cvxpy.Problem.solve()`. See
            https://www.cvxpy.org/tutorial/solvers/index.html for more information.
        """
        self._solver = solver
        self._solve_kwargs = kwargs

        num_detectors = check_matrix.shape[0]
        self._num_detectors = num_detectors
        self._obs_matrix = obs_matrix

        num_errors: int = check_matrix.shape[1]

        # Define decision variables:
        #   - self._errors: binary indicator for each error.
        #   - slack_vars: integer slack variables for each detector row.
        self._errors = cp.Variable(num_errors, boolean=True)
        dets = cp.Variable(num_detectors, integer=True)

        # Define a parameter for the syndrome to be updated during decoding.
        self._syndromes = cp.Parameter(num_detectors, boolean=True)

        # Set up the parity constraints for each detector.
        # For each detector d, the parity constraint is:
        #     (sum of errors affecting detector d) + 2 * slack_vars[d] == syndrome[d]
        constraints = [
            (
                cp.sum([self._errors[col] for col in check_matrix.getrow(d).indices])
                + 2 * dets[d]
                == self._syndromes[d]
            )
            for d in range(num_detectors)
        ]

        # Define the objective using error likelihoods.
        # The weight for each error is computed as log((1 - p) / p) where p is the error probability.
        error_wts = np.array([np.log((1 - p) / p) for p in error_probs])
        objective = cp.Minimize(error_wts @ self._errors)
        self._problem = cp.Problem(objective, constraints)  # type: ignore

    @staticmethod
    def from_detector_error_model(
        dem: stim.DetectorErrorModel,
        *,
        solver: str = "HIGHS",
        **kwargs,
    ) -> "ILPDecoder":
        """Construct a ILPDecoder object from a stim.DetectorErrorModel.

        Parameters
        ----------
        dem : stim.DetectorErrorModel used to construct the decoder.
        solver : the solver to use for the ILP. The default is "HIGHS".
        kwargs
            Additional keyword arguments are passed to `cvxpy.Problem.solve()`. See
            https://www.cvxpy.org/tutorial/solvers/index.html for more information.

        Returns
        -------
        ILPDecoder
            An ILPDecoder object initialized with the provided stim.DetectorErrorModel.
        """
        dem_matrices: DemMatrices = detector_error_model_to_check_matrices(
            dem, allow_undecomposed_hyperedges=True
        )
        return ILPDecoder(
            check_matrix=dem_matrices.check_matrix,
            obs_matrix=dem_matrices.observables_matrix,
            error_probs=dem_matrices.priors,
            solver=solver,
            **kwargs,
        )

    @staticmethod
    def from_circuit(
        circuit: stim.Circuit,
        *,
        solver: str = "HIGHS",
        **kwargs,
    ) -> "ILPDecoder":
        """Construct a ILPDecoder object from a stim.Circuit.

        Parameters
        ----------
        circuit : stim.Circuit used to construct the decoder.
        solver : the solver to use for the ILP. The default is "HIGHS".
        kwargs
            Additional keyword arguments are passed to `cvxpy.Problem.solve()`. See
            https://www.cvxpy.org/tutorial/solvers/index.html for more information.

        Returns
        -------
        ILPDecoder
            An ILPDecoder object initialized with the provided stim.Circuit.
        """
        dem = circuit.detector_error_model(decompose_errors=False)
        return ILPDecoder.from_detector_error_model(dem, solver=solver, **kwargs)

    def decode(self, syndrome: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
        """
        Decode the syndrome and return a prediction of which observables were flipped

        Parameters
        ----------
        syndrome : np.ndarray
            A single shot of syndrome data. This should be a binary array with a length equal to the
            number of detectors in the `stim.Circuit` or `stim.DetectorErrorModel`. E.g. the syndrome might be
            one row of shot data sampled from a `stim.CompiledDetectorSampler`.

        Returns
        -------
        np.ndarray
            A binary numpy array `predictions` which predicts which observables were flipped.
            Its length is equal to the number of observables in the `stim.Circuit` or `stim.DetectorErrorModel`.
            `predictions[i]` is 1 if the decoder predicts observable `i` was flipped and 0 otherwise.
        """
        decoded_errors = self.decode_return_errors(syndrome)
        predicted_obs = (self._obs_matrix @ decoded_errors) % 2
        return predicted_obs.astype(np.bool_)

    def decode_return_errors(
        self, syndrome: npt.NDArray[np.bool_]
    ) -> npt.NDArray[np.uint8]:
        """
        Decode the syndrome and return a the predicted errors.

        Parameters
        ----------
        syndrome : np.ndarray
            A single shot of syndrome data. This should be a binary array with a length equal to the
            number of detectors in the `stim.Circuit` or `stim.DetectorErrorModel`. E.g. the syndrome might be
            one row of shot data sampled from a `stim.CompiledDetectorSampler`.

        Returns
        -------
        np.ndarray
            A numpy array `predictions` which predicts which errors were flipped.
            Its length is equal to the number of errors in the `stim.Circuit` or `stim.DetectorErrorModel`.
            `predictions[i]` is 1 if the decoder predicts error `i` was flipped and 0 otherwise.
        """
        # Set the syndrome parameter.
        self._syndromes.value = syndrome

        # Solve the ILP problem with the specified solver.
        self._problem.solve(solver=self._solver, **self._solve_kwargs)

        # Ensure that an optimal solution is found.
        if self._problem.status != cp.OPTIMAL:
            raise ValueError(
                f"ILP did not solve optimally. Status: {self._problem.status}"
            )
        decoded_errors = np.array(np.round(self._errors.value), dtype=np.uint8)  # type: ignore
        return decoded_errors

    def decode_batch(
        self,
        shots: np.ndarray,
        *,
        bit_packed_shots: bool = False,
        bit_packed_predictions: bool = False,
    ) -> np.ndarray:
        """
        Decode a batch of shots of syndrome data. This is just a helper method, equivalent to iterating over each
        shot and calling `ILPDecoder.decode` on it.

        Parameters
        ----------
        shots : np.ndarray
            A binary numpy array of dtype `np.uint8` or `bool` with shape `(num_shots, num_detectors)`, where
            here `num_shots` is the number of shots and `num_detectors` is the number of detectors in the `stim.Circuit` or `stim.DetectorErrorModel`.

        bit_packed_shots : whether the input shots are bit-packed or not. If True, the input
            `shots` should be a 2D numpy array of dtype `np.uint8` with shape `(num_shots, (num_detectors + 7) // 8)`.
            Otherwise, it should be a 2D numpy array of dtype `bool` with shape `(num_shots, num_detectors)`.
        bit_packed_predictions : whether to return the predictions in bit-packed format or not. If True, the
            output predictions will be a 2D numpy array of dtype `np.uint8` with shape
            `(num_shots, (num_observables + 7) // 8)`. Otherwise, the output predictions will be a 2D numpy array
            of dtype `bool` with shape `(num_shots, num_observables)`.

        Returns
        -------
        np.ndarray
            If `bit_packed_predictions` is True, a 2D numpy array of dtype `np.uint8` with shape
            `(num_shots, num_observables + 7 // 8)` is returned. Otherwise, a 2D numpy array of dtype
            `bool` with shape `(num_shots, num_observables)` is returned.
        """
        if bit_packed_shots:
            shots = np.unpackbits(shots, axis=1, bitorder="little")[
                :, : self._num_detectors
            ]
        shots = shots.astype(np.bool_)
        predictions = np.zeros((shots.shape[0], self._obs_matrix.shape[0]), dtype=bool)
        for i in range(shots.shape[0]):
            predictions[i, :] = self.decode(shots[i, :])
        if bit_packed_predictions:
            predictions = np.packbits(predictions, axis=1, bitorder="little")
        return predictions
