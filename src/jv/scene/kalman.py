import numpy as np
from typing import Optional
from jv.representation import ObjectCoordData


def kalman(
    object: ObjectCoordData,
    state: Optional[tuple[np.ndarray, np.ndarray]] = None
) -> tuple[ObjectCoordData, tuple[np.ndarray, np.ndarray]]:
    # State transition matrix
    F = np.eye(6)
    F[0, 3] = F[1, 4] = F[2, 5] = 1.0  # dt = 1.0

    # Measurement matrix
    H = np.zeros((3, 6))
    H[0, 0] = H[1, 1] = H[2, 2] = 1.0

    # Covariance matrices
    Q = np.eye(6) * 0.1  # Process noise
    R = np.eye(3) * 5.0  # Measurement noise

    if state is None:
        # Initialize
        X = np.zeros((6, 1))
        X[0, 0], X[1, 0], X[2, 0] = object.x, object.y, object.depth
        P = np.eye(6) * 100.0
    else:
        X, P = state

    # Predict
    X = F @ X
    P = F @ P @ F.T + Q
    
    # Update
    Z = np.array([[object.x], [object.y], [object.depth]])
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    
    X = X + K @ (Z - H @ X)
    P = (np.eye(6) - K @ H) @ P

    # Reassign smoothed values to the current object
    object.x = float(X[0, 0])
    object.y = float(X[1, 0])
    object.depth = float(X[2, 0])

    return object, (X, P)