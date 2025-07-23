import numpy as np

def matrix_to_quaternion(R):
    """
    Converts a 3x3 rotation matrix to a quaternion [w, x, y, z].
    """
    m = R
    trace = np.trace(m)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (m[2,1] - m[1,2]) / s
        y = (m[0,2] - m[2,0]) / s
        z = (m[1,0] - m[0,1]) / s
    else:
        if (m[0,0] > m[1,1]) and (m[0,0] > m[2,2]):
            s = np.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 2
            w = (m[2,1] - m[1,2]) / s
            x = 0.25 * s
            y = (m[0,1] + m[1,0]) / s
            z = (m[0,2] + m[2,0]) / s
        elif m[1,1] > m[2,2]:
            s = np.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2
            w = (m[0,2] - m[2,0]) / s
            x = (m[0,1] + m[1,0]) / s
            y = 0.25 * s
            z = (m[1,2] + m[2,1]) / s
        else:
            s = np.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2
            w = (m[1,0] - m[0,1]) / s
            x = (m[0,2] + m[2,0]) / s
            y = (m[1,2] + m[2,1]) / s
            z = 0.25 * s
    return np.array([w, x, y, z])

def quat_multiply(q1, q2):
    """
    Hamilton product of two quaternions (w, x, y, z).
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def rt2dq(R, t):
    """
    Converts rotation matrix (R: 3x3) and translation (t: 3,) 
    to dual quaternion (real and dual part, both shape (4,))
    """
    q_r = matrix_to_quaternion(R)
    t_quat = np.concatenate(([0.0], t))
    q_d = 0.5 * quat_multiply(t_quat, q_r)
    return q_r, q_d


if __name__ == "__main__":
    R = np.eye(3)
    t = np.array([1.0, 2.0, 3.0])
    q_r, q_d = rt2dq(R, t)
    print("q_r:", q_r)
    print("q_d:", q_d)