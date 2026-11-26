import numpy as np

from hairline.face import BROWS, FACE_OVAL, Face


def fake_face(w=400, h=500):
    """A synthetic, front-facing landmark set: forehead at 30%, brows 40%, chin 80% of height."""
    pts = np.zeros((478, 2), dtype=np.float32)
    pts[:] = (w / 2, h * 0.6)
    for j, i in enumerate(FACE_OVAL):
        pts[i] = (w * 0.3 + (w * 0.4) * (j % 2), h * 0.55)
    for i in BROWS:
        pts[i] = (w / 2, h * 0.40)
    pts[10] = (w / 2, h * 0.30)
    pts[152] = (w / 2, h * 0.80)
    return Face(pts)
