"""Fast photo checks so a bad photo fails in milliseconds with a message people can act on,
instead of spending a model call on a result that will look wrong."""
from __future__ import annotations

import numpy as np
from PIL import Image

from .face import Face

MIN_FACE_FRACTION = 0.12  # face width / photo width
MAX_YAW = 0.18
MIN_HEADROOM = 0.15  # space above the forehead landmark, in face heights
MIN_BRIGHTNESS = 45  # mean luma of the face box, 0-255


def photo_problem(photo: Image.Image, face: Face) -> str | None:
    w, h = photo.size
    if face.width < MIN_FACE_FRACTION * w:
        return "Your face is too small in this photo. Move closer or crop the photo around your head."
    if abs(face.yaw) > MAX_YAW:
        return "Face the camera straight on so we can see your whole hairline."
    if face.forehead_y < MIN_HEADROOM * face.height:
        return "The top of your head is cut off. Use a photo that shows your full hairline."
    lo, hi = face.oval_x
    box = photo.convert("L").crop((int(max(0, lo)), int(max(0, face.forehead_y)), int(min(w, hi)), int(min(h, face.chin_y))))
    if box.width > 0 and box.height > 0 and float(np.asarray(box).mean()) < MIN_BRIGHTNESS:
        return "This photo is too dark. Try one taken in brighter light."
    return None
