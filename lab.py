#written by GPT-4 based on user instructions but heavily modified by human

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from PIL import Image
import mapping

# ==========================
# Load an example image
# ==========================
mapper = mapping.Mapper()
mapper.take_history = True


img = np.array(Image.open("underwater_images/T_S02951.png").resize((1000, 1000), Image.Resampling.BICUBIC), dtype=np.float32)[2:-2, 2:-2]  # Change to your image
img = np.mean(img, axis=2)
h, w = img.shape[:2]
mapper.resolution = max(h, w)
mapper.x_res = h
mapper.y_res = w

#color map gray
fig, ax = plt.subplots()
ax.imshow(img / 255, cmap='gray')
plt.title("Drag = move | Scroll = scale | Right-drag = rotate | ENTER = crop")
plt.axis('off')
ax.colormap = 'gray'

# ==========================
# Rectangle state variables
# ==========================
rect_size = np.array([w * 0.5, h * 0.5])
center = np.array([w * 0.5, h * 0.5])
angle = 0.0
scale = 1.0

# Main active rectangle
rect = Rectangle((0, 0), rect_size[0], rect_size[1],
                 linewidth=1.5, edgecolor='lime', facecolor='none')
ax.add_patch(rect)

# History tracking
previous_centers = []
previous_rectangles = []

is_dragging = False
is_rotating = False
last_mouse = None


# ==========================
# Helper Functions
# ==========================

def update_rectangle():
    """Apply transform to the active rectangle."""
    trans = Affine2D()
    trans.translate(-rect_size[0] / 2, -rect_size[1] / 2)
    trans.scale(scale, scale)
    trans.rotate_deg_around(0, 0, angle)
    trans.translate(center[0], center[1])

    rect.set_transform(trans + ax.transData)
    fig.canvas.draw_idle()


def draw_saved_rectangle(c, ang, sc):
    """Draw a permanent rectangle (used after saving a crop)."""
    trans = Affine2D()
    trans.scale(sc, sc)
    trans.rotate_deg_around(0, 0, ang)
    trans.translate(c[0], c[1])

    r = Rectangle(
        (-rect_size[0] / 2,
        -rect_size[1] / 2),
        rect_size[0],
        rect_size[1],
        linewidth=1.2,
        edgecolor='yellow',
        facecolor='none',
        alpha=0.7
    )
    r.set_transform(trans + ax.transData)
    ax.add_patch(r)
    previous_rectangles.append(r)


def extract_crop():
    """Extract rotated rectangle from the image."""
    trans = Affine2D()
    trans.scale(scale, scale)
    trans.rotate_deg(angle)
    trans.translate(center[0], center[1])

    xs = np.linspace(-rect_size[0]/2, rect_size[0]/2, w)
    ys = np.linspace(-rect_size[1]/2, rect_size[1]/2, h)
    xv, yv = np.meshgrid(xs, ys)

    coords = np.vstack([xv.flatten(), yv.flatten()]).T
    transformed = trans.transform(coords)
    tx = transformed[:, 0].reshape(xv.shape).astype(int)
    ty = transformed[:, 1].reshape(yv.shape).astype(int)

    crop = np.zeros((xv.shape[0], xv.shape[1]), dtype=np.float32)
    mask = (tx >= 0) & (tx < w) & (ty >= 0) & (ty < h)
    crop[mask] = img[ty[mask], tx[mask]]

    mapper.current_image = crop


# ==========================
# Event Handlers
# ==========================

def on_press(event):
    global is_dragging, is_rotating, last_mouse

    if not event.inaxes:
        return

    last_mouse = np.array([event.xdata, event.ydata])

    if event.button == 1:
        is_dragging = True
    elif event.button == 3:
        is_rotating = True


def on_release(event):
    global is_dragging, is_rotating
    is_dragging = False
    is_rotating = False


def on_motion(event):
    global center, angle, last_mouse

    if not event.inaxes or last_mouse is None:
        return

    current = np.array([event.xdata, event.ydata])

    if is_dragging:
        delta = current - last_mouse
        center += delta

    if is_rotating:
        prev_vec = last_mouse - center
        curr_vec = current - center
        angle += np.degrees(
            np.arctan2(curr_vec[1], curr_vec[0]) -
            np.arctan2(prev_vec[1], prev_vec[0])
        )

    last_mouse = current
    update_rectangle()


def on_scroll(event):
    global scale

    if event.button == 'up':
        scale *= 1.05
    elif event.button == 'down':
        scale /= 1.05

    update_rectangle()


def on_key(event):
    global center, angle, scale

    if event.key == "enter":

        # 1. Save crop
        extract_crop()
        print("GT: ", (center[1] - h / 2) * 2, (center[0] - w/2) *2, 1/scale, -angle)
        mapper.loop_step()
        mapper.add_gt((center[1] - h / 2) * 2, (center[0] - w/2) *2, 1/scale, -angle)



        # 2. Save the rectangle position permanently
        draw_saved_rectangle(center.copy(), angle, scale)

        # 3. Connect with previous if exists
        if len(previous_centers) > 0:
            prev = previous_centers[-1]
            ax.plot(
                [prev[0], center[0]],
                [prev[1], center[1]],
                color='cyan', linewidth=1.5
            )

        # 4. Save center for next line connection
        previous_centers.append(center.copy())

        # 5. Keep active rectangle where it is (user may continue)

        fig.canvas.draw_idle()



# Attach events
fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("button_release_event", on_release)
fig.canvas.mpl_connect("motion_notify_event", on_motion)
fig.canvas.mpl_connect("scroll_event", on_scroll)
fig.canvas.mpl_connect("key_press_event", on_key)

update_rectangle()
extract_crop()

mapper.start()
plt.show()

mapper.export_history("mapping_history.xlsx")
