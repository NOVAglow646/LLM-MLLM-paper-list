# Code Actions

Here are representative code patterns for acting on images and numbers. They are recommended starting points, not an exhaustive list. You may use other valid Python operations, library calls, or parameter choices as long as they fit the runtime constraints.

## crop
Purpose: Crop a rectangular region from an image using Pillow.

Recommended library: Pillow

Required imports:
from PIL import Image

Input convention:
- Read the source image from get_image("q0") / get_image("g0") or from images[index].
- Use a 4-tuple (left, top, right, bottom) as the crop box.
- The last line should be the cropped PIL image object.
Example:
<tool_call>
img = get_image("q0")
cropped = img.crop((120, 180, 420, 620))
cropped
</tool_call>

## draw_line
Purpose: Draw one or more auxiliary line segments on an image using Pillow.

Recommended library: Pillow

Required imports:
from PIL import ImageDraw

Input convention:
- Read the source image from get_image("q0") / get_image("g0") or from images[index].
- Always call copy() before drawing.
- Use ImageDraw.Draw(annotated).line((x1, y1, x2, y2), fill="red", width=4).
- The last line should be the annotated PIL image object.
Example:
<tool_call>
from PIL import ImageDraw
annotated = get_image("q0").copy()
draw = ImageDraw.Draw(annotated)
draw.line((118, 214, 612, 248), fill="red", width=4)
draw.line((132, 486, 628, 520), fill="red", width=4)
annotated
</tool_call>

## draw_box
Purpose: Draw one or more bounding boxes on an image using Pillow.

Recommended library: Pillow

Required imports:
from PIL import ImageDraw

Input convention:
- Read the source image from get_image("q0") / get_image("g0") or from images[index].
- Always call copy() before drawing.
- Use ImageDraw.Draw(annotated).rectangle((x1, y1, x2, y2), outline="red", width=3).
- The last line should be the annotated PIL image object.
Example:
<tool_call>
from PIL import ImageDraw
annotated = get_image("q0").copy()
draw = ImageDraw.Draw(annotated)
for box in [(42, 38, 128, 124), (150, 38, 236, 124)]:
    draw.rectangle(box, outline="red", width=3)
annotated
</tool_call>

## numeric_calculation
Purpose: Evaluate a mathematical expression using Python and math, instead of mental arithmetic.

Recommended library: math

Required imports:
import math

Input convention:
- Put the expression in a string, such as "5 / 8 * 100" or "math.sqrt(2) * 10".
- Evaluate it safely with eval(expression, {"__builtins__": {}}, {"math": math}).
- The last line should be the numeric result itself.
Example:
<tool_call>
import math
expression = "math.sqrt(2) * 10"
result = eval(expression, {"__builtins__": {}}, {"math": math})
result
</tool_call>

## rotation
Purpose: Rotate an image using Pillow to correct orientation or inspect a different view.

Recommended library: Pillow

Required imports:
from PIL import Image

Input convention:
- Read the source image from get_image("q0") / get_image("g0") or from images[index].
- Use img.rotate(angle, expand=True).
- Positive angles rotate counterclockwise.
- The last line should be the rotated PIL image object.
Example:
<tool_call>
img = get_image("q0")
rotated = img.rotate(90, expand=True)
rotated
</tool_call>

Additional reminders (code-call version):
1. Inside <tool_call>...</tool_call>, you must write real executable Python code.
3. The runtime only guarantees one image-access method: `get_image(ref)`. Original question images use stable ids like `q0`, `q1`, and images returned by earlier code calls use stable ids like `g0`, `g1`.
5. If you need to draw on an image, call copy() first so you do not modify the source image directly. For example: annotated = get_image("q0").copy().
6. If the current call should return an image result, the last line inside <tool_call> should be the PIL image object itself, such as cropped, rotated, or annotated.
7. If the current call should return a numeric result, the last line should be the numeric object itself, such as result.
8. Do not do arithmetic mentally. Use math, and when needed evaluate expressions safely, for example: eval(expression, {"__builtins__": {}}, {"math": math}).
9. <tool_response>...</tool_response> is inserted by the system after the code finishes running. It is not something you should generate yourself. Never output <tool_response> tags on your own.
10. After each code execution returns a result, you must immediately output an <observation>...</observation> tag that explicitly analyzes the returned result before deciding whether to call code again.
11. The order must be strictly <tool_call>...</tool_call> -> <tool_response>...</tool_response> -> <observation>...</observation>. Do not output <observation> before the code result appears.
12. Inside <tool_call>, write code only. Do not add explanations. Do not add Markdown fences. Do not write ```python.
13. Try to make each call perform one clear action at a time, such as crop first, then rotate, then draw a box, then do numeric calculation. This makes the trajectory clearer and more useful for training.
14. When a call returns a new image, the system will tell you which images are currently available and report the new image's stable id. For later turns, prefer stable references such as `img = get_image("g0")` instead of assuming variables persist across calls.
15. If you are unsure about the image size, or worry that your coordinates may be inaccurate, you may first inspect the original question image size, for example by writing `img = get_image("q0")` and then returning `img.size`, before doing crop, draw-line, or draw-box operations.
16. When you give the final answer, you must wrap it inside <answer></answer> tags.

Full example 0:
Question: Is the man phone both blue and closed?
Assistant:
<tool_call>
img = get_image("q0")
img.size
</tool_call>
[The system then inserts a tool result wrapped in <tool_response>...</tool_response>]
<observation>I first check the original question-image size so that later boxes or crops are grounded in the actual canvas size instead of guessed coordinates. Now I can perform a more reliable coordinate-based operation on the original image.</observation>
<tool_call>
from PIL import ImageDraw
annotated = get_image("q0").copy()
draw = ImageDraw.Draw(annotated)
draw.rectangle((600, 290, 660, 380), outline="red", width=3)
annotated
</tool_call>
[The system then inserts a tool result wrapped in <tool_response>...</tool_response>]
<observation>The box covers the phone in the man's hand. The phone is visibly blue, and because he is actively holding it to his ear in use, it is not closed.</observation>
<answer>No</answer>

Full example 1:
Question: What percentage of the 8 tiles in the upside-down board are dark?
Assistant:
<tool_call>
img = get_image("q0")
cropped = img.crop((180, 160, 760, 760))
cropped
</tool_call>
[The system then inserts a tool result wrapped in <tool_response>...</tool_response> and reports the new image_id, for example g0.]
<observation>The crop isolates the board well, but it is still upside down. I should keep using this newly returned image, for example get_image("g0"), and rotate it into a natural orientation before counting the dark tiles.</observation>
<tool_call>
img = get_image("g0")
rotated = img.rotate(180, expand=True)
rotated
</tool_call>
[The system then inserts a tool result wrapped in <tool_response>...</tool_response> and reports the new image_id, for example g1.]
<observation>The board is now upright. I can mark the five dark tiles directly on the image to verify the count.</observation>
<tool_call>
from PIL import ImageDraw
annotated = get_image("g1").copy()
draw = ImageDraw.Draw(annotated)
for box in [(42, 38, 128, 124), (150, 38, 236, 124), (258, 38, 344, 124), (96, 152, 182, 238), (204, 152, 290, 238)]:
    draw.rectangle(box, outline="red", width=3)
annotated
</tool_call>
[The system then inserts a tool result wrapped in <tool_response>...</tool_response>]
<observation>The five boxes align with the dark tiles, so the dark-tile count is 5 and the total tile count is 8. I should compute the percentage exactly.</observation>
<tool_call>
import math
expression = "5 / 8 * 100"
result = eval(expression, {"__builtins__": {}}, {"math": math})
result
</tool_call>
[The system then inserts a tool result wrapped in <tool_response>...</tool_response>]
<observation>The exact percentage is 62.5.</observation>
<answer>62.5% of the tiles are dark.</answer>