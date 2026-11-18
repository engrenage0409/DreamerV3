import io
from PIL import Image
from torch.utils.tensorboard import summary as tb_summary
from tensorboard.compat.proto.summary_pb2 import Summary

def windows_make_video(tensor, fps):
    frames = [Image.fromarray(frame) for frame in tensor]
    if not frames:
        raise ValueError("Video contains no frames")

    _, height, width, channels = tensor.shape
    output = io.BytesIO()

    frames[0].save(
        output,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=max(1, round(1000 / fps)),
        loop=0,
    )

    return Summary.Image(
        height=height,
        width=width,
        colorspace=channels,
        encoded_image_string=output.getvalue(),
    )

tb_summary.make_video = windows_make_video
